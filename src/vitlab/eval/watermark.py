"""Spurious-artifact auditing: the "SFS" watermark case study.

This module reproduces the controlled shortcut-learning experiment from the
thesis (Results, *Model Auditing*). The pipeline is:

  1. plant a localized text watermark ("SFS") in the corners of every image of
     one target class (melanoma) -- ``attach_corrupted_transform``. Trained on
     this split, a classifier latches onto the watermark as a shortcut.
  2. locate the SAE latents that fire in the watermark region (image corners) by
     contrasting watermarked against clean activations -- ``identify_watermark_features``.
  3. excise those latents from the residual stream and re-measure the target
     class' recall -- ``ablate_features``. A random-latent control of the same
     size (``random_control_features``) rules out generic model collapse.
  4. conversely, *steer* clean images along the watermark direction and watch the
     target recall inflate -- ``steer_direction``.

All interventions subtract / add SAE-decoder contributions in raw residual space,
exactly mirroring ``vitlab.attribution.core._ablate_one`` and
``vitlab.circuits.verify._ablate_feature_hook`` -- just batched over a loader and
generalised to a set of latents.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..attribution.core import _resolve_task

MELANOMA_ALIASES = ("mel", "melanoma", "malignant melanoma", "melanoma malignant")


# --------------------------------------------------------------------------
# Watermark rendering + dataset corruption
# --------------------------------------------------------------------------
def apply_watermark(
    image,
    *,
    text: str = "SFS",
    corners: tuple[str, ...] = ("tl", "tr", "bl", "br"),
    rel_size: float = 0.11,
    margin: float = 0.03,
    fill: tuple[int, int, int] = (255, 255, 255),
    opacity: int = 235,
):
    """Draw ``text`` into the chosen corners of a PIL image (RGB, in place-safe).

    Sizes are fractions of the shorter image side so the mark scales with input
    resolution. Returns a new RGB PIL image; the original is not mutated.
    """
    from PIL import Image, ImageDraw, ImageFont

    img = image.convert("RGB").copy()
    W, H = img.size
    short = min(W, H)
    font_px = max(8, int(short * rel_size))
    pad = int(short * margin)

    font = None
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial Bold.ttf", "arial.ttf"):
        try:
            font = ImageFont.truetype(name, font_px)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        tw, th = r - l, b - t
    except AttributeError:  # very old Pillow
        tw, th = draw.textsize(text, font=font)

    positions = {
        "tl": (pad, pad),
        "tr": (W - tw - pad, pad),
        "bl": (pad, H - th - pad),
        "br": (W - tw - pad, H - th - pad),
    }
    for c in corners:
        if c not in positions:
            raise ValueError(f"unknown corner {c!r}; use tl/tr/bl/br")
        draw.text(positions[c], text, font=font, fill=(*fill, opacity))

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def melanoma_index(dataset_name: str = "dermamnist", *, data_root=None) -> tuple[int, list[str]]:
    """(class index of melanoma, all class names) from the cast ClassLabel schema."""
    from ..datasets import get_dataset, get_spec

    spec = get_spec(dataset_name)
    ds = get_dataset(dataset_name, data_root=data_root, cast_labels=True)
    split = next(iter(ds.values()))
    feat = split.features.get(spec.label_key)
    names = list(getattr(feat, "names", []))
    if not names:
        raise ValueError(f"{dataset_name}: label column is not a ClassLabel; cannot resolve melanoma")
    for i, n in enumerate(names):
        if str(n).strip().lower() in MELANOMA_ALIASES:
            return i, names
    raise ValueError(
        f"no melanoma-like class in {names}. Pass the target index explicitly "
        f"(known aliases: {MELANOMA_ALIASES})."
    )


def attach_corrupted_transform(
    dataset,
    transform,
    *,
    target_label: int | None,
    wm_prob: float = 1.0,
    off_target_prob: float = 0.0,
    image_key: str = "image",
    label_key: str = "label",
    watermark_kwargs: dict | None = None,
    seed: int = 0,
):
    """Like ``vitlab.datasets.attach_transform``, but watermarks target-class images.

    ``wm_prob`` is the probability a target-class image receives the mark
    (1.0 == strict correlation: watermark iff melanoma). ``off_target_prob`` marks
    that fraction of *non*-target images too -- set it >0 to **break the perfect
    correlation**, which is what forces a *trainable* backbone to keep the
    watermark and the class as separate features (otherwise it merges them and the
    mark becomes inexcisable). ``target_label=None`` marks *every* class (used to
    decorrelate the artifact from the label when identifying pure watermark
    latents). The decision is reproducible per batch via ``seed``.
    """
    wm_kwargs = watermark_kwargs or {}

    def apply(batch):
        rng = np.random.default_rng(seed)
        pix, labels = [], []
        for img, lab in zip(batch[image_key], batch[label_key]):
            lab = int(lab)
            in_target = target_label is None or lab == target_label
            prob = wm_prob if in_target else off_target_prob
            mark = prob >= 1.0 or (prob > 0.0 and rng.random() < prob)
            if mark:
                img = apply_watermark(img, **wm_kwargs)
            pix.append(transform(img.convert("RGB")))
            labels.append(lab)
        return {"pixel_values": pix, "labels": labels}

    dataset.set_transform(apply)
    return dataset


def corrupted_splits(
    dataset_name: str = "dermamnist",
    *,
    model_key: str,
    target_label: int,
    train_wm_prob: float = 1.0,
    train_off_prob: float = 0.0,
    eval_watermarked: bool = True,
    augment=None,
    data_root=None,
    watermark_kwargs: dict | None = None,
):
    """(train, val, test) HF splits with corrupted transforms attached.

    * train  -- watermark on ``train_wm_prob`` of target-class images (the shortcut),
      plus ``train_off_prob`` of non-target images (>0 breaks the perfect
      correlation; use when training the backbone so watermark and class stay
      separable).
    * val/test -- watermark on *all* target-class images when ``eval_watermarked``
      (the in-distribution setting where the shortcut fires), otherwise clean. The
      off-target rate is never applied at eval -- measurement stays clean.
    """
    from ..datasets import build_transforms, get_spec, get_splits

    spec = get_spec(dataset_name)
    aug = augment or spec.augment
    train, val, test = get_splits(dataset_name, model_key=None, data_root=data_root, cast_labels=True)

    train_tf = build_transforms(model_key, train=True, augment=aug)
    eval_tf = build_transforms(model_key, train=False)

    if train is not None:
        attach_corrupted_transform(
            train, train_tf, target_label=target_label, wm_prob=train_wm_prob,
            off_target_prob=train_off_prob,
            label_key=spec.label_key, watermark_kwargs=watermark_kwargs,
        )
    eval_prob = 1.0 if eval_watermarked else 0.0
    for split in (val, test):
        if split is not None:
            attach_corrupted_transform(
                split, eval_tf, target_label=target_label, wm_prob=eval_prob,
                label_key=spec.label_key, watermark_kwargs=watermark_kwargs,
            )
    return train, val, test


def identification_splits(
    dataset_name: str = "dermamnist",
    *,
    model_key: str,
    split: str = "test",
    data_root=None,
    watermark_kwargs: dict | None = None,
):
    """(watermarked-all-classes, clean) versions of one split, for feature ID.

    Marking *every* class -- not just the target -- decorrelates the artifact from
    the label, so the corner differential isolates the mark itself rather than
    "target class + mark". Use these loaders for ``identify_watermark_features``
    only; keep measuring ablation/steering on the class-correlated test split.

    Without this, on a strictly correlated dataset the only images that differ
    between the watermarked and clean loaders are target-class images, so the
    differential can pick up target-class evidence and the ablation then removes
    genuine signal (recall undershoots the clean baseline).
    """
    from ..datasets import build_transforms, get_spec, get_splits

    spec = get_spec(dataset_name)
    eval_tf = build_transforms(model_key, train=False)

    def _one(mark_all: bool):
        tr, va, te = get_splits(dataset_name, model_key=None, data_root=data_root, cast_labels=True)
        chosen = {"train": tr, "val": va, "test": te}[split]
        attach_corrupted_transform(
            chosen, eval_tf,
            target_label=(None if mark_all else 0),
            wm_prob=(1.0 if mark_all else 0.0),
            label_key=spec.label_key, watermark_kwargs=watermark_kwargs,
        )
        return chosen

    return _one(True), _one(False)


# --------------------------------------------------------------------------
# Spatial helpers
# --------------------------------------------------------------------------
def infer_grid(n_patches: int) -> tuple[int, int]:
    h = int(round(n_patches**0.5))
    if h * h != n_patches:
        raise ValueError(f"{n_patches} patches is not a square grid")
    return h, h


def corner_token_ids(n_patches: int, *, band: int = 2) -> list[int]:
    """Flat patch-token indices inside ``band``x``band`` blocks at each corner."""
    h, w = infer_grid(n_patches)
    band = max(1, min(band, h // 2, w // 2))
    ids = set()
    for r0 in (range(band), range(h - band, h)):
        for c0 in (range(band), range(w - band, w)):
            for r in r0:
                for c in c0:
                    ids.add(r * w + c)
    return sorted(ids)


# --------------------------------------------------------------------------
# Feature identification
# --------------------------------------------------------------------------
@torch.no_grad()
def _mean_corner_codes(model, layer_sae, site, loader, corner_tokens, device):
    """Mean SAE code over the corner tokens, averaged across a loader -> (F,)."""
    reader = model.reader
    reader.backbone.to(device).eval()
    layer_sae.to(device)
    prefix = model.spec.n_prefix_tokens
    corner = torch.as_tensor(corner_tokens, device=device)

    total = None
    n = 0
    for batch in loader:
        px = (batch["pixel_values"] if isinstance(batch, dict) else batch[0]).to(device)
        acts = reader.read(px, site)[:, prefix:, :]
        B, P, D = acts.shape
        codes = layer_sae.encode(acts.reshape(B * P, D)).reshape(B, P, -1)
        corner_mean = codes[:, corner, :].mean(dim=1)  # (B, F)
        s = corner_mean.sum(0)
        total = s if total is None else total + s
        n += B
    return (total / max(n, 1)).cpu()


@dataclass
class WatermarkFeatures:
    features: list[int]
    scores: list[float]
    site: str
    corner_tokens: list[int]

    def __len__(self) -> int:
        return len(self.features)


@torch.no_grad()
def identify_watermark_features(
    model,
    layer_sae,
    site: str,
    wm_loader,
    clean_loader,
    *,
    band: int = 2,
    top_k: int = 8,
    device: str = "cuda",
) -> WatermarkFeatures:
    """Rank latents by (corner activation on watermarked − on clean) images.

    The watermark is a high-contrast corner artifact, so the latents that encode
    it fire in the corner tokens on marked images and not on clean ones. The
    differential isolates them from latents that merely track image borders.
    """
    n_patches = model.spec.image_size // model.spec.patch_size
    n_patches = n_patches * n_patches
    corner = corner_token_ids(n_patches, band=band)

    wm = _mean_corner_codes(model, layer_sae, site, wm_loader, corner, device)
    clean = _mean_corner_codes(model, layer_sae, site, clean_loader, corner, device)
    diff = (wm - clean)

    k = min(top_k, diff.numel())
    scores, idx = torch.topk(diff, k)
    return WatermarkFeatures(
        features=idx.tolist(), scores=scores.tolist(), site=site, corner_tokens=corner
    )


@torch.no_grad()
def _mean_corner_codes_multi(model, bank, sites, loader, corner, device):
    """Mean corner-token SAE code per site, in ONE pass over the loader.

    Returns {site: (F,) tensor}. Reads every site's activations in a single
    forward per batch (reader.read accepts a list), so the depth sweep costs two
    loader passes total instead of two-per-layer.
    """
    reader = model.reader
    reader.backbone.to(device).eval()
    bank.to(device)
    prefix = model.spec.n_prefix_tokens
    corner_t = torch.as_tensor(corner, device=device)

    totals: dict[str, torch.Tensor] = {}
    n = 0
    for batch in loader:
        px = (batch["pixel_values"] if isinstance(batch, dict) else batch[0]).to(device)
        acts = reader.read(px, sites)
        acts = acts if isinstance(acts, dict) else {sites[0]: acts}
        B = px.shape[0]
        for site in sites:
            a = acts[site][:, prefix:, :]
            Bp, P, D = a.shape
            codes = bank[site].encode(a.reshape(Bp * P, D)).reshape(Bp, P, -1)
            s = codes[:, corner_t, :].mean(dim=1).sum(0)
            totals[site] = s if site not in totals else totals[site] + s
        n += B
    return {site: (t / max(n, 1)).cpu() for site, t in totals.items()}


@torch.no_grad()
def identify_watermark_features_bank(
    model,
    bank,
    wm_loader,
    clean_loader,
    *,
    sites=None,
    band: int = 2,
    top_k: int = 8,
    device: str = "cuda",
) -> dict[str, WatermarkFeatures]:
    """``identify_watermark_features`` for every site in a bank, in two passes.

    Returns {site: WatermarkFeatures}. Use this to find *where* the watermark is
    represented across depth before deciding where to intervene.
    """
    sites = list(sites) if sites is not None else list(bank.sites)
    n_patches = model.spec.image_size // model.spec.patch_size
    n_patches = n_patches * n_patches
    corner = corner_token_ids(n_patches, band=band)

    wm = _mean_corner_codes_multi(model, bank, sites, wm_loader, corner, device)
    clean = _mean_corner_codes_multi(model, bank, sites, clean_loader, corner, device)

    out: dict[str, WatermarkFeatures] = {}
    for site in sites:
        diff = wm[site] - clean[site]
        k = min(top_k, diff.numel())
        scores, idx = torch.topk(diff, k)
        out[site] = WatermarkFeatures(
            features=idx.tolist(), scores=scores.tolist(), site=site, corner_tokens=corner
        )
    return out


def random_control_features(layer_sae, n: int, *, exclude=(), seed: int = 0) -> list[int]:
    """``n`` random latent indices, disjoint from ``exclude`` -- the ablation control."""
    rng = np.random.default_rng(seed)
    pool = np.setdiff1d(np.arange(layer_sae.n_concepts), np.asarray(list(exclude), dtype=int))
    if n > pool.size:
        raise ValueError(f"asked for {n} control latents but only {pool.size} available")
    return sorted(int(i) for i in rng.choice(pool, size=n, replace=False))


# --------------------------------------------------------------------------
# Interventions over a loader
# --------------------------------------------------------------------------
def _feature_ablation_hook(layer_sae, features, prefix, *, restrict_tokens=None):
    """Forward hook: subtract the joint decoder contribution of ``features``.

    delta_t = sum_{f in features} z_{t,f} * W_dec[f]     (raw residual space)
    Restricting to ``restrict_tokens`` zeroes the code elsewhere before the
    subtraction (equivalent to only removing the concept where it is planted).
    Mirrors attribution.core._ablate_one, batched over B and vectorised over F.
    """
    feats = torch.as_tensor(features, dtype=torch.long)
    W = layer_sae.dictionary[feats].detach()  # (k, D)

    def hook(_m, _in, out):
        is_t = isinstance(out, tuple)
        x = out[0] if is_t else out
        pre, sp = x[:, :prefix], x[:, prefix:]
        B, P, D = sp.shape
        codes = layer_sae.encode(sp.reshape(B * P, D)).reshape(B, P, -1)
        acts = codes[..., feats.to(sp.device)]  # (B, P, k)
        if restrict_tokens is not None:
            keep = torch.zeros(P, device=sp.device)
            keep[torch.as_tensor(restrict_tokens, device=sp.device)] = 1.0
            acts = acts * keep.view(1, P, 1)
        delta = torch.einsum("bpk,kd->bpd", acts, W.to(sp.dtype).to(sp.device))
        x_new = torch.cat([pre, sp - delta], dim=1)
        return (x_new,) + out[1:] if is_t else x_new

    return hook


def _steer_hook(direction, prefix, alpha, *, restrict_tokens=None, norm_scale=True):
    """Forward hook: add ``alpha * (scale) * unit(direction)`` to patch tokens.

    With ``norm_scale`` the direction is scaled to the batch's mean token norm, so
    ``alpha`` is a fraction of the residual magnitude (activation addition).
    """
    unit = direction.detach() / direction.norm().clamp_min(1e-8)

    def hook(_m, _in, out):
        is_t = isinstance(out, tuple)
        x = out[0] if is_t else out
        pre, sp = x[:, :prefix], x[:, prefix:]
        B, P, D = sp.shape
        u = unit.to(sp.dtype).to(sp.device)
        scale = sp.norm(dim=-1).mean() if norm_scale else 1.0
        add = (alpha * scale) * u.view(1, 1, D)
        if restrict_tokens is not None:
            mask = torch.zeros(P, device=sp.device)
            mask[torch.as_tensor(restrict_tokens, device=sp.device)] = 1.0
            add = add * mask.view(1, P, 1)
        x_new = torch.cat([pre, sp + add], dim=1)
        return (x_new,) + out[1:] if is_t else x_new

    return hook


@torch.no_grad()
def _run_loader(model, task, loader, site_hooks, device):
    """Predict over a loader with zero or more forward hooks. -> (preds, labels).

    ``site_hooks``: list of (site, hookfn) registered simultaneously (or None).
    Multiple hooks let us ablate several layers at once (cumulative ablation).
    """
    reader = model.reader
    model.to(device).eval()
    preds, labels = [], []
    handles = []
    for site, hook in (site_hooks or []):
        path, _ = reader.site_path(site)
        handles.append(reader.backbone.submodule(path).register_forward_hook(hook))
    try:
        for batch in loader:
            px = batch["pixel_values"].to(device)
            y = batch["labels"]
            logits = model(px, task)
            preds.append(logits.argmax(-1).cpu())
            labels.append(torch.as_tensor(y).cpu())
    finally:
        for h in handles:
            h.remove()
    return torch.cat(preds), torch.cat(labels)


def recall_of_class(preds: torch.Tensor, labels: torch.Tensor, target: int) -> float:
    """Recall for one class: TP / (TP + FN)."""
    mask = labels == target
    denom = int(mask.sum())
    if denom == 0:
        return float("nan")
    return float((preds[mask] == target).sum()) / denom


@torch.no_grad()
def ablate_features(
    model, layer_sae, site, features, loader, *, target_class, task=None,
    restrict_tokens=None, device="cuda",
):
    """Ablate ``features`` at ``site`` over a loader; return recall + accuracy."""
    task = _resolve_task(model, task)
    layer_sae.to(device)
    prefix = model.spec.n_prefix_tokens
    site_hooks = None
    if len(features):
        hook = _feature_ablation_hook(layer_sae, features, prefix, restrict_tokens=restrict_tokens)
        site_hooks = [(site, hook)]
    preds, labels = _run_loader(model, task, loader, site_hooks, device)
    return {
        "target_recall": recall_of_class(preds, labels, target_class),
        "accuracy": float((preds == labels).float().mean()),
        "n": int(labels.numel()),
    }


@torch.no_grad()
def ablate_features_multi(
    model, bank, site_features, loader, *, target_class, task=None,
    restrict_tokens=None, device="cuda",
):
    """Ablate features at *several* sites simultaneously (one pass).

    ``site_features``: {site: [feature indices]}. This is the cumulative-ablation
    primitive -- removing the watermark latents at every layer up to depth L at
    once, so downstream layers cannot re-encode the mark from an already-cleaned
    stream. Unlike the independent per-layer sweep, this localises where the
    shortcut is *irrecoverably* gone.
    """
    task = _resolve_task(model, task)
    prefix = model.spec.n_prefix_tokens
    site_hooks = []
    for site, feats in site_features.items():
        if not len(feats):
            continue
        bank[site].to(device)
        site_hooks.append(
            (site, _feature_ablation_hook(bank[site], feats, prefix, restrict_tokens=restrict_tokens))
        )
    preds, labels = _run_loader(model, task, loader, site_hooks or None, device)
    return {
        "target_recall": recall_of_class(preds, labels, target_class),
        "accuracy": float((preds == labels).float().mean()),
        "n": int(labels.numel()),
    }


@torch.no_grad()
def steer_direction(
    model, layer_sae, site, direction, loader, *, alpha, target_class, task=None,
    restrict_tokens=None, norm_scale=True, device="cuda",
):
    """Add ``alpha`` x ``direction`` at ``site``; return recall + accuracy.

    ``norm_scale=True`` (default): ``direction`` is unit-normalised and scaled to
    the batch's mean token norm, so ``alpha=1`` is a *full-magnitude* residual
    perturbation -- strong enough that a random direction drives the model off the
    manifold. ``norm_scale=False``: ``direction`` is added as-is (use with
    ``steering_vector``, whose magnitude is already the watermark's natural
    strength, so ``alpha`` is in units of "one watermark's worth").
    """
    task = _resolve_task(model, task)
    prefix = model.spec.n_prefix_tokens
    site_hooks = None
    if alpha != 0:
        hook = _steer_hook(direction.to(device), prefix, alpha,
                           restrict_tokens=restrict_tokens, norm_scale=norm_scale)
        site_hooks = [(site, hook)]
    preds, labels = _run_loader(model, task, loader, site_hooks, device)
    return {
        "target_recall": recall_of_class(preds, labels, target_class),
        "accuracy": float((preds == labels).float().mean()),
        "alpha": float(alpha),
    }


@torch.no_grad()
def steer_multi(
    model, site_vectors, loader, *, alpha, target_class, task=None,
    restrict_map=None, norm_scale=False, device="cuda",
):
    """Steer at *several* sites simultaneously (one pass).

    ``site_vectors``: {site: direction tensor}. The same ``alpha`` is applied at
    every site. This is the steering analogue of cumulative ablation: if a single
    early site cannot induce the concept but coordinated injection across the
    sites that carry it can, the pathway is distributed rather than absent -- the
    steering-space mirror of why cumulative (not single-site) ablation was needed
    to remove it. ``restrict_map`` optionally gives per-site corner tokens.
    """
    task = _resolve_task(model, task)
    prefix = model.spec.n_prefix_tokens
    site_hooks = None
    if alpha != 0:
        site_hooks = []
        for site, direction in site_vectors.items():
            restrict = None if restrict_map is None else restrict_map.get(site)
            hook = _steer_hook(direction.to(device), prefix, alpha,
                               restrict_tokens=restrict, norm_scale=norm_scale)
            site_hooks.append((site, hook))
    preds, labels = _run_loader(model, task, loader, site_hooks, device)
    return {
        "target_recall": recall_of_class(preds, labels, target_class),
        "accuracy": float((preds == labels).float().mean()),
        "alpha": float(alpha),
    }


@torch.no_grad()
def random_ablation_band(
    model, layer_sae, site, n, loader, *, target_class, task=None,
    exclude=(), seeds=(0, 1, 2, 3, 4), restrict_tokens=None, device="cuda",
):
    """Random-latent ablation over several seeds -> mean/std recall + the raw values.

    Averaging over seeds turns the single noisy control into an error band, so a
    real watermark ablation can be judged against the *spread* of random ablations
    rather than one draw.
    """
    recalls, feat_sets = [], []
    for s in seeds:
        ctrl = random_control_features(layer_sae, n, exclude=exclude, seed=s)
        r = ablate_features(
            model, layer_sae, site, ctrl, loader, target_class=target_class,
            task=task, restrict_tokens=restrict_tokens, device=device,
        )
        recalls.append(r["target_recall"])
        feat_sets.append(ctrl)
    arr = np.asarray(recalls, dtype=float)
    return {
        "mean": float(arr.mean()), "std": float(arr.std()),
        "min": float(arr.min()), "max": float(arr.max()),
        "recalls": recalls, "seeds": list(seeds), "feature_sets": feat_sets,
    }


def watermark_direction(layer_sae, features) -> torch.Tensor:
    """Unit-agnostic steering direction: the summed decoder atoms of the latents.

    Pair with ``steer_direction(..., norm_scale=True)`` (magnitude comes from the
    token norm).
    """
    D = layer_sae.dictionary
    feats = torch.as_tensor(features, dtype=torch.long, device=D.device)
    return D[feats].detach().sum(0)


def steering_vector(layer_sae, features, weights) -> torch.Tensor:
    """Natural-strength steering vector: sum_f w_f * W_dec[f].

    With ``weights`` = the watermark latents' induced activations (the ``scores``
    from ``identify_watermark_features``), this re-injects exactly the residual
    delta the watermark adds -- the mirror image of the ablation. Pair with
    ``steer_direction(..., norm_scale=False)`` so ``alpha=1`` == one watermark.
    """
    D = layer_sae.dictionary
    feats = torch.as_tensor(features, dtype=torch.long, device=D.device)
    w = torch.as_tensor(weights, dtype=D.dtype, device=D.device).view(-1, 1)
    return (D[feats].detach() * w).sum(0)


def random_direction(layer_sae, *, seed: int = 0, like: torch.Tensor | None = None) -> torch.Tensor:
    """A random vector in residual space -- the steering control.

    If ``like`` is given, the random vector is rescaled to match its norm, so the
    control perturbation has the *same magnitude* as the watermark steering vector
    (a fair "same push, wrong direction" baseline).
    """
    g = torch.Generator().manual_seed(seed)
    d = layer_sae.dictionary.shape[1]
    v = torch.randn(d, generator=g)
    if like is not None:
        v = v / v.norm().clamp_min(1e-8) * float(like.norm())
    return v