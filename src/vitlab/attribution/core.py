"""
Four methods:

  direct_logit_attribution   Projects each concept's decoder vector onto the classifier 
                             weights: contribution = z_f * (W_dec[f] . W_class[c]).
                             No gradient, no extra forward pass. Exact for the *linear*
                             read-out from this layer, ignores downstream nonlinearity.

  attribution_patching       act x grad: z_f * d(logit)/d(z_f), summed over positions.
                             The first-order Taylor estimate of the logit change from
                             ablating concept f to zero. One forward + one backward.

  ablation                   Empirical logit drop from actually subtracting concept f's
                             residual contribution and re-running the network. Measures
                             the *total* causal effect including downstream compensation.
                             One forward pass per concept.

  attribute (two-stage)      First attribution_patching, then ablation to verify.

All methods act on a MultiTaskViT (backbone + task head) and a SAEBank keyed by site. 
Layout (prefix tokens, patch grid) is read from the model spec.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class AttributionResult:
    """Ranked SAE concepts for one image + task."""

    features: list[int]
    sites: list[str]
    scores: list[float]
    method: str
    target_class: int
    spatial_maps: torch.Tensor = field(default_factory=lambda: torch.empty(0)) # (k, H, W)

    def __len__(self) -> int:
        return len(self.features)

    def top(self, n: int = 5):
        for i in range(min(n, len(self))):
            yield {"rank": i + 1, "site": self.sites[i], "feature": self.features[i],
                   "score": self.scores[i]}

def _resolve_task(model, task: str | None) -> str:
    if task is not None:
        return task
    names = model.task_names
    if len(names) != 1:
        raise ValueError(f"model has {len(names)} tasks {names}; pass task=")
    return names[0]


def _classifier_weight(model, task: str) -> torch.Tensor:
    """(num_classes, d_model) final linear weight of the task head."""
    return model.heads[task].classifier.weight.detach()


def _grid(n_patches: int) -> tuple[int, int]:
    import math
    h = int(round(math.sqrt(n_patches)))
    if h * h != n_patches:
        raise ValueError(f"{n_patches} patches not square; attribution maps need a square grid")
    return h, h


@torch.no_grad()
def _target_class(model, px, task, target_class_idx):
    if target_class_idx is not None:
        return target_class_idx
    return model(px, task).argmax(-1).item()


@torch.no_grad()
def direct_logit_attribution(model, bank, pixel_values, site, *, task=None,
                             target_class_idx=None, device="cuda", top_k=10):
    """
    DLA: concept contribution = activation * (decoder_vec . classifier_weight[c]).
    """
    model.to(device).eval()
    task = _resolve_task(model, task)
    px = pixel_values.to(device)
    cls = _target_class(model, px, task, target_class_idx)

    reader = model.reader
    prefix = model.spec.n_prefix_tokens
    layer_sae = bank[site].to(device)

    acts = reader.read(px, site)
    patches = acts[:, prefix:, :]
    B, P, D = patches.shape
    codes = layer_sae.encode(patches.reshape(B * P, D)).reshape(B, P, -1)

    # decoder atoms (F, D) . classifier row (D,) -> (F,) logit-per-unit-activation
    W_dec = layer_sae.dictionary.to(device) # (F, D)
    w_class = _classifier_weight(model, task)[cls].to(device) # (D,)
    per_unit = W_dec @ w_class # (F,)

    total_act = codes[0].sum(0)
    contrib = total_act * per_unit

    scores, idx = torch.topk(contrib, min(top_k, contrib.numel()))
    h, w = _grid(P)
    maps = (codes[0].T.reshape(-1, h, w))[idx].detach().cpu()
    return AttributionResult(
        features=idx.tolist(), sites=[site] * len(idx), scores=scores.tolist(),
        method="direct_logit_attribution", target_class=cls, spatial_maps=maps,
    )

@contextmanager
def _splice_codes(model, bank, sites, prefix, cache: dict):
    """
    Register forward hooks that replace each site's patch activations with the
    SAE's error-preserving reconstruction, and stash the (grad-enabled) codes.

    The splice is  x -> x_hat + (x - x_hat).detach()  at the patch tokens: the
    forward value equals the clean reconstruction plus a detached error term, so
    the network runs on a faithful reconstruction while gradients flow through the
    codes.
    """
    handles = []
    reader = model.reader

    def make_hook(site):
        layer_sae = bank[site]

        def hook(_m, _in, out):
            is_tuple = isinstance(out, tuple)
            x = out[0] if is_tuple else out
            B, S, D = x.shape
            pre, sp = x[:, :prefix], x[:, prefix:]
            codes = layer_sae.encode(sp.reshape(-1, D)).reshape(B, S - prefix, -1)
            codes = codes.detach().requires_grad_(True)
            cache[site] = codes
            x_hat = layer_sae.decode(codes.reshape(-1, codes.shape[-1])).reshape(B, S - prefix, D)
            spliced = sp + (x_hat - x_hat.detach())
            x_new = torch.cat([pre, spliced], dim=1)
            return (x_new,) + out[1:] if is_tuple else x_new

        return hook

    try:
        for site in sites:
            path, _ = reader.site_path(site)
            handles.append(reader.backbone.submodule(path).register_forward_hook(make_hook(site)))
        yield cache
    finally:
        for h in handles:
            h.remove()

def attribution_patching(model, bank, pixel_values, sites, *, task=None,
                         target_class_idx=None, device="cuda", top_k=10,
                         normalize_cross_site=True):
    """
    act x grad attribution: for each concept, z_f * d(logit)/d(z_f) summed over
    positions. First-order estimate of the logit change from ablating the concept.
    """
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    task = _resolve_task(model, task)
    sites = [sites] if isinstance(sites, str) else list(sites)
    px = pixel_values.to(device)
    cls = _target_class(model, px, task, target_class_idx)
    prefix = model.spec.n_prefix_tokens

    cache: dict = {}
    with _splice_codes(model, bank, sites, prefix, cache):
        logit = model(px, task)[0, cls]
        grads = torch.autograd.grad(logit, [cache[s] for s in sites])

    candidates = []
    for site, g in zip(sites, grads):
        codes = cache[site]
        eff = (codes * g).sum(dim=(0, 1))
        if eff.abs().max() == 0:
            import warnings
            warnings.warn(
                f"attribution_patching: all-zero gradient at {site!r}. This is expected "
                f"when the site is the final block AND the head pools CLS only -- patch "
                f"tokens cannot influence the logit there. Attribute at an earlier layer, "
                f"or use direct_logit_attribution (which reads the classifier directly).",
                stacklevel=2,
            )
        if normalize_cross_site and eff.abs().max() > 0:
            eff = eff / eff.abs().max()
        P = codes.shape[1]
        h, w = _grid(P)
        act_maps = codes[0].T.reshape(-1, h, w).detach().cpu()
        for f in range(eff.numel()):
            candidates.append((eff[f].item(), site, f, act_maps[f]))

    candidates.sort(key=lambda c: c[0], reverse=True)
    top = candidates[:top_k]
    return AttributionResult(
        features=[c[2] for c in top], sites=[c[1] for c in top], scores=[c[0] for c in top],
        method="attribution_patching", target_class=cls,
        spatial_maps=torch.stack([c[3] for c in top]) if top else torch.empty(0),
    )

@torch.no_grad()
def _ablate_one(model, bank, px, site, feature, cls, task, prefix, clean_logit,
                ablate_positions=None):
    """Subtract one concept's residual contribution at its site and re-run."""
    reader = model.reader
    layer_sae = bank[site]
    W_dec = layer_sae.dictionary
    vec = W_dec[feature].detach()

    acts = reader.read(px, site)[:, prefix:, :]
    B, P, D = acts.shape
    codes = layer_sae.encode(acts.reshape(B * P, D)).reshape(B, P, -1)
    feat_act = codes[0, :, feature].clone()
    if ablate_positions is not None:
        keep = torch.zeros_like(feat_act)
        for pos in ablate_positions:
            if 0 <= pos < P:
                keep[pos] = feat_act[pos]
        feat_act = keep

    # subtract feat_act[p] * decoder_vec from patch p, in raw activation space
    # TODO: currently this equals zero ablation, which might be better for this but feels iffy, think about it, but should be fine for now
    delta = feat_act.unsqueeze(-1) * vec.unsqueeze(0)

    def hook(_m, _in, out):
        is_tuple = isinstance(out, tuple)
        x = out[0] if is_tuple else out
        pre, sp = x[:, :prefix], x[:, prefix:]
        sp = sp - delta.unsqueeze(0).to(sp.dtype)
        x_new = torch.cat([pre, sp], dim=1)
        return (x_new,) + out[1:] if is_tuple else x_new

    path, _ = reader.site_path(site)
    hnd = reader.backbone.submodule(path).register_forward_hook(hook)
    try:
        new_logit = model(px, task)[0, cls].item()
    finally:
        hnd.remove()
    return clean_logit - new_logit


def ablation(model, bank, pixel_values, candidates, *, task=None,
             target_class_idx=None, device="cuda", ablate_positions=None):
    """
    Empirical logit drop for each (site, feature) in `candidates`.

    candidates: list of (site, feature) tuples.
    """
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    task = _resolve_task(model, task)
    px = pixel_values.to(device)
    cls = _target_class(model, px, task, target_class_idx)
    prefix = model.spec.n_prefix_tokens

    with torch.no_grad():
        clean = model(px, task)[0, cls].item()

    scored = []
    for site, feat in candidates:
        drop = _ablate_one(model, bank, px, site, feat, cls, task, prefix, clean,
                           ablate_positions=ablate_positions)
        scored.append((drop, site, feat))
    scored.sort(key=lambda c: c[0], reverse=True)
    return AttributionResult(
        features=[c[2] for c in scored], sites=[c[1] for c in scored], scores=[c[0] for c in scored],
        method="ablation", target_class=cls,
    )

def attribute(model, bank, pixel_values, sites, *, task=None, target_class_idx=None,
              device="cuda", top_k=5, candidate_multiplier=3, dedup_iou=0.6,
              ablate_positions=None):
    """
    1: attribution patching screens a wide net of candidates cheaply. 
    2: spatially deduplicate. # TODO: i am still unsure if thats the right move
    3: ablation for verification
    """
    task = _resolve_task(model, task)
    sites = [sites] if isinstance(sites, str) else list(sites)

    # 1
    screened = attribution_patching(
        model, bank, pixel_values, sites, task=task, target_class_idx=target_class_idx,
        device=device, top_k=top_k * candidate_multiplier,
    )
    # 2
    shortlist = _dedup_spatial(
        list(zip(screened.sites, screened.features, screened.spatial_maps)),
        iou_threshold=dedup_iou,
    )[:top_k * candidate_multiplier]

    # 3
    verified = ablation(
        model, bank, pixel_values, [(s, f) for s, f, _ in shortlist],
        task=task, target_class_idx=screened.target_class, device=device,
        ablate_positions=ablate_positions,
    )

    map_by_key = {(s, f): m for s, f, m in shortlist}
    verified.spatial_maps = torch.stack(
        [map_by_key[(s, f)] for s, f in zip(verified.sites, verified.features)]
    ) if len(verified) else torch.empty(0)
    verified.method = "two_stage(patching->ablation)"
    return AttributionResult(
        features=verified.features[:top_k], sites=verified.sites[:top_k],
        scores=verified.scores[:top_k], method=verified.method,
        target_class=verified.target_class,
        spatial_maps=verified.spatial_maps[:top_k] if len(verified) else torch.empty(0),
    )


def _dedup_spatial(items, iou_threshold=0.6):
    kept = []
    for site, feat, hmap in items:
        cur = (hmap > hmap.max() * 0.2).float()
        redundant = False
        for _, _, kmap in kept:
            k = (kmap > kmap.max() * 0.2).float()
            inter = (cur * k).sum()
            union = ((cur + k).clamp(0, 1)).sum() + 1e-8
            if (inter / union) > iou_threshold:
                redundant = True
                break
        if not redundant:
            kept.append((site, feat, hmap))
    return kept