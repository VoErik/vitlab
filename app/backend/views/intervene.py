"""Interventions: patch ablation (pixel space) and concept ablation (latent space).

Concept ablation is wired to vitlab's ablation primitive. Patch ablation has a working
mean-fill baseline; wire the ROAD/noise-linear imputation from scripts/morf.py for
benchmark consistency (marked TODO).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from vitlab.attribution.core import _ablate_one, _resolve_task, _target_class

from .. import state
from .._common import softmax_probs

router = APIRouter()


def _probs_payload(model, px, task):
    with torch.no_grad():
        logits = model(px, task)
    return {"probs": softmax_probs(logits), "predicted": int(logits.argmax(-1).item())}


class PatchAblateRequest(BaseModel):
    image_token: str
    model_id: str
    task: str
    patches: list[int]          # flat patch indices (0..P-1), NOT including prefix
    method: str = "mean"        # "mean" | "road" (TODO) | "noise"


@router.post("/intervene/patch_ablate")
def patch_ablate(req: PatchAblateRequest):
    model = state.get_model(req.model_id)
    cached = state.get_image(req.image_token)
    px = cached.pixel_values.clone().to(state.DEVICE)
    spec = model.spec
    side = spec.image_size // spec.patch_size
    ps = spec.patch_size

    clean = _probs_payload(model, px, req.task)

    # keep-mask: 1 where we KEEP the original pixels, 0 where we intervene
    keep = torch.ones_like(px)
    for pidx in req.patches:
        r, c = divmod(pidx, side)
        y0, x0 = r * ps, c * ps
        keep[:, :, y0:y0 + ps, x0:x0 + ps] = 0.0

    if req.method == "road":
        corrupted = _impute_road(px, keep)
    elif req.method == "noise":
        corrupted = px * keep + torch.randn_like(px) * (1 - keep)
    else:  # "mean"
        fill = px.mean(dim=(2, 3), keepdim=True)
        corrupted = px * keep + fill * (1 - keep)

    corr = _probs_payload(model, corrupted, req.task)
    # cache the corrupted image so the frontend can display exactly what the model saw
    from vitlab.datasets import denormalize
    from ..state import _tensor_to_png, CachedImage
    import uuid
    tok = uuid.uuid4().hex
    state._images[tok] = CachedImage(
        pixel_values=corrupted, model_key=spec.key,
        shown_png=_tensor_to_png(denormalize(corrupted[0].cpu(), spec.key)))
    return {"clean": clean, "corrupted": corr, "corrupted_image": f"/api/image/{tok}"}


def _impute_road(img: torch.Tensor, keep: torch.Tensor, noise_std: float = 0.1,
                 steps: int = 15) -> torch.Tensor:
    """ROAD diffusion imputation (ported from scripts/morf.py::impute_road_diffusion).

    keep: 1 = keep original, 0 = impute. Fills masked regions by iteratively diffusing
    neighbouring pixels inward, then adds mild noise. Consistent with the MoRF benchmark.
    """
    mean_val = img.mean(dim=(2, 3), keepdim=True)
    filled = img * keep + mean_val * (1 - keep)
    k = torch.tensor([[1 / 12, 1 / 6, 1 / 12],
                      [1 / 6, 0.0, 1 / 6],
                      [1 / 12, 1 / 6, 1 / 12]], device=img.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    for _ in range(steps):
        smoothed = F.conv2d(filled, k, padding=1, groups=3)
        filled = img * keep + smoothed * (1 - keep)
    noise = torch.randn_like(img) * noise_std
    return filled * keep + (filled + noise) * (1 - keep)


class ConceptAblateRequest(BaseModel):
    image_token: str
    model_id: str
    task: str
    bank_id: str
    sites: list[str]                       # one or more "blocks.{l}.resid_post"
    features: list[int]                    # feature indices to ablate (paired with sites, or all-at-each)
    patches: list[int] | None = None       # None = all patches
    target_class: int | None = None


@router.post("/intervene/concept_ablate")
def concept_ablate(req: ConceptAblateRequest):
    model = state.get_model(req.model_id)
    bank = state.get_bank(req.bank_id)
    task = _resolve_task(model, req.task)
    px = state.get_image(req.image_token).pixel_values.to(state.DEVICE)
    cls = _target_class(model, px, task, req.target_class)
    prefix = model.spec.n_prefix_tokens

    clean = _probs_payload(model, px, task)

    # Apply each (site, feature) ablation jointly by chaining hooks. _ablate_one returns
    # the logit DROP for one feature; for the full corrupted prediction we register the
    # same subtraction hooks it uses and run one forward. Simplest correct approach:
    # apply sequentially via repeated single-feature ablations is NOT joint. Instead we
    # replicate _ablate_one's hook for every (site,feature) and do one pass.
    reader = model.reader
    handles = []
    try:
        for site in req.sites:
            if site not in bank.sites:
                raise HTTPException(400, f"site {site!r} not in bank")
            layer_sae = bank[site]
            for feat in req.features:
                mod, hook = _make_ablate_hook(reader, layer_sae, feat, prefix, req.patches)
                handles.append(mod.register_forward_hook(hook))
        with torch.no_grad():
            logits = model(px, task)
    finally:
        for h in handles:
            h.remove()

    corr = {"probs": softmax_probs(logits), "predicted": int(logits.argmax(-1).item())}
    return {"clean": clean, "corrupted": corr, "target_class": cls}


def _make_ablate_hook(reader, layer_sae, feature, prefix, patch_positions):
    """Subtract one feature's decoder contribution at (optionally selected) patches.
    Mirrors vitlab.attribution.core._ablate_one's hook; study that for the exact math."""
    W_dec = layer_sae.dictionary
    vec = W_dec[feature].detach()

    def hook(_m, _in, out):
        is_t = isinstance(out, tuple)
        x = out[0] if is_t else out
        pre, sp = x[:, :prefix], x[:, prefix:]
        B, P, D = sp.shape
        codes = layer_sae.encode(sp.reshape(B * P, D)).reshape(B, P, -1)
        act = codes[..., feature].clone()             # (B,P)
        if patch_positions is not None:
            keep = torch.zeros_like(act)
            for p in patch_positions:
                if 0 <= p < P:
                    keep[:, p] = act[:, p]
            act = keep
        delta = act.unsqueeze(-1) * vec.view(1, 1, -1)
        sp = sp - delta.to(sp.dtype)
        x_new = torch.cat([pre, sp], dim=1)
        return (x_new,) + out[1:] if is_t else x_new

    path, _ = reader.site_path(layer_sae.site)
    return reader.backbone.submodule(path), hook
