from __future__ import annotations

import io

import torch
import torch.nn.functional as F
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

import vitlab.attribution as A
from vitlab import get_spec, list_models
from vitlab.attribution.core import _resolve_task, _target_class
from vitlab.datasets import get_dataset

from .. import config, state
from .._common import png_response, softmax_probs
from .._contrastive import contrastive_dla, contrastive_patching

router = APIRouter()


@router.get("/models")
def models():
    """Available fine-tuned checkpoints under CHECKPOINTS_DIR."""
    out = []
    if config.checkpoints_dir().exists():
        for d in sorted(config.checkpoints_dir().iterdir()):
            if (d / "config.json").exists():
                m = state.get_model(d.name)
                out.append({
                    "id": d.name,
                    "model_key": m.spec.key,
                    "tasks": m.task_names,
                    "n_layers": m.spec.n_layers,
                    "n_prefix_tokens": m.spec.n_prefix_tokens,
                    "image_size": m.spec.image_size,
                    "patch_size": m.spec.patch_size,
                })
    return {"models": out}


@router.get("/models/{model_id}/banks")
def banks(model_id: str):
    """SAE banks available for this model (dirs under BANKS_DIR that discover_bank can read)."""
    out = []
    if config.banks_dir().exists():
        for d in sorted(config.banks_dir().iterdir()):
            if d.is_dir():
                try:
                    bank = state.get_bank(d.name)
                    out.append({"id": d.name, "sites": list(bank.sites)})
                except Exception:
                    continue
    return {"banks": out}


@router.post("/classify")
async def classify(model_id: str = Form(...), task: str = Form(...),
                   image: UploadFile = File(...)):
    model = state.get_model(model_id)
    if task not in model.task_names:
        raise HTTPException(400, f"task {task!r} not in {model.task_names}")
    pil = Image.open(io.BytesIO(await image.read()))
    token = state.cache_upload(pil, model.spec.key)
    cached = state.get_image(token)

    with torch.no_grad():
        logits = model(cached.pixel_values.to(state.DEVICE), task)
    probs = softmax_probs(logits)
    names = _class_names(task, len(probs))
    ranked = sorted(
        [{"idx": i, "name": names[i], "prob": p} for i, p in enumerate(probs)],
        key=lambda r: -r["prob"],
    )
    return {
        "image_token": token,
        "predicted": int(logits.argmax(-1).item()),
        "classes": ranked,
        "shown_image": f"/api/image/{token}",
    }


@router.get("/image/{token}")
def image(token: str):
    try:
        return png_response(state.get_image(token).shown_png)
    except KeyError as e:
        raise HTTPException(404, str(e))


class AttributeRequest(BaseModel):
    image_token: str
    model_id: str
    task: str
    bank_id: str
    site: str 
    target_class: int | None = None
    contrast_class: int | None = None
    method: str = "two_stage"
    top_k: int = 5


def _run_attr(model, bank, px, site, method, top_k, common, contrast=None):
    """Dispatch one attribution method at a single site.

    When `contrast` is set the target becomes the margin logit(A)-logit(contrast); only
    DLA and attribution-patching express that, so two_stage falls back to patching."""
    if contrast is not None:
        a = common["target_class_idx"]
        task, device = common["task"], common["device"]
        if method == "dla":
            return contrastive_dla(model, bank, px, site, task, a, contrast, top_k, device)
        return contrastive_patching(model, bank, px, site, task, a, contrast, top_k, device)
    if method == "dla":
        return A.direct_logit_attribution(model, bank, px, site, top_k=top_k, **common)
    if method == "attribution_patching":
        return A.attribution_patching(model, bank, px, site, top_k=top_k, **common)
    return A.attribute(model, bank, px, site, top_k=top_k, **common)


def _features_payload(res):
    """AttributionResult -> list of {site, feature, score, grid_map}."""
    maps = res.spatial_maps  # (k, H, W)
    out = []
    for i, item in enumerate(res.top(len(res))):
        gm = maps[i].cpu().tolist() if i < len(maps) else None
        out.append({
            "site": item["site"], "feature": item["feature"],
            "score": item["score"], "grid_map": gm,
        })
    return out


def _contrast_note(contrast, method):
    if contrast is not None and method == "two_stage":
        return "Two-stage isn't defined for a contrastive target; used attribution-patching."
    return None


@router.post("/attribute")
def attribute(req: AttributeRequest):
    model = state.get_model(req.model_id)
    bank = state.get_bank(req.bank_id)
    if req.site not in bank.sites:
        raise HTTPException(400, f"site {req.site!r} not in bank sites {list(bank.sites)}")
    px = state.get_image(req.image_token).pixel_values.to(state.DEVICE)

    task = _resolve_task(model, req.task)
    target = req.target_class if req.target_class is not None else _target_class(model, px, task, None)
    common = dict(task=task, target_class_idx=target, device=state.DEVICE)
    res = _run_attr(model, bank, px, req.site, req.method, req.top_k, common, contrast=req.contrast_class)

    warning = _contrast_note(req.contrast_class, req.method)
    if all(abs(s) < 1e-12 for s in res.scores):
        zero = ("All attribution scores are ~0. If this is the final layer under CLS pooling, "
                "patch features cannot influence the logit there -- attribute at an earlier "
                "layer, or use method='dla'.")
        warning = f"{warning} {zero}" if warning else zero

    return {
        "method": res.method, "target_class": res.target_class,
        "contrast_class": req.contrast_class,
        "features": _features_payload(res), "warning": warning,
    }


class AttributeAllRequest(BaseModel):
    image_token: str
    model_id: str
    task: str
    bank_id: str
    target_class: int | None = None
    contrast_class: int | None = None
    method: str = "two_stage"
    top_k: int = 5


@router.post("/attribute_all_layers")
def attribute_all_layers(req: AttributeAllRequest):
    """Attribute at every site in the bank, returning the top features per layer.

    The target class is resolved once so every layer explains the same class (or margin),
    and the globally strongest (site, feature) is flagged so the UI can highlight it."""
    model = state.get_model(req.model_id)
    bank = state.get_bank(req.bank_id)
    px = state.get_image(req.image_token).pixel_values.to(state.DEVICE)

    task = _resolve_task(model, req.task)
    cls = req.target_class if req.target_class is not None else _target_class(model, px, task, None)
    common = dict(task=task, target_class_idx=cls, device=state.DEVICE)

    def layer_of(site: str) -> int:
        for part in site.split("."):
            if part.isdigit():
                return int(part)
        return -1

    sites = sorted(bank.sites, key=layer_of)
    layers_out, best, method_label, any_nonzero = [], None, req.method, False
    for site in sites:
        res = _run_attr(model, bank, px, site, req.method, req.top_k, common, contrast=req.contrast_class)
        method_label = res.method
        feats = _features_payload(res)
        for f in feats:
            if abs(f["score"]) > 1e-12:
                any_nonzero = True
            if best is None or abs(f["score"]) > abs(best["score"]):
                best = {"site": f["site"], "feature": f["feature"], "score": f["score"]}
        layers_out.append({"site": site, "layer": layer_of(site), "features": feats})

    warning = _contrast_note(req.contrast_class, req.method)
    if not any_nonzero:
        zero = ("All attribution scores are ~0 across every layer. Under CLS-only pooling patch "
                "features can't move the logit at the final block; earlier layers or method='dla' "
                "will be more informative.")
        warning = f"{warning} {zero}" if warning else zero

    return {
        "method": method_label, "target_class": cls, "contrast_class": req.contrast_class,
        "layers": layers_out, "top": best, "warning": warning,
    }

class TokenGroupsRequest(BaseModel):
    image_token: str
    model_id: str
    task: str
    target_class: int | None = None


@router.post("/token_groups")
def token_groups(req: TokenGroupsRequest):
    model = state.get_model(req.model_id)
    px = state.get_image(req.image_token).pixel_values.to(state.DEVICE)
    task = _resolve_task(model, req.task)
    res = A.token_group_attribution(
        model, px, task=task, target_class_idx=req.target_class, device=state.DEVICE)
    return {
        "layers": res.layers, "groups": res.groups,
        "gradient_mass": res.gradient_mass, "ablation_drop": res.ablation_drop,
        "target_class": res.target_class,
    }

class CoverageRequest(BaseModel):
    image_token: str
    model_id: str
    task: str
    bank_id: str
    target_class: int | None = None
    sites: list[str] | None = None          # default: every site in the bank


def _replace_patches_hook(prefix, fn):
    def hook(_m, _in, out):
        is_t = isinstance(out, tuple)
        x = out[0] if is_t else out
        pre, sp = x[:, :prefix], x[:, prefix:]
        x_new = torch.cat([pre, fn(sp)], dim=1)
        return (x_new,) + out[1:] if is_t else x_new
    return hook


@torch.no_grad()
def _logit_with_hook(model, reader, px, task, site, prefix, fn):
    path, _ = reader.site_path(site)
    h = reader.backbone.submodule(path).register_forward_hook(_replace_patches_hook(prefix, fn))
    try:
        return model(px, task)
    finally:
        h.remove()


@router.post("/coverage")
def coverage(req: CoverageRequest):
    model = state.get_model(req.model_id)
    bank = state.get_bank(req.bank_id)
    task = _resolve_task(model, req.task)
    px = state.get_image(req.image_token).pixel_values.to(state.DEVICE)
    cls = _target_class(model, px, task, req.target_class)
    prefix = model.spec.n_prefix_tokens
    reader = model.reader

    def layer_of(site: str) -> int:
        for part in site.split("."):
            if part.isdigit():
                return int(part)
        return -1

    sites = req.sites or sorted(bank.sites, key=layer_of)
    for s in sites:
        if s not in bank.sites:
            raise HTTPException(400, f"site {s!r} not in bank")

    with torch.no_grad():
        clean_logits = model(px, task)
    l_clean = float(clean_logits[0, cls])
    p_clean = float(F.softmax(clean_logits[0], -1)[cls])

    out_sites = []
    for site in sites:
        sae = bank[site].to(state.DEVICE)
        with torch.no_grad():
            acts = reader.read(px, site)[:, prefix:, :]            # (1, P, D)
            B, P, D = acts.shape
            codes = sae.encode(acts.reshape(B * P, D))
            x_hat = sae.decode(codes).reshape(B, P, D)
        e = acts - x_hat
        denom_var = ((acts - acts.mean(dim=1, keepdim=True)) ** 2).sum().clamp_min(1e-12)
        explained_var = float(1.0 - (e ** 2).sum() / denom_var)
        cos = float(F.cosine_similarity(acts.reshape(-1, D), x_hat.reshape(-1, D), dim=-1).mean())
        l0 = float((codes.abs() > 1e-8).float().sum(-1).mean())

        def recon_fn(sp, _sae=sae):
            c = _sae.encode(sp.reshape(-1, sp.shape[-1]))
            return _sae.decode(c).reshape(sp.shape).to(sp.dtype)

        def ablate_fn(sp):
            return sp.mean(dim=1, keepdim=True).expand_as(sp)

        rl = _logit_with_hook(model, reader, px, task, site, prefix, recon_fn)
        al = _logit_with_hook(model, reader, px, task, site, prefix, ablate_fn)
        l_recon, l_ablate = float(rl[0, cls]), float(al[0, cls])
        p_recon = float(F.softmax(rl[0], -1)[cls])
        denom = l_clean - l_ablate
        cov = (l_recon - l_ablate) / denom if abs(denom) > 1e-6 else None

        out_sites.append({
            "site": site, "layer": layer_of(site),
            "l_clean": l_clean, "l_recon": l_recon, "l_ablate": l_ablate,
            "logit_coverage": cov, "explained_var": explained_var, "cos": cos, "l0": l0,
            "p_clean": p_clean, "p_recon": p_recon,
        })

    valid = [s["logit_coverage"] for s in out_sites if s["logit_coverage"] is not None]
    overall = {
        "mean_logit_coverage": (sum(valid) / len(valid)) if valid else None,
        "mean_explained_var": (sum(s["explained_var"] for s in out_sites) / len(out_sites)) if out_sites else None,
    }
    return {"target_class": cls, "sites": out_sites, "overall": overall}

def _class_names(task: str, n: int) -> list[str]:
    """Real class names in classifier-index order, else string indices."""
    from .._labels import class_names_for_task
    return list(class_names_for_task(task, n))