"""Classification + attribution. This router is fully wired to vitlab and is the best
reference for how the others should call the library."""

from __future__ import annotations

import io

import torch
import torch.nn.functional as F
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

import vitlab.attribution as A
from vitlab import get_spec, list_models
from vitlab.datasets import get_dataset

from .. import config, state
from .._common import png_response, softmax_probs

router = APIRouter()


# ---- discovery -----------------------------------------------------------
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


# ---- classify ------------------------------------------------------------
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


# ---- attribution ---------------------------------------------------------
class AttributeRequest(BaseModel):
    image_token: str
    model_id: str
    task: str
    bank_id: str
    site: str                      # e.g. "blocks.9.resid_post"
    target_class: int | None = None
    method: str = "two_stage"      # two_stage | dla | attribution_patching
    top_k: int = 5


@router.post("/attribute")
def attribute(req: AttributeRequest):
    model = state.get_model(req.model_id)
    bank = state.get_bank(req.bank_id)
    if req.site not in bank.sites:
        raise HTTPException(400, f"site {req.site!r} not in bank sites {list(bank.sites)}")
    px = state.get_image(req.image_token).pixel_values.to(state.DEVICE)

    common = dict(task=req.task, target_class_idx=req.target_class, device=state.DEVICE)
    warning = None
    if req.method == "dla":
        res = A.direct_logit_attribution(model, bank, px, req.site, top_k=req.top_k, **common)
    elif req.method == "attribution_patching":
        res = A.attribution_patching(model, bank, px, req.site, top_k=req.top_k, **common)
    else:
        res = A.attribute(model, bank, px, req.site, top_k=req.top_k, **common)

    if all(abs(s) < 1e-12 for s in res.scores):
        warning = ("All attribution scores are ~0. If this is the final layer under CLS "
                   "pooling, patch features cannot influence the logit there -- attribute at "
                   "an earlier layer, or use method='dla'.")

    features = []
    maps = res.spatial_maps  # (k, H, W)
    for i, item in enumerate(res.top(len(res))):
        gm = maps[i].cpu().tolist() if i < len(maps) else None
        features.append({
            "site": item["site"], "feature": item["feature"],
            "score": item["score"], "grid_map": gm,
        })
    return {
        "method": res.method, "target_class": res.target_class,
        "features": features, "warning": warning,
    }


# ---- helpers -------------------------------------------------------------
def _class_names(task: str, n: int) -> list[str]:
    """Real class names from the dataset registry when available, else indices."""
    try:
        ds = get_dataset(task, cast_labels=True)
        feat = ds["train"].features.get("labels") or ds["train"].features.get("label")
        if feat is not None and hasattr(feat, "names"):
            return list(feat.names)
    except Exception:
        pass
    return [str(i) for i in range(n)]
