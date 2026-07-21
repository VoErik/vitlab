"""Feature Atlas: serves precomputed atlas artifacts (see scripts/app_precompute_atlas.py).

The heavy lifting (top-k images, UMAP, per-feature stats) happens offline. This router
reads those artifacts and serves feature tables, detail, and thumbnail/heatmap PNGs.
Dataset thumbnails are rendered live (denormalized so they match the model input).
"""

from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import APIRouter, HTTPException, Query
from PIL import Image

from vitlab.datasets import denormalize, get_splits

from .. import config
from .._common import png_response

router = APIRouter()


def _atlas_path(atlas_id: str) -> Path:
    p = config.atlas_dir() / atlas_id
    if not (p / "meta.json").exists():
        raise HTTPException(404, f"atlas {atlas_id!r} not found (run app_precompute_atlas.py)")
    return p


@lru_cache(maxsize=8)
def _atlas_meta(atlas_id: str) -> dict:
    return json.loads((_atlas_path(atlas_id) / "meta.json").read_text())


@lru_cache(maxsize=8)
def _atlas_features(atlas_id: str):
    return dict(np.load(_atlas_path(atlas_id) / "features.npz"))


@lru_cache(maxsize=8)
def _atlas_top(atlas_id: str) -> dict:
    return json.loads((_atlas_path(atlas_id) / "top_images.json").read_text())


@lru_cache(maxsize=4)
def _atlas_split(atlas_id: str):
    """The dataset split this atlas was computed on, in the SAME order as precompute
    (shuffle=False), so image_index maps directly to a row. Returns (split_ds, model_key)."""
    meta = _atlas_meta(atlas_id)["summary"]
    dataset, model_key, split = meta["dataset"], meta["model_key"], meta.get("split", "train")
    idx = {"train": 0, "val": 1, "validation": 1, "test": 2}[split]
    split_ds = get_splits(dataset, model_key=model_key, cast_labels=False)[idx]
    return split_ds, model_key


@router.get("/atlas/list")
def atlas_list():
    out = []
    root = config.atlas_dir()
    if root.exists():
        for meta in root.rglob("meta.json"):
            m = json.loads(meta.read_text())
            out.append({"id": str(meta.parent.relative_to(root)), **m.get("summary", {})})
    return {"atlases": out}


@router.get("/atlas/{atlas_id:path}/features")
def atlas_features(atlas_id: str,
                   filter: str | None = Query(None),
                   search: int | None = Query(None),
                   sort: str = Query("feature"),
                   scatter: str = Query("umap"),
                   offset: int = 0, limit: int = 5000):
    feats = _atlas_features(atlas_id)
    n = len(feats["feature"])
    idx = np.arange(n)
    if search is not None:
        idx = idx[feats["feature"][idx] == search]
    if filter == "dead":
        idx = idx[feats["dead"][idx]]
    elif filter == "alive":
        idx = idx[~feats["dead"][idx]]
    elif filter == "rare":
        idx = idx[(feats["firing_rate"][idx] < 0.01) & ~feats["dead"][idx]]
    elif filter == "common":
        idx = idx[feats["firing_rate"][idx] > 0.5]
    elif filter == "high_strength":
        alive = ~feats["dead"]
        thr = np.quantile(feats["mean_act"][alive], 0.9) if alive.any() else 0.0
        idx = idx[feats["mean_act"][idx] >= thr]

    if sort in feats:
        idx = idx[np.argsort(-feats[sort][idx])]

    xy = feats["umap"] if scatter == "umap" else np.stack(
        [feats["firing_rate"], feats["mean_act"]], axis=1)

    page = idx[offset:offset + limit]
    rows = [{
        "feature": int(feats["feature"][i]),
        "firing_rate": float(feats["firing_rate"][i]),
        "mean_act": float(feats["mean_act"][i]),
        "max_act": float(feats["max_act"][i]),
        "dead": bool(feats["dead"][i]),
        "x": float(xy[i, 0]), "y": float(xy[i, 1]),
    } for i in page]
    return {"total": int(len(idx)), "offset": offset, "rows": rows, "scatter": scatter}


@router.get("/atlas/{atlas_id:path}/feature/{f}")
def atlas_feature(atlas_id: str, f: int):
    feats = _atlas_features(atlas_id)
    where = np.where(feats["feature"] == f)[0]
    if not len(where):
        raise HTTPException(404, f"feature {f} not in atlas")
    i = int(where[0])
    top = _atlas_top(atlas_id).get(str(f), [])
    return {
        "feature": f,
        "stats": {
            "firing_rate": float(feats["firing_rate"][i]),
            "mean_act": float(feats["mean_act"][i]),
            "max_act": float(feats["max_act"][i]),
            "dead": bool(feats["dead"][i]),
        },
        "top_images": top,          # [{image_index, patch_index, score}]
    }


@router.get("/atlas/{atlas_id:path}/image/{image_index}")
def atlas_image(atlas_id: str, image_index: int, heatmap_feature: int | None = None,
                size: int = 224):
    """Render a dataset thumbnail (denormalized = what the model sees). If
    heatmap_feature is given, overlay that feature's activation map for THIS image,
    recomputed live (cheap: one forward for one image)."""
    split_ds, model_key = _atlas_split(atlas_id)
    if image_index < 0 or image_index >= len(split_ds):
        raise HTTPException(404, f"image_index {image_index} out of range")
    row = split_ds[image_index]
    px = row["pixel_values"]                                   # (3,H,W) preprocessed
    shown = denormalize(px, model_key).clamp(0, 1)            # (3,H,W) in [0,1]
    base = _to_pil(shown, size)

    if heatmap_feature is None:
        return png_response(_pil_png(base))

    # recompute this feature's activation map for this single image
    from .. import state
    meta = _atlas_meta(atlas_id)["summary"]
    hm = _feature_map_for_image(meta, px, heatmap_feature, size)
    overlay = _viridis_overlay(base, hm, size)
    return png_response(_pil_png(overlay))


# ---- helpers -------------------------------------------------------------
def _to_pil(chw: torch.Tensor, size: int) -> Image.Image:
    arr = (chw.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy())
    return Image.fromarray(arr).convert("RGB").resize((size, size))


def _pil_png(img: Image.Image) -> bytes:
    buf = io.BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()


def _feature_map_for_image(meta: dict, px: torch.Tensor, feature: int, size: int) -> np.ndarray:
    """(size,size) upsampled activation map of `feature` on one image."""
    from .. import state
    # the atlas id encodes model+site; meta carries them
    model = _model_for_meta(meta)
    bank = _bank_for_meta(meta)
    site = meta["site"]
    reader = model.reader
    prefix = model.spec.n_prefix_tokens
    with torch.no_grad():
        acts = reader.read(px.unsqueeze(0).to(state.DEVICE), site)[:, prefix:, :]
        codes = bank[site].encode(acts.reshape(-1, model.spec.d_model))[:, feature]
    P = codes.shape[0]
    side = int(round(P ** 0.5))
    grid = codes.reshape(side, side).float()
    up = F.interpolate(grid[None, None], size=(size, size), mode="bilinear",
                       align_corners=False)[0, 0].cpu().numpy()
    return up


def _model_for_meta(meta: dict):
    from .. import state
    return state.get_model(meta["checkpoint"])


def _bank_for_meta(meta: dict):
    from .. import state
    return state.get_bank(meta["bank"])


def _viridis_overlay(base: Image.Image, hm: np.ndarray, size: int) -> Image.Image:
    import matplotlib.cm as cm
    g = hm - hm.min()
    g = g / (g.max() + 1e-8)
    rgba = (cm.get_cmap("viridis")(g) * 255).astype("uint8")   # matplotlib<3.11 pinned
    heat = Image.fromarray(rgba, mode="RGBA").resize((size, size))
    heat.putalpha(150)
    return Image.alpha_composite(base.convert("RGBA"), heat).convert("RGB")
