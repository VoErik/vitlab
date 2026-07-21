"""Precompute the Feature Atlas artifacts for the app.

Produces, under ATLAS_DIR/<atlas_id>/:
    meta.json          spec + sae summary + params (+ checkpoint/bank/split for live rendering)
    features.npz       per-feature: feature, firing_rate, mean_act, max_act, dead, umap(x,y)
    top_images.json    feature -> [{image_index, patch_index, score}]  (top-k activating images)

Usage:
    uv run python scripts/app_precompute_atlas.py \
        --model runs/dinov2-base_full_epochs30/best \
        --bank saes/dinov2_fitz \
        --dataset fitzpatrick17k-skincon --split train \
        --site blocks.9.resid_post --top-k 12 [--force]

Design notes
------------
* Single streaming pass computes BOTH per-feature stats AND per-feature top-k images,
  so cost is O(dataset), not O(dataset * n_features). We keep a small top-k heap per
  feature of (max-activation-on-image, image_index, patch_index).
* image_index is the row position in the split loaded with shuffle=False -- the backend
  reloads the same split the same way, so image_index maps back to a row for rendering.
* UMAP is fit on the SAE decoder dictionary (F x D) -> (F, 2): feature geometry in
  dictionary space. (Swap to code-profile UMAP if you prefer; documented tradeoff.)
"""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import numpy as np
import torch

import vitlab
from vitlab import ActivationReader
from vitlab.batching import make_loader
from vitlab.datasets import get_splits
from vitlab.sae import load_layer_sae

# app config for the output dir (import works when run from repo root)
try:
    from app.backend import config as app_config
    ATLAS_DIR = app_config.atlas_dir()
except Exception:  # fallback: DATA_ROOT/atlas
    from vitlab.config import get_data_root
    ATLAS_DIR = get_data_root() / "atlas"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint dir (relative to runs/ or absolute)")
    ap.add_argument("--bank", required=True, help="SAE bank tree OR a single site dir")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val", "validation", "test"])
    ap.add_argument("--site", required=True)
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = vitlab.load_model(args.model, device=device)
    reader = ActivationReader(model.backbone)
    spec = model.spec
    prefix = spec.n_prefix_tokens
    D = spec.d_model

    sae_dir = Path(args.bank)
    bank_arg = args.bank
    if (sae_dir / args.site).exists():
        sae_dir = sae_dir / args.site
    sae = load_layer_sae(str(sae_dir), device=device)
    Fn = sae.n_concepts

    atlas_id = f"{Path(args.model).name}/{args.dataset}/{args.site}"
    out_dir = ATLAS_DIR / atlas_id
    if out_dir.exists() and not args.force:
        print(f"exists (use --force): {out_dir}")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = {"train": 0, "val": 1, "validation": 1, "test": 2}[args.split]
    ds = get_splits(args.dataset, model_key=spec.key, cast_labels=False)[idx]

    def _collate(rows):
        return {"pixel_values": torch.stack([r["pixel_values"] for r in rows])}
    loader = make_loader(ds, args.batch_size, shuffle=False, num_workers=4, collate_fn=_collate)

    # ---- single streaming pass: stats + per-feature top-k heaps ----
    fire = torch.zeros(Fn); act_sum = torch.zeros(Fn); act_max = torch.zeros(Fn)
    heaps: list[list] = [[] for _ in range(Fn)]     # per feature: heap of (score, img_idx, patch_idx)
    n_tokens = 0
    img_counter = 0
    K = args.top_k
    sae.to(device)

    print(f"streaming {len(ds)} images -> stats + top-{K} for {Fn} features ...")
    with torch.no_grad():
        for batch in loader:
            px = batch["pixel_values"].to(device)
            acts = reader.read(px, args.site)[:, prefix:, :]          # (B,P,D)
            B, P, _ = acts.shape
            codes = sae.encode(acts.reshape(B * P, D)).reshape(B, P, Fn)  # (B,P,F)
            active = codes > 1e-6
            fire += active.any(0).sum(0).cpu().float() if False else active.reshape(-1, Fn).sum(0).cpu().float()
            act_sum += (codes * active).reshape(-1, Fn).sum(0).cpu()
            act_max = torch.maximum(act_max, codes.reshape(-1, Fn).max(0).values.cpu())
            n_tokens += B * P

            # per-image best activation + patch for every feature (vectorised)
            best_val, best_patch = codes.max(dim=1)                   # (B,F),(B,F)
            bv = best_val.cpu(); bp = best_patch.cpu()
            for b in range(B):
                gi = img_counter + b
                row_vals = bv[b]; row_patch = bp[b]
                # only push features that actually fire on this image (row_vals>eps)
                nz = torch.nonzero(row_vals > 1e-6, as_tuple=False).flatten().tolist()
                for f in nz:
                    entry = (float(row_vals[f]), gi, int(row_patch[f]))
                    h = heaps[f]
                    if len(h) < K:
                        heapq.heappush(h, entry)
                    elif entry[0] > h[0][0]:
                        heapq.heapreplace(h, entry)
            img_counter += B

    firing_rate = (fire / max(n_tokens, 1)).numpy()
    mean_act = (act_sum / fire.clamp(min=1)).numpy()
    max_act = act_max.numpy()
    dead = fire.numpy() == 0

    # ---- UMAP on the decoder dictionary ----
    try:
        import umap
        xy = umap.UMAP(n_components=2, metric="cosine", random_state=0).fit_transform(
            sae.dictionary.detach().cpu().numpy()).astype("float32")
    except Exception as e:
        print(f"UMAP unavailable ({e}); storing zeros -- `uv add umap-learn` to enable")
        xy = np.zeros((Fn, 2), dtype="float32")

    np.savez(out_dir / "features.npz",
             feature=np.arange(Fn), firing_rate=firing_rate, mean_act=mean_act,
             max_act=max_act, dead=dead, umap=xy)

    top_images = {
        str(f): [{"image_index": gi, "patch_index": pi, "score": sc}
                 for (sc, gi, pi) in sorted(heaps[f], key=lambda e: -e[0])]
        for f in range(Fn)
    }
    (out_dir / "top_images.json").write_text(json.dumps(top_images))

    meta = {
        "summary": {
            "model_key": spec.key, "dataset": args.dataset, "site": args.site,
            "split": args.split, "checkpoint": args.model, "bank": bank_arg,
            "n_features": int(Fn), "n_dead": int(dead.sum()), "n_tokens": int(n_tokens),
            "n_images": int(len(ds)), "top_k": K,
        },
        "sae": sae.spec.summary() if hasattr(sae.spec, "summary") else str(sae.spec),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote atlas -> {out_dir}\n  {Fn} features ({int(dead.sum())} dead), "
          f"{len(ds)} images, {n_tokens} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
