"""Depth sweep: run the watermark audit at every site

    uv run python scripts/watermark_layer_sweep.py \
        --run runs/dermamnist-watermarked/best \
        --bank saes/dinov2_dermamnist_alllayers \
        --out out/watermark/layer_sweep.json

"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

import vitlab
from vitlab.batching import make_loader
from vitlab.eval import watermark as wm
from vitlab.sae import discover_bank


def _layer_of(site: str) -> int:
    m = re.search(r"blocks\.(\d+)\.", site)
    return int(m.group(1)) if m else -1


def _loader(split, batch_size, workers):
    return make_loader(split, batch_size, shuffle=False, num_workers=workers)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="corrupted checkpoint dir")
    p.add_argument("--bank", type=Path, required=True, help="SAE bank tree (one SAE dir per layer)")
    p.add_argument("--site-kind", default="resid_post",
                   help="only sweep sites of this kind (resid_post | mlp_out | attn_out | ...)")
    p.add_argument("--dataset", default="dermamnist")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--target-class", type=int, default=None)
    p.add_argument("--wm-text", default="SFS")
    p.add_argument("--band", type=int, default=2)
    p.add_argument("--n-features", type=int, default=8)
    p.add_argument("--restrict-corners", action="store_true")
    p.add_argument("--identify-all-classes", action="store_true",
                   help="identify watermark latents on ALL classes marked (decorrelates the "
                        "artifact from the label; recommended on strictly correlated data)")
    p.add_argument("--cumulative", action="store_true",
                   help="also compute cumulative ablation: remove the watermark latents at all "
                        "sites up to layer L jointly (defeats downstream self-repair, gives a "
                        "monotone localisation curve)")
    p.add_argument("--control-seeds", type=int, default=5,
                   help="random-latent seeds to average the per-layer control over")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=Path("out/watermark/layer_sweep.json"))
    args = p.parse_args()

    if args.data_root is not None:
        vitlab.set_data_root(args.data_root)
    vitlab.seed_everything(args.seed)

    model = vitlab.load_model(args.run, device=args.device)
    bank = discover_bank(args.bank, device=args.device)
    task = model.task_names[0]

    sites = sorted(
        (s for s in bank.sites if args.site_kind in s), key=_layer_of
    )
    if not sites:
        raise SystemExit(f"no '{args.site_kind}' sites in bank {args.bank} (have: {bank.sites})")

    if args.target_class is None:
        target, names = wm.melanoma_index(args.dataset, data_root=args.data_root)
    else:
        target, names = args.target_class, None
    print(f"target class {target}" + (f" ({names[target]})" if names else ""))
    print(f"sweeping {len(sites)} sites: {sites}")

    wm_kwargs = {"text": args.wm_text}
    _, wm_val, wm_test = wm.corrupted_splits(
        args.dataset, model_key=model.spec.key, target_label=target,
        eval_watermarked=True, data_root=args.data_root, watermark_kwargs=wm_kwargs,
    )
    _, cl_val, cl_test = wm.corrupted_splits(
        args.dataset, model_key=model.spec.key, target_label=target,
        eval_watermarked=False, data_root=args.data_root, watermark_kwargs=wm_kwargs,
    )
    wm_loader = _loader({"val": wm_val, "test": wm_test}[args.split], args.batch_size, args.workers)
    clean_loader = _loader({"val": cl_val, "test": cl_test}[args.split], args.batch_size, args.workers)

    # constant references (no intervention)
    print("baselines ...")
    watermarked_baseline = wm.ablate_features(
        model, bank[sites[-1]], sites[-1], [], wm_loader,
        target_class=target, task=task, device=args.device,
    )["target_recall"]
    clean_baseline = wm.ablate_features(
        model, bank[sites[-1]], sites[-1], [], clean_loader,
        target_class=target, task=task, device=args.device,
    )["target_recall"]
    print(f"  watermarked baseline {watermarked_baseline:.3f} | clean baseline {clean_baseline:.3f}")

    # representational profile: identify watermark latents at every site (2 passes)
    print("identifying watermark latents across depth ...")
    if args.identify_all_classes:
        id_wm_split, id_clean_split = wm.identification_splits(
            args.dataset, model_key=model.spec.key, split=args.split,
            data_root=args.data_root, watermark_kwargs=wm_kwargs,
        )
        id_wm_loader = _loader(id_wm_split, args.batch_size, args.workers)
        id_clean_loader = _loader(id_clean_split, args.batch_size, args.workers)
    else:
        id_wm_loader, id_clean_loader = wm_loader, clean_loader
    ident = wm.identify_watermark_features_bank(
        model, bank, id_wm_loader, id_clean_loader,
        sites=sites, band=args.band, top_k=args.n_features, device=args.device,
    )

    # causal profile: per-site ablation
    print("per-layer ablation ...")
    seeds = list(range(args.control_seeds))
    restrict_map = {s: (ident[s].corner_tokens if args.restrict_corners else None) for s in sites}
    layers = []
    for i, site in enumerate(sites):
        feats = ident[site]
        restrict = restrict_map[site]
        wm_abl = wm.ablate_features(
            model, bank[site], site, feats.features, wm_loader,
            target_class=target, task=task, restrict_tokens=restrict, device=args.device,
        )
        rand_band = wm.random_ablation_band(
            model, bank[site], site, len(feats), wm_loader,
            target_class=target, task=task, exclude=feats.features, seeds=seeds,
            restrict_tokens=restrict, device=args.device,
        )
        row = {
            "layer": _layer_of(site),
            "site": site,
            "watermark_ablation": wm_abl["target_recall"],
            "random_ablation": rand_band["mean"],
            "random_ablation_std": rand_band["std"],
            "watermark_ablation_acc": wm_abl["accuracy"],
            "id_score": float(sum(feats.scores)),   # corner differential strength
            "watermark_features": feats.features,
        }

        if args.cumulative:
            # ablate the watermark latents at every site up to and including this one
            site_feats = {sites[j]: ident[sites[j]].features for j in range(i + 1)}
            cum = wm.ablate_features_multi(
                model, bank, site_feats, wm_loader,
                target_class=target, task=task, restrict_tokens=restrict, device=args.device,
            )
            row["watermark_ablation_cumulative"] = cum["target_recall"]

        layers.append(row)
        msg = (f"  L{row['layer']:>2} {site:<24} recall wm {row['watermark_ablation']:.3f} "
               f"| rand {row['random_ablation']:.3f}+/-{row['random_ablation_std']:.3f} "
               f"| id {row['id_score']:.3f}")
        if args.cumulative:
            msg += f" | cum {row['watermark_ablation_cumulative']:.3f}"
        print(msg)

    best = min(layers, key=lambda r: r["watermark_ablation"])
    print(f"\nmost causal single site: {best['site']} "
          f"(recall {best['watermark_ablation']:.3f} vs baseline {watermarked_baseline:.3f})")
    if args.cumulative:
        # first layer at which cumulative ablation reaches (near) the clean baseline
        thresh = clean_baseline + 0.05
        reached = next((r for r in layers if r["watermark_ablation_cumulative"] <= thresh), None)
        if reached is not None:
            print(f"shortcut removed by cumulative depth L{reached['layer']} "
                  f"(cum recall {reached['watermark_ablation_cumulative']:.3f} <= {thresh:.3f})")

    results = {
        "meta": {
            "run": str(args.run), "bank": str(args.bank), "site_kind": args.site_kind,
            "dataset": args.dataset, "split": args.split, "target_class": target,
            "target_name": (names[target] if names else str(target)),
            "watermark_text": args.wm_text, "band": args.band,
            "n_features": args.n_features, "restrict_corners": args.restrict_corners,
            "identify_all_classes": args.identify_all_classes,
            "cumulative": args.cumulative, "control_seeds": len(seeds),
            "most_causal_site": best["site"],
        },
        "clean_baseline": clean_baseline,
        "watermarked_baseline": watermarked_baseline,
        "layers": layers,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nresults -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())