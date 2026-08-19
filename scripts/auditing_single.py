"""Run the watermark audit and write a results JSON for the figure.

Given the corrupted model (from train_watermark_head.py) and an SAE trained at
the intervention site, this:

  1. builds a watermarked test loader (shortcut active) and a clean one,
  2. locates the watermark latents in the corner tokens,
  3. ablates them -> melanoma recall, vs a random-latent control of equal size,
  4. steers clean images along the watermark direction over a sweep of alpha,
     vs a random-direction control.

    uv run python scripts/watermark_experiment.py \
        --run runs/dermamnist-watermarked/best \
        --sae saes/dinov2_dermamnist_l9 --site blocks.9.resid_post \
        --out out/watermark/results.json

Multi-site steering: pass a comma-separated ``--site`` list and a ``--bank`` tree. The watermark direction is
injected at every listed site jointly (per-site feature-scaled, corner-restricted
if ``--restrict-corners``); ablation still runs at the first site.

    uv run python scripts/watermark_experiment.py \
        --run runs/dinov3-watermarked/best --bank saes/dinov3_dermamnist_corrupt \
        --site blocks.0.resid_post,blocks.1.resid_post \
        --identify-all-classes --restrict-corners --steer-scale feature \
        --alphas 0 1 2 4 8 --out out/watermark/dinov3_multisite.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import vitlab
from vitlab.batching import make_loader
from vitlab.eval import watermark as wm
from vitlab.sae import discover_bank, load_layer_sae


def _loader(split, batch_size, workers):
    return make_loader(split, batch_size, shuffle=False, num_workers=workers)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="corrupted checkpoint dir (has config.json)")
    p.add_argument("--sae", type=Path, default=None,
                   help="single SAE dir (sae.pt + normalizer.pt) at --site; use --bank for >1 site")
    p.add_argument("--bank", type=Path, default=None,
                   help="SAE bank tree; required when --site lists several sites (multi-site steering)")
    p.add_argument("--site", default="blocks.9.resid_post",
                   help="one site, or a comma-separated list to steer at several sites jointly. "
                        "Ablation always runs at the first site.")
    p.add_argument("--dataset", default="dermamnist")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--target-class", type=int, default=None, help="default: auto-detect melanoma")
    p.add_argument("--wm-text", default="SFS")
    p.add_argument("--band", type=int, default=2, help="corner block half-width in patches")
    p.add_argument("--n-features", type=int, default=8, help="how many watermark latents to excise")
    p.add_argument("--restrict-corners", action="store_true",
                   help="shorthand: restrict BOTH ablation and steering to the corner tokens")
    p.add_argument("--ablate-restrict-corners", action="store_true",
                   help="restrict only the ablation to corner tokens. Usually unnecessary: "
                        "ablation is activation-gated, so it already only acts where the feature "
                        "fires, and restricting it misses evidence attention copied elsewhere.")
    p.add_argument("--steer-restrict-corners", action="store_true",
                   help="restrict only the steering injection to corner tokens. Recommended: "
                        "steering is additive and ungated, so injecting across all patches paints "
                        "the artifact where the model never sees it (off-manifold).")
    p.add_argument("--identify-all-classes", action="store_true",
                   help="identify watermark latents on ALL classes marked (decorrelates the "
                        "artifact from the label; recommended on strictly correlated data)")
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 2.0, 4.0, 6.0])
    p.add_argument("--steer-scale", choices=["feature", "token_norm"], default="feature",
                   help="feature: alpha is multiples of the watermark's natural strength "
                        "(control is norm-matched, degrades gracefully). token_norm: alpha is "
                        "a fraction of the residual norm (alpha=1 is a full-magnitude push).")
    p.add_argument("--control-seeds", type=int, default=5,
                   help="number of random-latent seeds to average the ablation control over")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=Path("out/watermark/results.json"))
    args = p.parse_args()

    if args.data_root is not None:
        vitlab.set_data_root(args.data_root)
    vitlab.seed_everything(args.seed)

    model = vitlab.load_model(args.run, device=args.device)
    task = model.task_names[0]

    sites = [s.strip() for s in args.site.split(",") if s.strip()]
    primary = sites[0]
    multi = len(sites) > 1
    if multi and args.bank is None:
        raise SystemExit("--site lists several sites; pass --bank <tree> for multi-site steering")
    if args.bank is not None:
        bank = discover_bank(args.bank, device=args.device)
        missing = [s for s in sites if s not in bank.sites]
        if missing:
            raise SystemExit(f"sites not in bank: {missing} (have {bank.sites})")
        layer_saes = {s: bank[s] for s in sites}
    else:
        if args.sae is None:
            raise SystemExit("pass --sae <dir> (single site) or --bank <tree> (one or more sites)")
        layer_saes = {primary: load_layer_sae(args.sae, device=args.device)}
    primary_sae = layer_saes[primary]

    if args.target_class is None:
        target, names = wm.melanoma_index(args.dataset, data_root=args.data_root)
    else:
        target, names = args.target_class, None
    print(f"target class {target}" + (f" ({names[target]})" if names else ""))

    wm_kwargs = {"text": args.wm_text}
    # watermarked + clean versions of the same evaluation split
    _, wm_val, wm_test = wm.corrupted_splits(
        args.dataset, model_key=model.spec.key, target_label=target,
        eval_watermarked=True, data_root=args.data_root, watermark_kwargs=wm_kwargs,
    )
    _, cl_val, cl_test = wm.corrupted_splits(
        args.dataset, model_key=model.spec.key, target_label=target,
        eval_watermarked=False, data_root=args.data_root, watermark_kwargs=wm_kwargs,
    )
    wm_split = {"val": wm_val, "test": wm_test}[args.split]
    cl_split = {"val": cl_val, "test": cl_test}[args.split]
    wm_loader = _loader(wm_split, args.batch_size, args.workers)
    clean_loader = _loader(cl_split, args.batch_size, args.workers)

    # 1) locate the watermark latents (corner activation: watermarked - clean)
    print(f"identifying watermark latents at {sites} ...")
    if args.identify_all_classes:
        id_wm_split, id_clean_split = wm.identification_splits(
            args.dataset, model_key=model.spec.key, split=args.split,
            data_root=args.data_root, watermark_kwargs=wm_kwargs,
        )
        id_wm_loader = _loader(id_wm_split, args.batch_size, args.workers)
        id_clean_loader = _loader(id_clean_split, args.batch_size, args.workers)
    else:
        id_wm_loader, id_clean_loader = wm_loader, clean_loader

    if multi:
        ident = wm.identify_watermark_features_bank(
            model, bank, id_wm_loader, id_clean_loader,
            sites=sites, band=args.band, top_k=args.n_features, device=args.device,
        )
    else:
        ident = {primary: wm.identify_watermark_features(
            model, primary_sae, primary, id_wm_loader, id_clean_loader,
            band=args.band, top_k=args.n_features, device=args.device,
        )}
    feats = ident[primary]
    for s in sites:
        print(f"  {s}: {ident[s].features}")

    abl_corners = args.restrict_corners or args.ablate_restrict_corners
    steer_corners = args.restrict_corners or args.steer_restrict_corners
    restrict = feats.corner_tokens if abl_corners else None
    seeds = list(range(args.control_seeds))

    # 2) ablation on the watermarked split (shortcut active) -- at the primary site
    print(f"ablation (site {primary}) ...")
    baseline = wm.ablate_features(
        model, primary_sae, primary, [], wm_loader,
        target_class=target, task=task, device=args.device,
    )
    wm_abl = wm.ablate_features(
        model, primary_sae, primary, feats.features, wm_loader,
        target_class=target, task=task, restrict_tokens=restrict, device=args.device,
    )
    rand_band = wm.random_ablation_band(
        model, primary_sae, primary, len(feats), wm_loader,
        target_class=target, task=task, exclude=feats.features, seeds=seeds,
        restrict_tokens=restrict, device=args.device,
    )
    # clean baseline: corrupted model on un-watermarked images (no shortcut)
    clean_baseline = wm.ablate_features(
        model, primary_sae, primary, [], clean_loader,
        target_class=target, task=task, device=args.device,
    )
    print(f"  clean baseline recall      {clean_baseline['target_recall']:.3f}")
    print(f"  watermarked baseline       {baseline['target_recall']:.3f}")
    print(f"  watermark-ablation recall  {wm_abl['target_recall']:.3f}")
    print(f"  random-ablation recall     {rand_band['mean']:.3f} +/- {rand_band['std']:.3f} "
          f"(n={len(seeds)}, range {rand_band['min']:.3f}-{rand_band['max']:.3f})")

    # 3) steering on clean images along the watermark direction
    where = "+".join(s.split(".")[1] for s in sites) if multi else primary
    print(f"steering sweep (scale={args.steer_scale}, sites={where}) ...")
    if args.steer_scale == "feature":
        norm_scale = False
        wm_vecs = {s: wm.steering_vector(layer_saes[s], ident[s].features, ident[s].scores)
                   for s in sites}
        rand_vecs = {s: wm.random_direction(layer_saes[s], seed=args.seed, like=wm_vecs[s])
                     for s in sites}
    else:
        norm_scale = True
        wm_vecs = {s: wm.watermark_direction(layer_saes[s], ident[s].features) for s in sites}
        rand_vecs = {s: wm.random_direction(layer_saes[s], seed=args.seed) for s in sites}
    restrict_map = {s: (ident[s].corner_tokens if steer_corners else None) for s in sites}

    steer_wm, steer_rand = [], []
    for a in args.alphas:
        r_wm = wm.steer_multi(
            model, wm_vecs, clean_loader, alpha=a, target_class=target, task=task,
            restrict_map=restrict_map, norm_scale=norm_scale, device=args.device,
        )
        r_rand = wm.steer_multi(
            model, rand_vecs, clean_loader, alpha=a, target_class=target, task=task,
            restrict_map=restrict_map, norm_scale=norm_scale, device=args.device,
        )
        steer_wm.append(r_wm)
        steer_rand.append(r_rand)
        print(f"  alpha {a:>4}: watermark {r_wm['target_recall']:.3f} (acc {r_wm['accuracy']:.3f})  "
              f"random {r_rand['target_recall']:.3f} (acc {r_rand['accuracy']:.3f})")

    results = {
        "meta": {
            "run": str(args.run), "sae": str(args.sae), "bank": str(args.bank),
            "sites": sites, "primary_site": primary, "site": args.site,
            "dataset": args.dataset, "split": args.split, "target_class": target,
            "target_name": (names[target] if names else str(target)),
            "watermark_text": args.wm_text, "band": args.band,
            "n_features": len(feats),
            "ablate_restrict_corners": abl_corners, "steer_restrict_corners": steer_corners,
            "identify_all_classes": args.identify_all_classes,
            "steer_scale": args.steer_scale, "control_seeds": len(seeds),
            "watermark_features": feats.features,
        },
        "ablation": {
            "clean_baseline": clean_baseline["target_recall"],
            "watermarked_baseline": baseline["target_recall"],
            "watermark_ablation": wm_abl["target_recall"],
            "random_ablation": rand_band["mean"],
            "random_ablation_std": rand_band["std"],
            "random_ablation_recalls": rand_band["recalls"],
            "accuracy": {
                "watermarked_baseline": baseline["accuracy"],
                "watermark_ablation": wm_abl["accuracy"],
            },
        },
        "steering": {
            "scale": args.steer_scale,
            "sites": sites,
            "alphas": list(args.alphas),
            "watermark_recall": [r["target_recall"] for r in steer_wm],
            "random_recall": [r["target_recall"] for r in steer_rand],
            "watermark_accuracy": [r["accuracy"] for r in steer_wm],
            "random_accuracy": [r["accuracy"] for r in steer_rand],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nresults -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())