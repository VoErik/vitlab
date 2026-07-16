"""
Generate the standard figure set for a trained SAE (and its backbone).

    uv run python scripts/make_figures.py --sae saes/dinov2_fitz_l9 \
        --backbone runs/full-derm/dinov2-base/best --dataset fitzpatrick17k \
        --out out/figures/dinov2_l9

Produces (into --out, all PDF):
    dictionary_coherence.pdf     atom redundancy
    dictionary_babel.pdf         cumulative coherence
    firing_rates.pdf             latent firing-rate distribution
    reconstruction.pdf           true vs reconstructed
    concept_<latent>.pdf         top-activating images for each --latents
    token_geometry.pdf           Joseph-style geometry (with --token-geometry)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import vitlab
from vitlab import viz
from vitlab.eval.metrics import babel_function
from vitlab import ActivationStore, seed_everything
from vitlab.sae import load_layer_sae


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sae", type=Path, required=True)
    p.add_argument("--acts", type=Path, default=None, help="activation store (dictionary/recon figs)")
    p.add_argument("--backbone", type=Path, default=None, help="trained model dir (concept/geometry figs)")
    p.add_argument("--dataset", default=None)
    p.add_argument("--latents", type=int, nargs="*", default=[], help="latents to make evidence grids for")
    p.add_argument("--token-geometry", action="store_true")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="cap tokens for firing-rate/dead stats (default: whole store)")
    p.add_argument("--scatter-points", type=int, default=5000,
                   help="activation sample size for the reconstruction scatter")
    p.add_argument("--max-batches", type=int, default=100, help="cap for geometry extraction")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-root", type=Path, default=None)
    args = p.parse_args()

    if args.data_root is not None:
        vitlab.set_data_root(args.data_root)
    seed_everything(0)
    viz.use_style()
    args.out.mkdir(parents=True, exist_ok=True)

    sae = load_layer_sae(args.sae, device=args.device)
    site = sae.site
    print(f"SAE: {sae.spec.summary()}")

    viz.coherence_heatmap(sae.dictionary, save=args.out / "dictionary_coherence.pdf")
    viz.babel_curve(babel_function(sae.dictionary.detach().cpu(), k_max=64),
                    save=args.out / "dictionary_babel.pdf")
    print("  wrote dictionary_coherence.pdf, dictionary_babel.pdf")

    if args.acts is not None:
        from vitlab.eval import collect_codes

        store = ActivationStore(args.acts)
        codes, acts_s, recons_s = collect_codes(
            sae, store, device=args.device, max_tokens=args.max_tokens,
            keep_acts_sample=args.scatter_points,
        )
        n = codes.shape[0]
        viz.firing_rate_distribution(codes, save=args.out / "firing_rates.pdf")
        viz.activation_strength_vs_firing_rate(codes, color_by="max", save=args.out / "latent_scatter.pdf")
        viz.reconstruction_scatter(acts_s, recons_s, save=args.out / "reconstruction.pdf")
        n_dead = int((~(codes.abs() > 1e-6).any(0)).sum())
        print(f"  wrote firing_rates.pdf, latent_scatter.pdf, reconstruction.pdf "
              f"(over {n:,} tokens, {n_dead}/{codes.shape[1]} dead)")

    if args.backbone and args.dataset:
        from vitlab.activations import ActivationReader
        from vitlab.datasets import get_splits

        model = vitlab.load_model(args.backbone, device=args.device)
        reader = ActivationReader(model.backbone)
        train, _, _ = get_splits(args.dataset, model_key=model.spec.key)

        for latent in args.latents:
            viz.concept_evidence_grid(sae, reader, train, site, latent,
                                      top_k=9, device=args.device, model_key=model.spec.key,
                                      save=args.out / f"concept_{latent}.pdf")
            print(f"  wrote concept_{latent}.pdf")

        if args.token_geometry:
            from vitlab.batching import make_loader
            loader = make_loader(train, 32, shuffle=True, num_workers=4)
            viz.token_geometry_figure(reader, loader, max_batches=args.max_batches,
                                      device=args.device, title=f"{model.spec.key} token geometry",
                                      save=args.out / "token_geometry.pdf")
            print("  wrote token_geometry.pdf")

    print(f"\nfigures -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
