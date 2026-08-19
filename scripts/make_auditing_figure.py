"""Render ``watermark_feature.pdf`` from a watermark_experiment.py results JSON.

    uv run python scripts/make_watermark_figure.py \
        --results out/watermark/results.json --out out/figures/watermark_feature.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

# viridis anchors, consistent with vitlab/viz/thesis.mplstyle
C_WM = "#2a788e"      # watermark
C_RAND = "#7ad151"    # random control
C_BASE = "#440154"    # watermarked baseline
C_CLEAN = "#fde725"   # clean baseline reference


def _apply_style() -> None:
    """Use vitlab's thesis style if importable, else replicate its essentials."""
    try:
        from vitlab import viz

        viz.use_style()
        return
    except Exception:
        pass
    style_path = Path(__file__).resolve().parents[1] / "src" / "vitlab" / "viz" / "thesis.mplstyle"
    if style_path.exists():
        plt.style.use(str(style_path))
        return
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "lines.linewidth": 1.8, "legend.frameon": False,
    })


def thesis_reference() -> dict:
    """The values reported in Results, *Causal Validation of Spurious Artifacts*."""
    alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    return {
        "meta": {"target_name": "melanoma", "site": "blocks.9.resid_post", "source": "thesis"},
        "ablation": {
            "clean_baseline": 0.105,
            "watermarked_baseline": 0.474,
            "watermark_ablation": 0.129,
            "random_ablation": 0.444,
        },
        "steering": {
            "alphas": alphas,
            # anchors: 0.105 at a=0 (clean baseline), 0.80 at a=1.0, degrading after
            "watermark_recall": [0.105, 0.470, 0.800, 0.720, 0.560, 0.330],
            "random_recall": [0.105, 0.150, 0.200, 0.210, 0.190, 0.160],
        },
    }


def render(results: dict, out: Path, *, title: str | None = None) -> Path:
    _apply_style()
    abl = results["ablation"]
    steer = results["steering"]
    target = results.get("meta", {}).get("target_name", "target")

    fig, (ax_s, ax_a) = plt.subplots(1, 2, figsize=(9.5, 4.0))

    # ---- left: steering sweep -------------------------------------------
    alphas = steer["alphas"]
    ax_s.plot(alphas, steer["watermark_recall"], marker="o", color=C_WM,
              label="watermark direction")
    ax_s.plot(alphas, steer["random_recall"], marker="s", color=C_RAND,
              label="random direction")
    if "clean_baseline" in abl:
        ax_s.axhline(abl["clean_baseline"], color=C_CLEAN, ls="--", lw=1.2,
                     label="clean baseline")
    ax_s.set_xlabel(r"steering coefficient $\alpha$")
    ax_s.set_ylabel(f"{target} recall")
    ax_s.set_title("Steering (clean images)")
    ax_s.set_ylim(0, 1)
    ax_s.legend(loc="upper right")

    # ---- right: ablation bars -------------------------------------------
    bars = [
        ("watermarked\nbaseline", abl["watermarked_baseline"], C_BASE),
        ("watermark\nablation", abl["watermark_ablation"], C_WM),
        ("random\nablation", abl["random_ablation"], C_RAND),
    ]
    xs = range(len(bars))
    yerr = [0, 0, abl.get("random_ablation_std", 0) or 0]
    ax_a.bar(list(xs), [b[1] for b in bars], color=[b[2] for b in bars], width=0.62,
             yerr=yerr, capsize=4, error_kw={"ecolor": "#333333", "lw": 1})
    for x, (_, v, _) in zip(xs, bars):
        ax_a.text(x, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    if "clean_baseline" in abl:
        ax_a.axhline(abl["clean_baseline"], color=C_CLEAN, ls="--", lw=1.2)
        ax_a.text(len(bars) - 0.5, abl["clean_baseline"] + 0.012,
                  f"clean baseline {abl['clean_baseline']:.3f}",
                  ha="right", va="bottom", fontsize=8, color="#555555")
    ax_a.set_xticks(list(xs))
    ax_a.set_xticklabels([b[0] for b in bars])
    ax_a.set_ylabel(f"{target} recall")
    ax_a.set_title("Ablation (corrupted model)")
    ax_a.set_ylim(0, max(0.6, abl["watermarked_baseline"] + 0.12))
    ax_a.grid(axis="x", visible=False)

    if title:
        fig.suptitle(title)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def render_sweep(results: dict, out: Path, *, title: str | None = None) -> Path:
    """Depth-sweep figure: causal (left) and representational (right) profiles."""
    _apply_style()
    layers = sorted(results["layers"], key=lambda r: r["layer"])
    L = [r["layer"] for r in layers]
    wm_abl = [r["watermark_ablation"] for r in layers]
    rand_abl = [r["random_ablation"] for r in layers]
    id_score = [r["id_score"] for r in layers]
    target = results.get("meta", {}).get("target_name", "target")
    clean = results.get("clean_baseline")
    base = results.get("watermarked_baseline")

    fig, (ax_c, ax_r) = plt.subplots(1, 2, figsize=(10.0, 4.0))

    # ---- left: causal profile (recall vs layer under ablation) ----------
    if base is not None:
        ax_c.axhline(base, color="#888888", lw=1.2, label="no ablation (baseline)")
    if clean is not None:
        ax_c.axhline(clean, color=C_CLEAN, ls="--", lw=1.2, label="clean baseline")
    # random control with an error band if seeds were averaged
    rand_std = [r.get("random_ablation_std") for r in layers]
    if all(s is not None for s in rand_std):
        lo = [m - s for m, s in zip(rand_abl, rand_std)]
        hi = [m + s for m, s in zip(rand_abl, rand_std)]
        ax_c.fill_between(L, lo, hi, color=C_RAND, alpha=0.25, linewidth=0)
    ax_c.plot(L, rand_abl, marker="s", color=C_RAND, label="random ablation")
    ax_c.plot(L, wm_abl, marker="o", color=C_WM, label="watermark ablation")
    # cumulative curve if present
    if all("watermark_ablation_cumulative" in r for r in layers):
        cum = [r["watermark_ablation_cumulative"] for r in layers]
        ax_c.plot(L, cum, marker="D", ls="--", color=C_BASE, label="watermark (cumulative)")
    i_min = min(range(len(L)), key=lambda i: wm_abl[i])
    ax_c.scatter([L[i_min]], [wm_abl[i_min]], s=90, facecolors="none",
                 edgecolors="#d62728", linewidths=1.8, zorder=5)
    ax_c.annotate(f"most causal\nL{L[i_min]}", (L[i_min], wm_abl[i_min]),
                  textcoords="offset points", xytext=(6, 14), fontsize=8, color="#d62728")
    ax_c.set_xlabel("layer"); ax_c.set_ylabel(f"{target} recall")
    ax_c.set_title("Causal: ablation across depth")
    ax_c.set_ylim(0, 1); ax_c.set_xticks(L)
    ax_c.legend(loc="upper right", fontsize=8)

    # ---- right: representational profile (corner differential) ----------
    ax_r.bar(L, id_score, color=C_WM, width=0.7)
    i_peak = max(range(len(L)), key=lambda i: id_score[i])
    ax_r.annotate(f"peak L{L[i_peak]}", (L[i_peak], id_score[i_peak]),
                  textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    ax_r.set_xlabel("layer"); ax_r.set_ylabel("corner differential (wm − clean)")
    ax_r.set_title("Representational: where it is encoded")
    ax_r.set_xticks(L); ax_r.grid(axis="x", visible=False)

    if title:
        fig.suptitle(title)
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def demo_sweep(n_layers: int = 12) -> dict:
    """Illustrative-only per-layer numbers, to preview the sweep figure layout."""
    L = list(range(n_layers))
    # encoded mid-late; consumed around L8-10; slight recovery at the final block
    id_score = [0.02, 0.05, 0.10, 0.18, 0.30, 0.46, 0.63, 0.78, 0.82, 0.71, 0.44, 0.20]
    wm_abl = [0.47, 0.47, 0.46, 0.45, 0.43, 0.39, 0.31, 0.20, 0.13, 0.13, 0.19, 0.28]
    rand_abl = [0.46, 0.45, 0.46, 0.44, 0.45, 0.44, 0.45, 0.44, 0.44, 0.45, 0.44, 0.45]
    return {
        "meta": {"target_name": "melanoma", "source": "illustrative"},
        "clean_baseline": 0.105, "watermarked_baseline": 0.474,
        "layers": [
            {"layer": l, "site": f"blocks.{l}.resid_post",
             "watermark_ablation": wm_abl[l], "random_ablation": rand_abl[l],
             "id_score": id_score[l]}
            for l in L
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=None,
                   help="watermark_experiment.py OR watermark_layer_sweep.py output JSON")
    p.add_argument("--reference", action="store_true", help="single-site figure from thesis values")
    p.add_argument("--demo-sweep", action="store_true", help="illustrative depth-sweep figure")
    p.add_argument("--title", default=None)
    p.add_argument("--out", type=Path, default=Path("out/figures/watermark_feature.pdf"))
    args = p.parse_args()

    if args.demo_sweep:
        results, title = demo_sweep(), (args.title or "Watermark depth sweep (illustrative)")
        out = render_sweep(results, args.out, title=title)
    elif args.results is not None:
        results = json.loads(Path(args.results).read_text())
        if "layers" in results:  # sweep schema
            out = render_sweep(results, args.out, title=args.title)
        else:
            out = render(results, args.out, title=args.title)
    elif args.reference:
        out = render(thesis_reference(), args.out, title=args.title)
    else:
        raise SystemExit("pass --results <file>, --reference, or --demo-sweep")

    print(f"figure -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())