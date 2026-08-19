"""Render ``watermark_feature.pdf`` from a watermark_experiment.py results JSON.

    uv run python scripts/make_watermark_figure.py \
        --results out/watermark/results.json --out out/figures/watermark_feature.pdf

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

# viridis anchors, matching the existing thesis figures (watermark_feature.pdf,
# fst_features.pdf): feature = yellow-green, random control = teal,
# baselines = dark/mid purple.
C_FEAT = "#bddf26"    # feature (watermark) recall
C_FEAT_ACC = "#440154"  # feature accuracy
C_RAND = "#22a884"    # random control recall
C_RAND_ACC = "#756bb1"  # random control accuracy
C_CORRUPT = "#2a788e"  # corrupted baseline bar
C_CLEAN = "#440154"   # clean baseline bar


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
    """The values plotted in the current thesis figure (watermark_feature.pdf).

    Reproduces the existing figure exactly: ablation bars and the four steering
    traces (recall + accuracy, feature + random control) as reported in Results,
    *Causal Validation of Spurious Artifacts*.
    """
    return {
        "meta": {"target_name": "melanoma", "source": "thesis"},
        "ablation": {
            "clean_baseline": 0.105,
            "watermarked_baseline": 0.474,
            "random_ablation": 0.444,
            "watermark_ablation": 0.129,
        },
        "steering": {
            "alphas": [0.0, 1.0, 2.5, 5.0, 10.0],
            "watermark_recall":   [0.105, 0.810, 0.920, 1.000, 1.000],
            "watermark_accuracy": [0.790, 0.390, 0.160, 0.115, 0.120],
            "random_recall":      [0.105, 0.100, 0.170, 0.605, 0.880],
            "random_accuracy":    [0.790, 0.780, 0.690, 0.325, 0.120],
        },
    }


def render(results: dict, out: Path, *, title: str | None = None) -> Path:
    """Two-panel intervention figure, in the thesis' own convention.

    left   Activation Addition -- recall AND accuracy, for the feature direction
           and the random control, versus the steering coefficient. Plotting
           accuracy alongside recall is what makes off-manifold degradation at
           high alpha visible (a recall rise that comes with an accuracy collapse
           is model damage, not concept induction).
    right  Causal Excision -- four bars: clean baseline, corrupted baseline,
           random ablation (with its multi-seed error bar), feature ablation.
    """
    _apply_style()
    abl = results["ablation"]
    steer = results["steering"]
    target = results.get("meta", {}).get("target_name", "target")
    tname = target.capitalize()

    fig, (ax_s, ax_a) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # ---- left: activation addition (recall + accuracy) -------------------
    alphas = steer["alphas"]
    x = list(range(len(alphas)))          # categorical spacing, as in the thesis fig
    ax_s.plot(x, steer["watermark_recall"], marker="o", ls="-", color=C_FEAT,
              label="Feature Addition Recall")
    if "watermark_accuracy" in steer:
        ax_s.plot(x, steer["watermark_accuracy"], marker="^", ls="-", color=C_FEAT_ACC,
                  label="Feature Addition Accuracy")
    ax_s.plot(x, steer["random_recall"], marker="X", ls="--", color=C_RAND,
              label="Random Control Recall")
    if "random_accuracy" in steer:
        ax_s.plot(x, steer["random_accuracy"], marker="v", ls="--", color=C_RAND_ACC,
                  label="Random Control Accuracy")
    ax_s.set_xticks(x)
    ax_s.set_xticklabels([f"{a:g}" for a in alphas])
    ax_s.set_xlabel(r"Steering Coefficient ($\alpha$)")
    ax_s.set_ylabel("Metric")
    ax_s.set_title("Activation Addition")
    ax_s.set_ylim(-0.02, 1.05)

    # ---- right: causal excision (four bars) ------------------------------
    bars = [
        ("Clean\nBaseline", abl["clean_baseline"], C_CLEAN, 0.0),
        ("Corrupted\nBaseline", abl["watermarked_baseline"], C_CORRUPT, 0.0),
        ("Random\nAblation", abl["random_ablation"], C_RAND,
         abl.get("random_ablation_std", 0.0) or 0.0),
        ("Feature\nAblation", abl["watermark_ablation"], C_FEAT, 0.0),
    ]
    xs = list(range(len(bars)))
    ax_a.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], width=0.68,
             edgecolor="black", linewidth=0.6,
             yerr=[b[3] for b in bars], capsize=4,
             error_kw={"ecolor": "#333333", "lw": 1})
    for xi, b in zip(xs, bars):
        ax_a.text(xi, b[1] + b[3] + 0.02, f"{b[1]:.3f}", ha="center", va="bottom", fontsize=9)
    # clean baseline reference line
    ax_a.axhline(abl["clean_baseline"], color="#444444", ls=":", lw=1.2, zorder=0)
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels([b[0] for b in bars])
    ax_a.set_ylabel("Recall")
    ax_a.set_title("Causal Excision")
    ax_a.set_ylim(0, 1.05)
    ax_a.grid(axis="x", visible=False)

    # shared legend under the steering panel, as in the thesis figure
    handles, labels = ax_s.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.29, -0.13))

    if title:
        fig.suptitle(title, fontweight="bold")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
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
    p.add_argument("--title", default=None,
                   help=r"figure suptitle; use \n for a line break")
    p.add_argument("--out", type=Path, default=Path("out/figures/watermark_feature.pdf"))
    args = p.parse_args()

    if args.title:
        args.title = args.title.replace("\\n", "\n")

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