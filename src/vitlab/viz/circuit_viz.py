from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .style import finish


def visualize_circuit(circuit, *, figsize=(11, 8), max_edges=60, title=None,
                      show_labels=True, save=None, show=False):
    layers = sorted(circuit.nodes)
    if not layers:
        print("empty circuit"); return None

    pos = {}
    for li, layer in enumerate(sorted(layers, reverse=True)):
        nodes = circuit.nodes[layer]
        n = len(nodes)
        for j, node in enumerate(nodes):
            x = (j - (n - 1) / 2)
            pos[node.key()] = (x, li)

    fig, ax = plt.subplots(figsize=figsize)

    verified = getattr(circuit, "verified_edges", {})
    edges = sorted(circuit.edges, key=lambda e: -abs(e[2]))[:max_edges]
    if edges:
        wmax = max(abs(w) for _, _, w in edges) or 1.0
    for u, d, w in edges:
        if u.key() not in pos or d.key() not in pos:
            continue
        x0, y0 = pos[u.key()]; x1, y1 = pos[d.key()]
        frac = abs(w) / wmax
        color = plt.cm.viridis(frac)
        style = "-"
        if verified:
            style = "-" if abs(verified.get((u.key(), d.key()), 0)) > 1e-4 else ":"
        ax.plot([x0, x1], [y0, y1], color=color, lw=0.5 + 3 * frac, alpha=0.6, ls=style, zorder=1)

    nimps = [n.node_importance for layer in layers for n in circuit.nodes[layer]]
    vmax = max((abs(v) for v in nimps), default=1.0) or 1.0
    for layer in layers:
        for node in circuit.nodes[layer]:
            x, y = pos[node.key()]
            c = plt.cm.viridis(0.5 + 0.5 * node.node_importance / vmax)
            ax.scatter([x], [y], s=420, c=[c], edgecolors="black", linewidths=0.8, zorder=2)
            txt = f"L{node.layer}\n#{node.feature_idx}"
            if show_labels and node.label:
                txt = node.label
            ax.text(x, y, txt, ha="center", va="center", fontsize=6.5, zorder=3)

    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"layer {l}" for l in sorted(layers, reverse=True)])
    ax.set_xticks([])
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title(title or "SAE feature circuit"
                 + ("  (solid = causally verified)" if verified else ""))
    ax.margins(0.15)
    return finish(fig, save, show=show)


def circuit_metrics_curve(results, *, auc_faith=None, auc_comp=None, max_features=None,
                          title=None, save=None, show=False):
    """Faithfulness/completeness vs circuit size.
    """
    import numpy as np

    ks = sorted(results)
    xs = np.array(ks, dtype=float)
    if max_features:
        xs = xs / max_features
    faith = [results[k]["faithfulness"] for k in ks]
    comp = [results[k]["completeness"] for k in ks]

    fig, ax = plt.subplots()
    ax.plot(xs, faith, marker="o", color=plt.cm.viridis(0.2),
            label="faithfulness" + (f" (AUC {auc_faith:.3f})" if auc_faith is not None else ""))
    ax.plot(xs, comp, marker="s", color=plt.cm.viridis(0.6),
            label="completeness" + (f" (AUC {auc_comp:.3f})" if auc_comp is not None else ""))
    ax.set_xlabel("fraction of features in circuit")
    ax.set_ylabel("metric"); ax.set_ylim(-0.02, 1.02)
    if max_features:
        ax.set_xscale("log")
    ax.set_title(title or "circuit faithfulness & completeness vs size")
    ax.legend()
    return finish(fig, save, show=show)