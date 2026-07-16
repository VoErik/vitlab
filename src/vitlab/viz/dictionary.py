from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from .style import finish

# TODO: dunno if useful
def coherence_heatmap(dictionary, *, max_atoms=200, title="atom cosine similarity",
                      save=None, show=False):
    """|cosine| between decoder atoms. Bright off-diagonal blocks = redundant
    atoms. Subsampled to `max_atoms` so a 16k dictionary stays legible."""
    import torch.nn.functional as F

    D = dictionary.detach()
    if D.shape[0] > max_atoms:
        idx = torch.randperm(D.shape[0])[:max_atoms]
        D = D[idx]
    G = (F.normalize(D, dim=1) @ F.normalize(D, dim=1).T).abs().cpu().numpy()
    fig, ax = plt.subplots()
    im = ax.imshow(G, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title); ax.set_xlabel("atom"); ax.set_ylabel("atom")
    return finish(fig, save, show=show)


def babel_curve(curve, *, title="Babel function (cumulative coherence)", save=None, show=False):
    c = curve.detach().cpu().numpy() if torch.is_tensor(curve) else np.asarray(curve)
    fig, ax = plt.subplots()
    ax.plot(range(1, len(c) + 1), c)
    ax.set_xlabel("k (neighbours)"); ax.set_ylabel(r"$\mu_1(k)$"); ax.set_title(title)
    return finish(fig, save, show=show)


def concept_umap(dictionary, *, labels=None, n_neighbors=15, min_dist=0.1,
                 title="decoder atoms (UMAP)", save=None, show=False):
    """2-D UMAP of decoder atoms, optionally coloured by a per-atom label."""
    try:
        import umap
    except ImportError:
        raise ImportError("concept_umap needs umap-learn (uv add umap-learn)") # TODO: fix import
    D = dictionary.detach().cpu().numpy()
    emb = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42).fit_transform(D)
    fig, ax = plt.subplots()
    if labels is None:
        ax.scatter(emb[:, 0], emb[:, 1], s=6, alpha=0.6, c=range(len(emb)), cmap="viridis")
    else:
        lab = np.asarray(labels)
        sc = ax.scatter(emb[:, 0], emb[:, 1], s=6, alpha=0.7, c=lab, cmap="viridis")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    return finish(fig, save, show=show)


def metrics_by_layer(reports: dict[str, dict], metric: str = "r2",
                     *, title=None, save=None, show=False):
    """Plot one metric across layers, given {site: dictionary_report}."""
    def block_idx(site):
        parts = site.split(".")
        return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

    sites = sorted(reports, key=block_idx)
    xs = [block_idx(s) for s in sites]
    ys = [reports[s].get(metric, reports[s].get("dictionary", {}).get(metric)) for s in sites]
    fig, ax = plt.subplots()
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("layer"); ax.set_ylabel(metric); ax.set_title(title or f"{metric} by layer")
    return finish(fig, save, show=show)
