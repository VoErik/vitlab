"""
Token-geometry analysis (Joseph et al. replication).

Pipeline:
  extract_layer_tokens  -> (N, L, T, D) residual-stream activations
  alpha_map             -> (L, T) eigenspectrum power-law exponent per layer/token
  then UMAP + k-means over per-patch alpha trajectories, and inter-token correlation.
"""

from __future__ import annotations

import numpy as np
import torch

from .style import finish


@torch.no_grad()
def extract_layer_tokens(reader, loader, *, layers=None, max_batches=None, device="cuda"):
    """
    (N, L, T, D) residual-stream activations across layers.

    layers: block indices to probe (default: all). Uses resid_post per block.
    """
    reader.backbone.to(device).eval()
    if layers is None:
        layers = list(range(reader.backbone.n_layers))
    sites = [f"blocks.{l}.resid_post" for l in layers]

    batches = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        px = (batch["pixel_values"] if isinstance(batch, dict) else batch[0]).to(device)
        cache = reader.read(px, sites)
        per_layer = [cache[s] if isinstance(cache, dict) else cache for s in sites]
        batches.append(torch.stack(per_layer, dim=1).cpu())   # (B, L, T, D)
    return torch.cat(batches, dim=0)


def alpha_signature(X: torch.Tensor, k_min: int = 10, k_max: int = 200) -> float:
    """
    Power-law exponent alpha of the covariance eigenspectrum of X (N, D).
    Larger alpha = faster spectral decay = lower effective dimensionality.
    """
    X = X - X.mean(dim=0, keepdim=True)
    cov = (X.T @ X) / (X.shape[0] - 1)
    eigvals = torch.linalg.eigvalsh(cov).flip(0)
    actual_max = min(k_max, len(eigvals))
    if actual_max <= k_min:
        return np.nan
    vals = eigvals[k_min:actual_max].cpu().numpy()
    if np.any(vals <= 1e-9):
        return np.nan
    j = np.arange(k_min, actual_max) + 1
    slope, _ = np.polyfit(np.log(j), np.log(vals), 1)
    return float(-slope)


def alpha_map(acts: torch.Tensor, *, k_min=10, k_max=200) -> np.ndarray:
    """(N, L, T, D) -> (L, T) alpha per layer and token."""
    N, L, T, D = acts.shape
    out = np.zeros((L, T))
    for l in range(L):
        for t in range(T):
            out[l, t] = alpha_signature(acts[:, l, t, :], k_min, k_max)
    return out


def inter_token_correlation(acts: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of off-diagonal inter-token Pearson correlation per layer."""
    N, L, T, D = acts.shape
    means, stds = [], []
    for l in range(L):
        X = acts[:, l, :, :].permute(1, 0, 2).reshape(T, -1)   # (T, N*D)
        C = torch.corrcoef(X)
        off = C[torch.tril(torch.ones_like(C), diagonal=-1).bool()]
        means.append(off.mean().item()); stds.append(off.std().item())
    return np.array(means), np.array(stds)


def token_geometry_figure(reader, loader, *, layers=None, max_batches=200,
                          n_clusters=5, k_min=10, k_max=200, device="cuda",
                          title=None, save=None, show=False):
    """
    The full four-panel Joseph replication for one backbone.

    (a) alpha trajectories per spatial cluster + CLS
    (b) UMAP of per-patch alpha trajectories
    (c) spatial layout of the clusters
    (d) inter-token correlation across layers
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from sklearn.cluster import KMeans

    acts = extract_layer_tokens(reader, loader, layers=layers, max_batches=max_batches, device=device)
    N, L, T, D = acts.shape
    prefix = reader.spec.n_prefix_tokens
    n_patches = T - prefix
    H = W = int(np.sqrt(n_patches))
    if H * W != n_patches:
        raise ValueError(f"{n_patches} patches not square; token-geometry map needs a square grid")

    amap = alpha_map(acts, k_min=k_min, k_max=k_max)
    cls_traj = amap[:, 0]
    spatial = amap[:, prefix:]

    try:
        import umap
        emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(spatial.T)
    except ImportError: # TODO: fix umap import
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2).fit_transform(spatial.T)
    clusters = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(spatial.T)
    colors = [plt.cm.viridis(v) for v in np.linspace(0, 0.9, n_clusters)]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(L), cls_traj, "k--", lw=2, label="CLS")
    for c in range(n_clusters):
        cd = spatial[:, clusters == c]
        m, s = np.nanmean(cd, 1), np.nanstd(cd, 1)
        ax1.plot(range(L), m, color=colors[c], lw=2, label=f"cluster {c}")
        ax1.fill_between(range(L), m - s, m + s, color=colors[c], alpha=0.2)
    ax1.set_xlabel("layer"); ax1.set_ylabel(r"$\alpha$"); ax1.set_title("(a) intra-token geometry")
    ax1.legend(fontsize=7)

    ax2 = fig.add_subplot(gs[0, 1])
    for c in range(n_clusters):
        m = clusters == c
        ax2.scatter(emb[m, 0], emb[m, 1], c=[colors[c]], s=12, alpha=0.8, label=f"{c}")
    ax2.set_title(r"(b) UMAP of $\alpha$ trajectories"); ax2.set_xticks([]); ax2.set_yticks([])

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(clusters.reshape(H, W), cmap=mcolors.ListedColormap(colors), interpolation="nearest")
    ax3.set_title("(c) cluster spatial map"); ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, :])
    m, s = inter_token_correlation(acts)
    ax4.plot(range(L), m, "k-o")
    ax4.fill_between(range(L), m - s, m + s, color="gray", alpha=0.2)
    ax4.set_xlabel("layer"); ax4.set_ylabel(r"Pearson $r$"); ax4.set_title("(d) inter-token correlation")

    fig.suptitle(title or "token geometry")
    return finish(fig, save, show=show)
