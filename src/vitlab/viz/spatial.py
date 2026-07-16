from __future__ import annotations

import math
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from .style import finish


def infer_grid(num_patches: int) -> tuple[int, int]:
    h = int(round(math.sqrt(num_patches)))
    if h * h != num_patches:
        raise ValueError(
            f"{num_patches} patches is not a square grid. Pass grid=(H, W) explicitly "
            f"(non-square inputs or pooled tokens need it)."
        )
    return h, h


def patch_values_to_grid(
    values: torch.Tensor | np.ndarray,
    *,
    n_prefix: int = 0,
    grid: tuple[int, int] | None = None,
    has_prefix_in_values: bool = False,
) -> np.ndarray:
    """(T,) or (P,) per-token values -> (H, W) grid."""
    v = values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
    if has_prefix_in_values:
        v = v[n_prefix:]
    h, w = grid or infer_grid(v.shape[0])
    return v.reshape(h, w)


def activation_heatmap(
    patch_values,
    *,
    image=None,
    n_prefix: int = 0,
    grid: tuple[int, int] | None = None,
    has_prefix_in_values: bool = False,
    title: str | None = None,
    alpha: float = 0.6,
    cmap: str = "viridis",
    save=None,
    show: bool = False,
):
    """
    Overlay a per-patch quantity on the patch grid, optionally over the source image.
    """
    heat = patch_values_to_grid(
        patch_values, n_prefix=n_prefix, grid=grid, has_prefix_in_values=has_prefix_in_values
    )
    fig, ax = plt.subplots()

    if image is not None:
        img = np.asarray(image)
        ax.imshow(img)
        extent = (0, img.shape[1], img.shape[0], 0)
        ax.imshow(heat, cmap=cmap, alpha=alpha, extent=extent,
                  interpolation="nearest", aspect="auto")
    else:
        im = ax.imshow(heat, cmap=cmap, interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title or "patch activation")
    ax.axis("off")
    return finish(fig, save, show=show)


def spatial_cluster_map(
    labels,
    *,
    grid: tuple[int, int] | None = None,
    n_clusters: int | None = None,
    title: str = "spatial clusters",
    save=None,
    show: bool = False,
):
    """Show a per-patch integer label (e.g. k-means cluster) as a spatial map."""
    lab = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    h, w = grid or infer_grid(lab.shape[0])
    k = n_clusters or int(lab.max() + 1)

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([plt.cm.viridis(v) for v in np.linspace(0, 0.9, k)])

    fig, ax = plt.subplots()
    im = ax.imshow(lab.reshape(h, w), cmap=cmap, interpolation="nearest")
    fig.colorbar(im, ax=ax, ticks=range(k), fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.axis("off")
    return finish(fig, save, show=show)
