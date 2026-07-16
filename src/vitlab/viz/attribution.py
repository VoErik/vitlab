from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .style import finish


def _prep_image(image_tensor, model_key=None):
    img = image_tensor.squeeze(0).detach().cpu()
    if model_key is not None:
        from ..datasets import denormalize
        img = denormalize(img, model_key)
    rgb = img.permute(1, 2, 0).numpy()
    rgb = (rgb - rgb.min()) / (np.ptp(rgb) + 1e-8)
    gray = rgb @ [0.2989, 0.5870, 0.1140]
    return rgb, gray


def _upscale(spatial_map, H, W):
    up = F.interpolate(spatial_map.unsqueeze(0).unsqueeze(0).float(), size=(H, W),
                       mode="bilinear", align_corners=False).squeeze().cpu().numpy() # TODO: possibly change mode?
    return (up - up.min()) / (np.ptp(up) + 1e-8)


def attribution_grid(image_tensor, result, *, model_key=None, top_k=None, alpha=0.5,
                     cmap="viridis", title=None, save=None, show=False):
    """Grid of the top attributed concepts, each map overlaid on the image."""
    n = len(result) if top_k is None else min(top_k, len(result))
    if n == 0:
        print("no concepts to visualise")
        return None
    rgb, gray = _prep_image(image_tensor, model_key)
    H, W = rgb.shape[:2]

    cols = min(5, n + 1)
    rows = 1 + math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.8 * rows))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")

    axes[0, 0].imshow(rgb); axes[0, 0].set_title(f"class {result.target_class}", fontsize=9)
    # aggregated map (score-weighted)
    if result.spatial_maps.numel():
        w = torch.tensor(result.scores[:n]).view(-1, 1, 1)
        agg = (result.spatial_maps[:n] * w).sum(0)
        axes[0, 1].imshow(gray, cmap="gray")
        axes[0, 1].imshow(_upscale(agg, H, W), cmap=cmap, alpha=alpha)
        axes[0, 1].set_title(f"aggregated (top {n})", fontsize=9)

    for i in range(n):
        r, c = 1 + i // cols, i % cols
        ax = axes[r, c]
        ax.imshow(gray, cmap="gray")
        ax.imshow(_upscale(result.spatial_maps[i], H, W), cmap=cmap, alpha=alpha)
        ax.set_title(f"#{i+1} {result.sites[i].split('.')[1] if '.' in result.sites[i] else ''}"
                     f"·F{result.features[i]}\nΔ={result.scores[i]:.3f}", fontsize=8)

    fig.suptitle(title or f"SAE attribution ({result.method})", fontweight="bold")
    return finish(fig, save, show=show)


def attribution_stats(result, *, title=None, save=None, show=False):
    """Ranked scores, per-site distribution, and cumulative curve."""
    n = len(result)
    if n == 0:
        print("no concepts to plot")
        return None
    scores = np.array(result.scores)
    sites = result.sites

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].bar(range(1, n + 1), scores, color=plt.cm.viridis(0.4))
    axes[0].set_xlabel("rank"); axes[0].set_ylabel("score / logit drop")
    axes[0].set_title("by rank")

    from collections import Counter
    by_site = Counter(sites)
    ss = sorted(by_site)
    axes[1].bar(range(len(ss)), [by_site[s] for s in ss], color=plt.cm.viridis(0.6))
    axes[1].set_xticks(range(len(ss)))
    axes[1].set_xticklabels([s.split(".")[1] if "." in s else s for s in ss], rotation=45, fontsize=7)
    axes[1].set_ylabel("count"); axes[1].set_title("concepts per site")

    axes[2].plot(range(1, n + 1), np.cumsum(scores), marker="o", color=plt.cm.viridis(0.3))
    axes[2].set_xlabel("rank"); axes[2].set_ylabel("cumulative")
    axes[2].set_title("cumulative attribution")

    fig.suptitle(title or f"attribution statistics ({result.method})", fontweight="bold")
    return finish(fig, save, show=show)


def token_group_trajectory(share, *, measure="gradient_mass", title=None, save=None, show=False):
    """Plot each token group's share of the decision across layers."""
    import numpy as np

    data = getattr(share, measure)
    layers = share.layers
    fig, ax = plt.subplots()

    if measure == "gradient_mass":
        # stacked area: fractions sum to 1 per layer
        bottom = np.zeros(len(layers))
        for i, g in enumerate(share.groups):
            vals = np.array(data[g])
            ax.fill_between(layers, bottom, bottom + vals, alpha=0.8,
                            label=g, color=plt.cm.viridis(i / max(1, len(share.groups) - 1)))
            bottom += vals
        ax.set_ylabel("fraction of gradient mass"); ax.set_ylim(0, 1)
    else: # ablation_drop
        for i, g in enumerate(share.groups):
            ax.plot(layers, data[g], marker="o", label=g,
                    color=plt.cm.viridis(i / max(1, len(share.groups) - 1)))
        ax.set_ylabel("logit drop when group ablated")
        ax.axhline(0, color="gray", lw=0.5)

    ax.set_xlabel("layer")
    ax.set_title(title or f"decision share by token group ({measure})")
    ax.legend()
    return finish(fig, save, show=show)