from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from .style import finish


def latent_activation_histogram(codes, latent: int, *, bins=50,
                                title=None, save=None, show=False):
    """Firing distribution of one latent over a batch of codes."""
    z = codes[:, latent]
    z = z.detach().cpu().numpy() if torch.is_tensor(z) else np.asarray(z)
    active = z[z > 0]
    fig, ax = plt.subplots()
    if active.size:
        ax.hist(active, bins=bins, color=plt.cm.viridis(0.5))
    fire_rate = (z > 0).mean()
    ax.set_title(title or f"latent {latent}: fires on {fire_rate*100:.1f}% of tokens")
    ax.set_xlabel("activation"); ax.set_ylabel("count")
    return finish(fig, save, show=show)


def firing_rate_distribution(codes, *, title="latent firing rates", save=None, show=False):
    """Histogram of per-latent firing rates over the batch."""
    z = codes.detach().cpu() if torch.is_tensor(codes) else torch.as_tensor(codes)
    n_tokens = z.shape[0]
    rates = (z > 0).float().mean(0).numpy()
    fig, ax = plt.subplots()
    ax.hist(rates, bins=50, color=plt.cm.viridis(0.4))
    ax.set_xlabel("fraction of tokens the latent fires on"); ax.set_ylabel("num latents")
    ax.set_yscale("log")
    ax.set_title(f"{title}  (dead: {(rates == 0).mean()*100:.1f}% over {n_tokens:,} tokens)")
    return finish(fig, save, show=show)


def activation_strength_vs_firing_rate(codes, *, color_by="max", title="latent activation vs firing rate",
                                       save=None, show=False):
    """
    The standard SAE health scatter.

    color_by: "max" (peak activation) | "count" (fire count) | an array of len n_latents.
    """
    z = codes.detach().cpu() if torch.is_tensor(codes) else torch.as_tensor(codes)
    N, F = z.shape
    active = z > 0
    fire_rate = active.float().mean(0).numpy()
    summed = (z * active).sum(0)
    counts = active.sum(0).clamp_min(1)
    mean_strength = (summed / counts).numpy()

    alive = fire_rate > 0
    fr = fire_rate[alive]; ms = mean_strength[alive]

    if color_by == "max":
        c = z.max(0).values.numpy()[alive]; clabel = "peak activation"
    elif color_by == "count":
        c = active.sum(0).numpy()[alive]; clabel = "fire count"
    else:
        c = np.asarray(color_by)[alive]; clabel = "value"

    fig, ax = plt.subplots()
    sc = ax.scatter(fr, ms, c=c, cmap="viridis", s=10, alpha=0.6,
                    norm="log" if color_by == "count" else None)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("firing rate (fraction of tokens)")
    ax.set_ylabel("mean activation when active")
    fig.colorbar(sc, ax=ax, label=clabel, fraction=0.046, pad=0.04)
    n_dead = int((~alive).sum())
    ax.set_title(f"{title}  ({n_dead} dead of {F}, over {N:,} tokens)")
    return finish(fig, save, show=show)


def reconstruction_scatter(x, x_hat, *, max_points=5000, title="reconstruction",
                           save=None, show=False):
    """True vs reconstructed activation values."""
    xf = x.detach().cpu().flatten(); xh = x_hat.detach().cpu().flatten()
    if xf.numel() > max_points:
        idx = torch.randperm(xf.numel())[:max_points]
        xf, xh = xf[idx], xh[idx]
    fig, ax = plt.subplots()
    ax.scatter(xf.numpy(), xh.numpy(), s=3, alpha=0.3, color=plt.cm.viridis(0.5))
    lo, hi = xf.min().item(), xf.max().item()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("true"); ax.set_ylabel("reconstructed"); ax.set_title(title)
    return finish(fig, save, show=show)