from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from .spatial import patch_values_to_grid
from .style import finish


@torch.no_grad()
def top_activating_patches(layer_sae, reader, loader, site, latent: int, *,
                           top_k: int = 12, device="cuda", id_key=None):
    """Scan a dataset for the patches that activate `latent` most.

    Returns a list of dicts: {score, image_index, patch_index, patch_grid_value}.
    Keeps only a running top-k, so it streams over an arbitrarily large dataset.
    """
    import heapq

    reader.backbone.to(device).eval()
    layer_sae.to(device)
    prefix = reader.spec.n_prefix_tokens
    heap: list = []
    img_counter = 0

    for batch in loader:
        px = (batch["pixel_values"] if isinstance(batch, dict) else batch[0]).to(device)
        acts = reader.read(px, site)
        patches = acts[:, prefix:, :]
        B, P, Dm = patches.shape
        codes = layer_sae.encode(patches.reshape(B * P, Dm))[:, latent].reshape(B, P)
        best_per_img, best_patch = codes.max(dim=1)
        for b in range(B):
            entry = (best_per_img[b].item(), img_counter + b, int(best_patch[b].item()))
            if len(heap) < top_k:
                heapq.heappush(heap, entry)
            elif entry[0] > heap[0][0]:
                heapq.heapreplace(heap, entry)
        img_counter += B

    return sorted(heap, key=lambda e: -e[0])


@torch.no_grad()
def concept_evidence_grid(layer_sae, reader, dataset, site, latent, *,
                          loader=None, top_k=9, device="cuda", denorm_key=None,
                          model_key=None, title=None, save=None, show=False):
    """
    Grid of the top-activating images for a latent."""
    from ..batching import make_loader
    from ..datasets import denormalize

    if loader is None:
        def _px_collate(b):
            items = [x["pixel_values"] if isinstance(x, dict) else x[0] for x in b]
            return {"pixel_values": torch.stack(items)}
        loader = make_loader(dataset, 32, shuffle=False, num_workers=0, collate_fn=_px_collate)
    top = top_activating_patches(layer_sae, reader, loader, site, latent,
                                 top_k=top_k, device=device)

    prefix = reader.spec.n_prefix_tokens
    ncol = int(np.ceil(np.sqrt(top_k)))
    nrow = int(np.ceil(top_k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 2.2 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")

    for i, (score, img_idx, patch_idx) in enumerate(top):
        item = dataset[img_idx]
        px = item["pixel_values"] if isinstance(item, dict) else item[0]
        acts = reader.read(px.unsqueeze(0).to(device), site)[:, prefix:, :]
        B, P, Dm = acts.shape
        code_map = layer_sae.encode(acts.reshape(P, Dm))[:, latent]
        heat = patch_values_to_grid(code_map, grid=None)

        img = px
        if model_key is not None:
            img = denormalize(px, model_key).permute(1, 2, 0).numpy()
        else:
            img = px.permute(1, 2, 0).cpu().numpy()
            img = (img - img.min()) / (np.ptp(img) + 1e-8)

        ax = axes[i]
        ax.imshow(img)
        ax.imshow(heat, cmap="viridis", alpha=0.5,
                  extent=(0, img.shape[1], img.shape[0], 0),
                  interpolation="nearest", aspect="auto")
        ax.set_title(f"{score:.2f}", fontsize=8)
        ax.axis("off")

    fig.suptitle(title or f"latent {latent}: top activations @ {site}")
    return finish(fig, save, show=show)
