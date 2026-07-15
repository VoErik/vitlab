from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def extract_input(batch):
    """Pull the activation tensor out of whatever the loader yields."""
    if isinstance(batch, (tuple, list)):
        return batch[0]
    if isinstance(batch, dict):
        return batch.get("data", batch.get("pixel_values"))
    return batch


def aggregate_latents(z, aggregation: str = "max"):
    """(B, T, F) token-level codes -> (B, F') per image. 'both' concatenates."""
    if z.dim() != 3:
        return z
    if aggregation == "max":
        return z.max(dim=1).values
    if aggregation == "mean":
        return z.mean(dim=1)
    if aggregation == "both":
        return torch.cat([z.max(dim=1).values, z.mean(dim=1)], dim=-1)
    raise ValueError("aggregation must be 'max', 'mean', or 'both'")


@torch.no_grad()
def pytorch_kmeans(embeddings, n_clusters, *, niter=100, batch_size=4096, tol=1e-4,
                   device="cuda", seed=42, verbose=True):
    """VRAM-safe minibatch k-means. `embeddings` stays on CPU; only minibatches
    move to the GPU. Used to seed archetypal dictionaries."""
    if embeddings.is_cuda:
        embeddings = embeddings.cpu()
        torch.cuda.empty_cache()
    N = embeddings.shape[0]
    g = torch.Generator().manual_seed(seed)
    centroids = embeddings[torch.randperm(N, generator=g)[:n_clusters]].to(device)

    for i in range(niter):
        old = centroids.clone()
        sums = torch.zeros_like(centroids)
        counts = torch.zeros(n_clusters, device=device)
        c_norm = (centroids ** 2).sum(1).view(1, -1)
        for s in range(0, N, batch_size):
            xb = embeddings[s : s + batch_size].to(device, non_blocking=True)
            dist = (xb ** 2).sum(1, keepdim=True) + c_norm - 2 * xb @ centroids.t()
            lbl = dist.argmin(1)
            sums.index_add_(0, lbl, xb)
            counts.index_add_(0, lbl, torch.ones(xb.shape[0], device=device))
        centroids = sums / counts.clamp(min=1).view(-1, 1)
        shift = (centroids - old).pow(2).sum()
        if verbose and i % 5 == 0:
            print(f"  kmeans iter {i}: shift {shift:.6f}")
        if shift < tol:
            break
    return centroids


class MemmapDataset(Dataset):
    def __init__(self, path, shape, dtype=np.float32):
        self.data = np.memmap(path, dtype=dtype, mode="c", shape=shape)
        self.shape = shape

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx].copy())
