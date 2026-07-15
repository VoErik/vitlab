"""SAE quality metrics: reconstruction, dictionary geometry, code structure."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Reconstruction / sparsity
def r2_score(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Fraction of variance explained, over all elements."""
    ss_res = (x - x_hat).pow(2).sum()
    ss_tot = (x - x.mean(0, keepdim=True)).pow(2).sum().clamp_min(1e-12)
    return (1 - ss_res / ss_tot).item()


def fvu(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Fraction of variance *unexplained* -- the number to actually report."""
    num = (x - x_hat).pow(2).sum()
    den = (x - x.mean(0, keepdim=True)).pow(2).sum().clamp_min(1e-12)
    return (num / den).item()


def l0(codes: torch.Tensor, eps: float = 1e-6) -> float:
    """Mean number of active latents per token."""
    return (codes.abs() > eps).float().sum(-1).mean().item()


def dead_fraction(codes: torch.Tensor, eps: float = 1e-6) -> float:
    """Fraction of latents that never fire over this batch of codes."""
    fired = (codes.abs() > eps).any(dim=0)
    return (~fired).float().mean().item()

# Dictionary geometry
def coherence(dictionary: torch.Tensor) -> dict[str, float]:
    """Off-diagonal absolute cosine similarity of decoder atoms."""
    D = F.normalize(dictionary, p=2, dim=1)
    gram = D @ D.T
    n = D.shape[0]
    off = gram[~torch.eye(n, dtype=torch.bool, device=D.device)].abs()
    return {"max": off.max().item(), "mean": off.mean().item(), "median": off.median().item()}


def stable_rank(dictionary: torch.Tensor) -> float:
    """||D||_F^2 / ||D||_2^2 -- a soft, cheap proxy for rank."""
    fro_sq = dictionary.pow(2).sum()
    op_sq = torch.linalg.matrix_norm(dictionary, ord=2).pow(2)
    return (fro_sq / op_sq.clamp_min(1e-12)).item()


def effective_rank(dictionary: torch.Tensor) -> float:
    """exp(entropy of the normalised singular-value spectrum)."""
    S = torch.linalg.svdvals(dictionary)
    p = S / S.sum().clamp_min(1e-12)
    entropy = -(p * (p + 1e-9).log()).sum()
    return entropy.exp().item()


def babel_function(dictionary: torch.Tensor, k_max: int = 64, quantile: float = 1.0) -> torch.Tensor:
    """Cumulative coherence mu_1(k): for each atom, the sum of its k largest
    absolute correlations with other atoms; reduced across atoms by `quantile`
    (1.0 = worst case = the standard definition)."""
    D = F.normalize(dictionary, p=2, dim=1)
    gram = (D @ D.T).abs()
    gram.fill_diagonal_(0.0)
    k = min(k_max, gram.shape[1] - 1)
    if k <= 0:
        return torch.zeros(k_max, device=dictionary.device)
    top, _ = torch.topk(gram, k=k, dim=1)
    cum = torch.cumsum(top, dim=1)
    curve = cum.max(0).values if quantile >= 1.0 else torch.quantile(cum, quantile, dim=0)
    if k < k_max:
        curve = torch.cat([curve, curve[-1].expand(k_max - k)])
    return curve

# Code structure
def negative_interference(codes: torch.Tensor, dictionary: torch.Tensor) -> float:
    """||ReLU( -(Z^T Z / N) * (D D^T) )||_F.

    Large when latents that co-activate have decoder atoms pointing in opposite
    directions -- i.e. features that fight each other in the reconstruction.
    """
    N = codes.shape[0]
    D = F.normalize(dictionary, p=2, dim=1)
    atom_sim = D @ D.T
    code_corr = (codes.T @ codes) / N
    return torch.relu(-(code_corr * atom_sim)).norm(p="fro").item()


@torch.no_grad()
def ood_score(dictionary: torch.Tensor, activations: torch.Tensor, chunk: int = 8192) -> float:
    """1 - mean_i max_j <D_i, A_j>.  0 = atoms lie on data, 1 = orthogonal."""
    D = F.normalize(dictionary, p=2, dim=1)
    best = torch.full((D.shape[0],), -float("inf"), device=D.device)
    for i in range(0, activations.shape[0], chunk):
        A = F.normalize(activations[i : i + chunk].to(D.device), p=2, dim=1)
        best = torch.maximum(best, (D @ A.T).max(dim=1).values)
    return (1 - best.mean()).item()


# Connectivity (co-activation graph)
@torch.no_grad()
def cooccurrence(codes: torch.Tensor, threshold: float = 1e-3, chunk: int = 8192) -> torch.Tensor:
    """(n_concepts, n_concepts) co-activation counts Z^T Z over active latents,
    accumulated in chunks so a huge N never lands in memory at once."""
    n = codes.shape[1]
    ztz = torch.zeros(n, n, device=codes.device)
    for i in range(0, codes.shape[0], chunk):
        active = (codes[i : i + chunk].abs() > threshold).float()
        ztz += active.T @ active
    return ztz


def connectivity(ztz: torch.Tensor, tau: float = 0.0) -> float:
    """1 - density of the co-activation graph (paper Eq. 13). Higher = sparser
    interactions between concepts."""
    d = ztz.shape[0]
    return 1.0 - (ztz > tau).float().sum().item() / (d * d)


def connectivity_components(ztz: torch.Tensor) -> tuple[int, list[int]]:
    """Connected components of the co-activation graph. Torch-native union-find,
    so no networkx dependency and it handles a few thousand concepts fine."""
    d = ztz.shape[0]
    adj = (ztz > 0).cpu()
    parent = list(range(d))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    idx = adj.nonzero(as_tuple=False).tolist()
    for i, j in idx:
        if i < j:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    from collections import Counter
    sizes = Counter(find(a) for a in range(d))
    ordered = sorted(sizes.values(), reverse=True)
    return len(ordered), ordered


# Stability across runs
def dictionary_stability(dict_a: torch.Tensor, dict_b: torch.Tensor, threshold: float = 0.85):
    """Match atoms of two dictionaries by cosine similarity (Hungarian) and report mean matched similarity + fraction matched above `threshold`."""
    from scipy.optimize import linear_sum_assignment

    A = F.normalize(dict_a, p=2, dim=1)
    B = F.normalize(dict_b, p=2, dim=1)
    sim = (A @ B.T).cpu().numpy()
    row, col = linear_sum_assignment(1.0 - sim)
    matched = sim[row, col]
    return float(matched.mean()), float((matched > threshold).mean())


# Bundles
@dataclass
class ReconResult:
    r2: float
    fvu: float
    l0: float
    dead_fraction: float


def reconstruction_metrics(x: torch.Tensor, x_hat: torch.Tensor, codes: torch.Tensor) -> ReconResult:
    return ReconResult(r2_score(x, x_hat), fvu(x, x_hat), l0(codes), dead_fraction(codes))


def dictionary_metrics(dictionary: torch.Tensor, k_max: int = 64) -> dict:
    coh = coherence(dictionary)
    babel = babel_function(dictionary, k_max=k_max, quantile=0.95)
    return {
        "stable_rank": stable_rank(dictionary),
        "effective_rank": effective_rank(dictionary),
        "coherence_max": coh["max"],
        "coherence_mean": coh["mean"],
        "coherence_median": coh["median"],
        "mutual_coherence": babel[0].item(),
        "babel_at_5": babel[min(4, k_max - 1)].item(),
        "babel_at_kmax": babel[k_max - 1].item(),
    }


# Intrinsic dimensionality
def _subsample_rows(X: torch.Tensor, max_rows: int = 20_000) -> torch.Tensor:
    if X.shape[0] <= max_rows:
        return X
    idx = torch.randperm(X.shape[0], device=X.device)[:max_rows]
    return X[idx]


def intrinsic_dimensionality_spectral(
    X: torch.Tensor, variance_levels=(0.9, 0.95), max_rows: int = 20_000
) -> dict:
    """Global intrinsic dimensionality from the activation spectrum.

    effective_rank = exp(entropy of the normalised singular values); dim_{p} =
    number of principal directions needed to retain p of the variance. Cheap
    sanity check on how many dimensions an SAE actually has to cover.
    """
    X = _subsample_rows(X, max_rows=max_rows)
    X = X - X.mean(dim=0, keepdim=True)
    S = torch.linalg.svdvals(X)

    p = S / S.sum().clamp_min(1e-12)
    eff_rank = (-(p * (p + 1e-9).log()).sum()).exp().item()

    var_ratio = (S ** 2) / (S ** 2).sum().clamp_min(1e-12)
    cumvar = torch.cumsum(var_ratio, dim=0)
    dims = {f"dim_{int(lvl * 100)}": int((cumvar < lvl).sum().item() + 1) for lvl in variance_levels}
    return {"effective_rank": eff_rank, **dims}


def _estimate_local_id(X: torch.Tensor, k: int = 20, eps: float = 1e-8) -> torch.Tensor:
    """Levina-Bickel MLE of local intrinsic dimension, per point.

    cdist is O(N^2); keep the sample (sample_size in local_id_mle) modest.
    """
    N = X.shape[0]
    k = min(k, N - 1)
    dists = torch.cdist(X, X, p=2)
    dists, _ = torch.sort(dists, dim=1)
    knn = dists[:, 1 : k + 1]
    d_k = knn[:, -1].unsqueeze(1)
    ratios = torch.log((d_k + eps) / (knn + eps))
    return (k - 1) / ratios.sum(dim=1).clamp_min(eps)


def local_id_mle(
    X: torch.Tensor,
    sample_size: int = 20_000,
    k: int = 20,
    quantile_values=(0.5, 0.75, 0.9, 0.95, 0.99),
) -> dict:
    """Distribution of local intrinsic dimension over a sample of points."""
    idx = torch.randperm(X.shape[0])[:sample_size]
    X_sub = X[idx]
    lid = _estimate_local_id(X_sub, k=k)
    qs = torch.tensor(list(quantile_values), device=lid.device, dtype=lid.dtype)
    quantiles = torch.quantile(lid, qs)
    return {
        "local_id_mean": lid.mean().item(),
        "local_id_median": lid.median().item(),
        "local_id_std": lid.std().item(),
        "local_id_min": lid.min().item(),
        "local_id_max": lid.max().item(),
        "quantiles": {f"q{int(q * 100)}": float(v) for q, v in zip(quantile_values, quantiles)},
    }
