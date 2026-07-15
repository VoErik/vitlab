"""
SAE concept-bottleneck model (SAE-CBM).

Two modes:
  separate  train the SAE first, freeze it, pool latents, train the head.
  joint     train SAE and head together; the classification loss shapes the SAE.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import aggregate_latents


class SparseLinearHead(nn.Module):
    """LayerNorm -> Linear with an elastic-net penalty on the weight matrix.

    Same math as vitlab.heads.TaskHead, on a flat latent vector.
    """

    def __init__(self, in_features: int, num_classes: int, *, alpha: float = 1e-3,
                 l1_ratio: float = 0.5, standardize: bool = True):
        super().__init__()
        self.alpha, self.l1_ratio = alpha, l1_ratio
        self.pre = nn.LayerNorm(in_features, elementwise_affine=not standardize)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pre(z))

    def penalty(self, *, include_l1: bool = False) -> torch.Tensor:
        w = self.classifier.weight
        if self.alpha == 0.0:
            return w.new_zeros(())
        pen = 0.5 * self.alpha * (1 - self.l1_ratio) * w.pow(2).sum()
        if include_l1:
            pen = pen + self.alpha * self.l1_ratio * w.abs().sum()
        return pen

    @torch.no_grad()
    def prox_l1_(self, lr: float, optimizer: torch.optim.Optimizer | None = None) -> None:
        a, r = self.alpha, self.l1_ratio
        if a == 0.0 or r == 0.0:
            return
        w = self.classifier.weight
        t: torch.Tensor | float = lr * a * r
        if optimizer is not None:
            st = optimizer.state.get(w, {})
            v, step = st.get("exp_avg_sq"), st.get("step")
            if v is not None and step is not None:
                beta2 = optimizer.param_groups[0].get("betas", (0.9, 0.999))[1]
                eps = optimizer.param_groups[0].get("eps", 1e-8)
                k = float(step)
                v_hat = v / (1 - beta2 ** k) if k > 0 else v
                t = lr * a * r / (v_hat.sqrt() + eps)
        w.copy_(w.sign() * (w.abs() - t).clamp_min(0.0))

    @torch.no_grad()
    def sparsity(self) -> float:
        return (self.classifier.weight == 0).float().mean().item()


@dataclass
class CBMConfig:
    aggregation: str = "max"          # max | mean | both
    alpha: float = 1e-2               # elastic-net strength
    l1_ratio: float = 0.5
    clf_lr: float = 1e-3
    clf_epochs: int = 15
    prox_l1: bool = True


def _class_weights(labels: torch.Tensor, num_classes: int, device) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float().clamp_min(1)
    w = 1.0 / counts
    return (w / w.sum() * num_classes).to(device)


@torch.no_grad()
def pool_latents(sae, embeddings, normalizer, *, aggregation="max", batch_size=1024, device="cuda"):
    """(N, T, D) raw embeddings -> (N, F') pooled latents. Normalises, encodes per
    token, pools over tokens. Kept on CPU output so a big N does not fill VRAM."""
    sae.to(device).eval()
    N, T, D = embeddings.shape
    out = None
    for i in range(0, N, batch_size):
        xb = embeddings[i : i + batch_size].to(device)
        B = xb.shape[0]
        xn = normalizer.to(device).norm(xb.reshape(B * T, D))
        z = sae.encode(xn)
        z = z[1] if isinstance(z, tuple) else z
        pooled = aggregate_latents(z.reshape(B, T, -1), aggregation).cpu()
        if out is None:
            out = torch.empty(N, pooled.shape[-1])
        out[i : i + B] = pooled
    return out


def train_head(head, X, y, cfg: CBMConfig, *, class_weights=None, Xte=None, yte=None, device="cuda"):
    """Train the sparse head on pooled latents (separate mode)."""
    head.to(device)
    opt = torch.optim.Adam(head.parameters(), lr=cfg.clf_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.clf_epochs)
    Xd, yd = X.to(device), y.to(device)

    test_acc = None
    for epoch in range(cfg.clf_epochs):
        head.train()
        opt.zero_grad()
        logits = head(Xd)
        loss = F.cross_entropy(logits, yd, weight=class_weights)
        loss = loss + head.penalty(include_l1=not cfg.prox_l1)
        loss.backward()
        opt.step()
        if cfg.prox_l1:
            head.prox_l1_(sched.get_last_lr()[0], optimizer=opt)
        sched.step()

        if Xte is not None:
            head.eval()
            with torch.no_grad():
                test_acc = (head(Xte.to(device)).argmax(1).cpu() == yte).float().mean().item()
    return head, test_acc


def train_cbm_separate(sae, embeddings, labels, normalizer, cfg: CBMConfig, *,
                       test_embeddings=None, test_labels=None, device="cuda"):
    """SAE already trained -> pool -> train sparse head. Returns (head, info)."""
    num_classes = int(labels.max().item()) + 1
    cw = _class_weights(labels, num_classes, device)

    Xtr = pool_latents(sae, embeddings, normalizer, aggregation=cfg.aggregation, device=device)
    Xte = (pool_latents(sae, test_embeddings, normalizer, aggregation=cfg.aggregation, device=device)
           if test_embeddings is not None else None)

    head = SparseLinearHead(Xtr.shape[-1], num_classes, alpha=cfg.alpha, l1_ratio=cfg.l1_ratio)
    head, test_acc = train_head(head, Xtr, labels, cfg, class_weights=cw,
                                Xte=Xte, yte=test_labels, device=device)
    return head, {"test_acc": test_acc, "head_sparsity": head.sparsity(), "latent_dim": Xtr.shape[-1]}


def train_cbm_joint(sae, embeddings, labels, normalizer, criterion, sae_optimizer, cfg: CBMConfig, *,
                    test_embeddings=None, test_labels=None, batch_size=256,
                    lambda_sae=1.0, lambda_clf=1.0, epochs=20, device="cuda"):
    """Train SAE and head together: classification gradients flow into the SAE, so
    the bottleneck is shaped to be classifiable, not only reconstructive."""
    from torch.utils.data import DataLoader, TensorDataset

    num_classes = int(labels.max().item()) + 1
    cw = _class_weights(labels, num_classes, device)
    N, T, D = embeddings.shape

    # probe latent dim once
    sae.to(device)
    with torch.no_grad():
        z0 = sae.encode(normalizer.to(device).norm(embeddings[:1].reshape(T, D).to(device)))
        z0 = z0[1] if isinstance(z0, tuple) else z0
    F_ = z0.shape[-1] * (2 if cfg.aggregation == "both" else 1)
    head = SparseLinearHead(F_, num_classes, alpha=cfg.alpha, l1_ratio=cfg.l1_ratio).to(device)

    clf_opt = torch.optim.Adam(head.parameters(), lr=cfg.clf_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(clf_opt, T_max=epochs)

    loader = DataLoader(TensorDataset(embeddings, labels), batch_size=batch_size, shuffle=True)
    test_acc = None
    for epoch in range(epochs):
        sae.train(); head.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            B = xb.shape[0]
            xf = normalizer.norm(xb.reshape(B * T, D))
            sae_optimizer.zero_grad(); clf_opt.zero_grad()
            pre, z, xhat = sae(xf)
            sae_loss = criterion(xf, xhat, pre, z, sae.get_dictionary())
            h = aggregate_latents(z.reshape(B, T, -1), cfg.aggregation)
            clf_loss = F.cross_entropy(head(h), yb, weight=cw) + head.penalty(include_l1=not cfg.prox_l1)
            (lambda_sae * sae_loss + lambda_clf * clf_loss).backward()
            sae_optimizer.step(); clf_opt.step()
            if cfg.prox_l1:
                head.prox_l1_(sched.get_last_lr()[0], optimizer=clf_opt)
        sched.step()

        if test_embeddings is not None:
            sae.eval(); head.eval()
            Xte = pool_latents(sae, test_embeddings, normalizer, aggregation=cfg.aggregation, device=device)
            with torch.no_grad():
                test_acc = (head(Xte.to(device)).argmax(1).cpu() == test_labels).float().mean().item()

    return sae, head, {"test_acc": test_acc, "head_sparsity": head.sparsity()}
