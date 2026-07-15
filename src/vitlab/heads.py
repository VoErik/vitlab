from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn

from .backbone import BackboneOutput

Pooling = Literal["cls", "mean", "cls_mean", "attn"]
# cls_mean concatenates the cls and mean of patch embeddings
# used by dino family, but couldnt really see any difference here so far, so default to cls for all

@dataclass
class TaskSpec:
    name: str
    num_classes: int
    pooling: Pooling = "cls"
    head: Literal["linear", "mlp"] = "linear"
    hidden_dim: int | None = None
    dropout: float = 0.0

    norm: Literal["layernorm", "standardize", "none"] = "layernorm"

    alpha: float = 0.0 # elastic-net strength; 0.0 = off
    l1_ratio: float = 0.5 # 1.0 = lasso, 0.0 = ridge

    loss_weight: float = 1.0 # TODO: possible tuning target, but out of scope
    class_weights: list[float] | None = None
    multilabel: bool = False
    metadata: dict = field(default_factory=dict)

    def in_features(self, d_model: int) -> int:
        return 2 * d_model if self.pooling == "cls_mean" else d_model


class AttentionPool(nn.Module):
    """Single learned query attending over patch tokens."""

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) * d_model**-0.5)
        self.scale = d_model**-0.5

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        attn = torch.einsum("bpd,d->bp", patches, self.query) * self.scale
        return torch.einsum("bp,bpd->bd", attn.softmax(dim=-1), patches)


class TaskHead(nn.Module):
    """Pooling -> (norm) -> (dropout) -> [hidden] -> Linear.

    `self.classifier` is always the final Linear, and it is the only thing the
    elastic-net penalty touches. For `head="linear"` that means the penalty acts
    on the probe itself; for `head="mlp"` it acts on the readout layer only.
    """

    def __init__(self, spec: TaskSpec, d_model: int):
        super().__init__()
        self.spec = spec
        self.pool = AttentionPool(d_model) if spec.pooling == "attn" else None

        d_in = spec.in_features(d_model)
        pre: list[nn.Module] = []
        if spec.norm == "layernorm":
            pre.append(nn.LayerNorm(d_in))
        elif spec.norm == "standardize":
            pre.append(nn.LayerNorm(d_in, elementwise_affine=False))
        if spec.dropout:
            pre.append(nn.Dropout(spec.dropout))
        if spec.head == "mlp":
            hidden = spec.hidden_dim or d_in
            pre += [nn.Linear(d_in, hidden), nn.GELU()]
            d_in = hidden
        self.pre = nn.Sequential(*pre)
        self.classifier = nn.Linear(d_in, spec.num_classes)

        weights = (
            torch.tensor(spec.class_weights, dtype=torch.float32)
            if spec.class_weights is not None
            else None
        )
        self.register_buffer("class_weight", weights, persistent=True)

    def pooled(self, feats: BackboneOutput) -> torch.Tensor:
        if self.pool is not None:
            return self.pool(feats.patches)
        return feats.pooled(self.spec.pooling)

    def forward(self, feats: BackboneOutput) -> torch.Tensor:
        return self.classifier(self.pre(self.pooled(feats)))

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.spec.multilabel:
            return nn.functional.binary_cross_entropy_with_logits(
                logits, labels.float(), pos_weight=self.class_weight
            )
        return nn.functional.cross_entropy(logits, labels, weight=self.class_weight)

    def penalty(self, *, include_l1: bool = False) -> torch.Tensor:
        """alpha * (l1_ratio * ||W||_1 + 0.5 * (1 - l1_ratio) * ||W||_2^2)."""
        w = self.classifier.weight
        if self.spec.alpha == 0.0:
            return w.new_zeros(())
        a, r = self.spec.alpha, self.spec.l1_ratio
        pen = 0.5 * a * (1.0 - r) * w.pow(2).sum()
        if include_l1:
            pen = pen + a * r * w.abs().sum()
        return pen

    @torch.no_grad()
    def prox_l1_(self, lr: float, optimizer: torch.optim.Optimizer | None = None) -> None:
        """Soft-threshold the classifier weights in place: the proximal operator of
        the L1 term. This is what produces exact zeros; a subgradient step never does.

        `optimizer` matters more than it looks. The textbook threshold `lr*alpha*r`
        is the operator for plain SGD, whose step is proportional to the gradient.
        Adam's step is *not*: it is roughly `lr * sign(g)` whatever the gradient's
        magnitude. So an irrelevant weight still gets shoved by ~lr every step while
        the threshold claws back only `lr*alpha*r` of it, and with a small alpha
        nothing ever reaches zero -- the penalty looks silently inert.

        Passing the optimizer lets us threshold in Adam's own geometry, using its
        second-moment estimate:

            t_i = lr * alpha * r / (sqrt(v_i) + eps)

        which restores the usual lasso stationarity condition: weight i sits at zero
        exactly when |grad_i| <= alpha * r. That is what makes `alpha` mean the same
        thing it means in sklearn, rather than "some number that depends on the
        optimiser you happened to pick".
        """
        a, r = self.spec.alpha, self.spec.l1_ratio
        if a == 0.0 or r == 0.0:
            return
        w = self.classifier.weight

        t: torch.Tensor | float = lr * a * r
        if optimizer is not None:
            state = optimizer.state.get(w, {})
            v, step = state.get("exp_avg_sq"), state.get("step")
            if v is not None and step is not None:
                beta2 = optimizer.param_groups[0].get("betas", (0.9, 0.999))[1]
                eps = optimizer.param_groups[0].get("eps", 1e-8)
                k = float(step)
                v_hat = v / (1 - beta2**k) if k > 0 else v
                t = lr * a * r / (v_hat.sqrt() + eps)

        w.copy_(w.sign() * (w.abs() - t).clamp_min(0.0))

    @torch.no_grad()
    def sparsity(self) -> float:
        """Fraction of exactly-zero classifier weights. The number to report."""
        w = self.classifier.weight
        return (w == 0).float().mean().item()

    @torch.no_grad()
    def active_features(self, class_idx: int | None = None) -> torch.Tensor:
        """Indices of the input dimensions the probe actually uses."""
        w = self.classifier.weight
        row = w.abs().sum(0) if class_idx is None else w[class_idx].abs()
        return (row > 0).nonzero(as_tuple=True)[0]
