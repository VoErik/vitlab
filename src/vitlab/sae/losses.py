"""
SAE training losses.
Signature convention (forced by overcomplete): every criterion is f(x, x_hat, pre_codes, codes, dictionary, **kwargs) -> scalar
"""

from __future__ import annotations

import torch


def mse(x, x_hat, pre_codes=None, codes=None, dictionary=None):
    return (x - x_hat).square().mean()


def reanimation(x, x_hat, pre_codes, codes, dictionary, coeff: float = 1e-3):
    """MSE minus a small pull on pre-activations of batch-dead latents."""
    loss = (x - x_hat).square().mean()
    is_dead = ((codes > 0).sum(dim=0) == 0).float().detach()
    loss = loss - (pre_codes * is_dead[None, :]).mean() * coeff
    return loss


def mse_with_l1(x, x_hat, pre_codes, codes, dictionary, penalty: float = 3.0,
                reanim_dead_codes: bool = True):
    """Classic SAE loss: MSE + L1, with an optional ghost-gradient term that lets
    dead latents reconstruct the residual so they can revive."""
    loss = (x - x_hat).pow(2).mean() + penalty * codes.abs().sum(-1).mean()
    if reanim_dead_codes:
        dead = (codes > 0).sum(dim=0) == 0
        if dead.any():
            residual = (x - x_hat).detach()
            ghost = torch.exp(pre_codes[:, dead]) @ dictionary[dead, :]
            loss = loss + 0.1 * (residual - ghost).pow(2).mean()
    return loss


def gated_sae_loss(x, x_hat, pi_gate, codes, dictionary, penalty: float = 3.0):
    """Loss for GatedSAE: reconstruction + a gate auxiliary term + L1 on the gate,
    plus a ghost term for gate-dead latents."""
    mse_loss = (x - x_hat).pow(2).mean()
    gate = torch.relu(pi_gate)
    aux = (x - gate @ dictionary).pow(2).mean()
    l1 = gate.sum(-1).mean()
    ghost = x.new_tensor(0.0)
    dead = (gate > 0).sum(dim=0) == 0
    if dead.any():
        residual = (x - x_hat).detach()
        ghost_recon = torch.exp(pi_gate[:, dead]) @ dictionary[dead, :]
        ghost = (residual - ghost_recon).pow(2).mean()
    return mse_loss + aux + penalty * l1 + 0.1 * ghost


class MatryoshkaLossWrapper:
    """Wrap any criterion to add nested (Matryoshka) reconstruction losses: each
    prefix of the dictionary must reconstruct on its own, encouraging the most
    important concepts into the earliest latents."""

    def __init__(self, base_criterion, nested_dims=(64, 128, 512, 1024), weight: float = 1.0):
        self.base_criterion = base_criterion
        self.nested_dims = sorted(nested_dims)
        self.weight = weight

    def __call__(self, x, x_hat, pre_codes, codes, dictionary, **kwargs):
        total = self.base_criterion(x, x_hat, pre_codes, codes, dictionary, **kwargs)
        x_hat_full = x_hat.detach()
        for k in self.nested_dims:
            if k >= codes.shape[1]:
                continue
            tail = codes[:, k:].detach() @ dictionary[k:, :].detach()
            total = total + self.weight * (x - (x_hat_full - tail)).pow(2).mean()
        return total
