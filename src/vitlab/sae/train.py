from __future__ import annotations

import copy
import functools
import time
from collections import defaultdict
from functools import partial
from typing import Optional

import torch

from .utils import extract_input


class SparsityPID:
    """Nudges the L1 penalty up or down to drive mean L0 toward a target."""

    def __init__(self, target_l0, kp=0.1, ki=0.01, kd=0.05):
        self.target, self.kp, self.ki, self.kd = target_l0, kp, ki, kd
        self.integral = 0.0
        self.last_error = 0.0

    def update(self, current_l0, current_penalty):
        error = (current_l0 - self.target) / self.target
        # reset the integrator when we cross the target (anti-windup)
        if (self.last_error > 0) != (error > 0):
            self.integral = 0.0
        else:
            self.integral += error
        self.integral = max(min(self.integral, 50.0), -50.0)
        derivative = error - self.last_error
        self.last_error = error
        adjustment = self.kp * error + self.ki * self.integral + self.kd * derivative
        step = max(min(adjustment, 0.2), -0.2)
        return max(current_penalty * (1.0 + step), 1e-5)


def _r2(x, x_hat):
    ss_res = (x - x_hat).pow(2).sum()
    ss_tot = (x - x.mean(0, keepdim=True)).pow(2).sum().clamp_min(1e-12)
    return (1 - ss_res / ss_tot).item()


def _l0(z, eps=0.0):
    return (z.abs() > eps).float().sum(-1).mean().item()


def train_sae(
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler=None,
    *,
    nb_epochs: int = 20,
    clip_grad: float = 1.0,
    device: str = "cuda",
    target_l0: Optional[int] = None,
    l1_warmup_epochs: int = 5,
    log_every: int = 1,
    verbose: bool = True,
):
    """Returns (logs, best_model). best_model is the deep copy at the highest R2."""
    logs = defaultdict(list)
    best_r2 = -float("inf")
    best_model = model

    pid = None
    current_penalty = 0.0
    if target_l0 is not None:
        pid = SparsityPID(target_l0=target_l0)
        if not isinstance(criterion, functools.partial):
            criterion = functools.partial(criterion, penalty=1e-3)
        current_penalty = criterion.keywords.get("penalty", 1.0)

    for epoch in range(nb_epochs):
        model.train()
        t0 = time.time()
        ep_loss = ep_r2 = ep_l0 = 0.0
        n_batches = 0
        n_concepts = None
        fired = None

        for batch in dataloader:
            x = extract_input(batch).to(device, non_blocking=True).float()
            optimizer.zero_grad()
            pre_codes, codes, x_hat = model(x)
            loss = criterion(x, x_hat, pre_codes, codes, model.get_dictionary())
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if n_concepts is None:
                n_concepts = codes.shape[1]
                fired = torch.zeros(n_concepts, device=device)
            fired += (codes.abs() > 0).any(dim=0).float()

            ep_loss += loss.item()
            ep_r2 += _r2(x, x_hat)
            ep_l0 += _l0(codes)
            n_batches += 1

        if n_batches == 0:
            continue
        avg_loss = ep_loss / n_batches
        avg_r2 = ep_r2 / n_batches
        avg_l0 = ep_l0 / n_batches
        dead = (fired == 0).float().mean().item()

        if epoch >= l1_warmup_epochs and pid is not None:
            current_penalty = pid.update(avg_l0, current_penalty)
            criterion = partial(criterion.func, **{**criterion.keywords, "penalty": current_penalty})

        logs["avg_loss"].append(avg_loss)
        logs["r2"].append(avg_r2)
        logs["z_sparsity"].append(avg_l0)
        logs["dead_features"].append(dead)
        logs["current_penalty"].append(current_penalty)
        logs["time_epoch"].append(time.time() - t0)

        if avg_r2 > best_r2:
            best_r2 = avg_r2
            best_model = copy.deepcopy(model)

        if verbose and (epoch % log_every == 0):
            print(f"epoch {epoch+1}/{nb_epochs}  loss {avg_loss:.4f}  R2 {avg_r2:.4f}  "
                  f"L0 {avg_l0:.1f}  dead {dead*100:.1f}%"
                  + (f"  penalty {current_penalty:.4g}" if pid else ""))

    return logs, best_model
