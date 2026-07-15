from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from .batching import MultiTaskLoader, TaskBatch
from .model import ModelConfig, MultiTaskViT, save_model


@dataclass
class OptimConfig:
    backbone_lr: float = 1e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    prox_l1: bool = True

@dataclass
class TrainConfig:
    model: ModelConfig
    optim: OptimConfig = field(default_factory=OptimConfig)
    epochs: int = 10
    grad_accum: int = 1
    mixed_precision: str = "bf16" # "no" | "fp16" | "bf16"
    output_dir: str = "runs/default"
    seed: int = 0
    eval_every: int = 1
    log_every: int = 50


def head_lr_now(optimizer) -> float:
    for g in optimizer.param_groups:
        if g.get("name", "").startswith("head_weights"):
            return g["lr"]
    return optimizer.param_groups[-1]["lr"]


def _cosine_with_warmup(optimizer, total_steps: int, warmup: int):
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model: MultiTaskViT, loaders: dict[str, DataLoader], accelerator) -> dict[str, float]:
    model.eval()
    correct: dict[str, int] = defaultdict(int)
    seen: dict[str, int] = defaultdict(int)
    losses: dict[str, float] = defaultdict(float)

    for task, loader in loaders.items():
        for batch in loader:
            px = batch["pixel_values"].to(accelerator.device)
            y = batch["labels"].to(accelerator.device)
            logits, loss = model(px, task, y)
            preds = logits.argmax(-1)
            correct[task] += (preds == y).sum().item()
            seen[task] += y.numel()
            losses[task] += loss.item() * y.shape[0]

    metrics = {}
    for task in loaders:
        n = max(seen[task], 1)
        metrics[f"{task}/acc"] = correct[task] / n
        metrics[f"{task}/loss"] = losses[task] / n
    for task in loaders:
        if model.heads[task].spec.alpha > 0:  # only meaningful for sparse probes
            metrics[f"{task}/head_sparsity"] = model.heads[task].sparsity()
    metrics["mean_acc"] = sum(
        v for k, v in metrics.items() if k.endswith("/acc")
    ) / max(len(loaders), 1)
    model.train()
    return metrics


def train(
    cfg: TrainConfig,
    train_loader: MultiTaskLoader,
    val_loaders: dict[str, DataLoader] | None = None,
) -> MultiTaskViT:
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.grad_accum,
    )
    torch.manual_seed(cfg.seed)

    model = MultiTaskViT.from_config(cfg.model)
    optimizer = torch.optim.AdamW(
        model.param_groups(cfg.optim.backbone_lr, cfg.optim.head_lr, cfg.optim.weight_decay),
        betas=cfg.optim.betas,
    )
    total_steps = (len(train_loader) * cfg.epochs) // cfg.grad_accum
    scheduler = _cosine_with_warmup(
        optimizer, total_steps, int(total_steps * cfg.optim.warmup_ratio)
    )

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    if val_loaders:
        val_loaders = {t: accelerator.prepare(dl) for t, dl in val_loaders.items()}

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))

    unwrapped = lambda: accelerator.unwrap_model(model) # noqa: E731
    best = -1.0
    step = 0

    for epoch in range(cfg.epochs):
        model.train()
        running: dict[str, list[float]] = defaultdict(list)

        for batch in train_loader:
            batch: TaskBatch = batch.to(accelerator.device)
            head = unwrapped().heads[batch.task]
            with accelerator.accumulate(model):
                _, loss = model(batch.pixel_values, batch.task, batch.labels)
                total = loss * head.spec.loss_weight + head.penalty(
                    include_l1=not cfg.optim.prox_l1
                )
                accelerator.backward(total)
                if accelerator.sync_gradients and cfg.optim.grad_clip:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
                optimizer.step()
                if accelerator.sync_gradients and cfg.optim.prox_l1:
                    head.prox_l1_(head_lr_now(optimizer), optimizer=optimizer)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running[batch.task].append(loss.item())
            step += 1
            if step % cfg.log_every == 0 and accelerator.is_main_process:
                msg = " ".join(
                    f"{t}:{sum(v[-cfg.log_every:]) / len(v[-cfg.log_every:]):.3f}"
                    for t, v in running.items()
                    if v
                )
                print(f"epoch {epoch} step {step} lr {scheduler.get_last_lr()[0]:.2e} | {msg}")

        if val_loaders and (epoch + 1) % cfg.eval_every == 0:
            metrics = evaluate(unwrapped(), val_loaders, accelerator)
            if accelerator.is_main_process:
                print(f"[eval] epoch {epoch}: {metrics}")
                if metrics["mean_acc"] > best:
                    best = metrics["mean_acc"]
                    save_model(unwrapped(), out_dir / "best")

    if accelerator.is_main_process:
        save_model(unwrapped(), out_dir / "last")
    return unwrapped()
