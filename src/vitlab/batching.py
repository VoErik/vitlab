"""Batching strategy for multi-task fine-tuning..

Strategies:

    proportional   P(task) ~ dataset size. Matches single-task training in
                   expectation; small tasks get starved.
    uniform        every task equally often. Small tasks over-sampled.
    temperature    P(task) ~ size**(1/T). T=1 is proportional, T -> inf is uniform.
                   Default: T ~ 2.
    round_robin    deterministic cycle.

This balances *tasks* against each other. To balance *classes within* a task, use
`vitlab.datasets.class_balanced_sampler`. The two are independent.

Datasets *must* yield dicts with "pixel_values" and "labels".
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, Literal

import torch
from torch.utils.data import DataLoader, Dataset

Strategy = Literal["proportional", "uniform", "temperature", "round_robin"]


@dataclass
class TaskBatch:
    task: str
    pixel_values: torch.Tensor
    labels: torch.Tensor

    def to(self, device) -> "TaskBatch":
        return TaskBatch(
            self.task,
            self.pixel_values.to(device, non_blocking=True),
            self.labels.to(device, non_blocking=True),
        )


def collate(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack(
            [
                b["labels"] if torch.is_tensor(b["labels"]) else torch.tensor(b["labels"])
                for b in batch
            ]
        ),
    }


def make_loader(
    dataset: Dataset,
    batch_size: int,
    *,
    shuffle: bool = True,
    sampler=None,
    **kw,
) -> DataLoader:
    if sampler is not None:
        shuffle = False  # mutually exclusive; torch raises otherwise
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=kw.pop("collate_fn", collate),
        num_workers=kw.pop("num_workers", 4),
        pin_memory=kw.pop("pin_memory", torch.cuda.is_available()),
        drop_last=kw.pop("drop_last", shuffle or sampler is not None),
        **kw,
    )


class MultiTaskLoader:
    """Yields (task, batch) pairs. Exhausted task loaders are restarted."""

    def __init__(
        self,
        loaders: dict[str, DataLoader],
        strategy: Strategy = "temperature",
        *,
        temperature: float = 2.0,
        steps_per_epoch: int | None = None,
        seed: int = 0,
    ):
        if not loaders:
            raise ValueError("need at least one task loader")
        self.loaders = loaders
        self.tasks = list(loaders)
        self.strategy = strategy
        self.rng = random.Random(seed)
        self.steps_per_epoch = steps_per_epoch or sum(len(dl) for dl in loaders.values())

        sizes = [len(dl) for dl in loaders.values()]
        if strategy == "uniform":
            weights = [1.0] * len(sizes)
        elif strategy == "proportional":
            weights = [float(s) for s in sizes]
        elif strategy == "temperature":
            weights = [float(s) ** (1.0 / temperature) for s in sizes]
        else:  # round_robin
            weights = []
        total = sum(weights) or 1.0
        self.weights = [w / total for w in weights]

    def __len__(self) -> int:
        return self.steps_per_epoch

    @property
    def task_probs(self) -> dict[str, float]:
        if self.strategy == "round_robin":
            return {t: 1 / len(self.tasks) for t in self.tasks}
        return dict(zip(self.tasks, self.weights))

    def __iter__(self) -> Iterator[TaskBatch]:
        iters = {t: iter(dl) for t, dl in self.loaders.items()}
        for step in range(self.steps_per_epoch):
            if self.strategy == "round_robin":
                task = self.tasks[step % len(self.tasks)]
            else:
                task = self.rng.choices(self.tasks, weights=self.weights, k=1)[0]
            try:
                batch = next(iters[task])
            except StopIteration:
                iters[task] = iter(self.loaders[task])
                batch = next(iters[task])
            yield TaskBatch(task, batch["pixel_values"], batch["labels"])
