from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import WeightedRandomSampler

from .config import get_data_root

Augment = Literal["none", "standard", "dermoscopy"]


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset. Paths are relative to DATA_ROOT."""

    name: str
    images: str
    num_classes: int
    label_key: str = "label"
    multilabel: bool = False
    metadata: str | None = None
    concepts: str | None = None
    augment: Augment = "standard"
    notes: str = ""

    def path(self, key: str, data_root: str | Path | None = None) -> Path:
        rel = getattr(self, key)
        if rel is None:
            raise ValueError(f"{self.name} has no {key!r}")
        return get_data_root(data_root) / rel


REGISTRY: dict[str, DatasetSpec] = {
    s.name: s
    for s in [
        DatasetSpec(
            name="7ptderm",
            images="7ptderm/imgs",
            metadata="7ptderm/metadata2.csv",
            concepts="7ptderm/concepts.csv",
            num_classes=7,
            augment="dermoscopy",
        ),
        DatasetSpec(
            name="dermamnist",
            images="derma_mnist/imgs",
            metadata="derma_mnist/DermaMNIST-E.csv",
            num_classes=7,
            augment="dermoscopy",
        ),
        DatasetSpec(
            name="fitzpatrick17k",
            images="fitzpatrick/imgs",
            metadata="fitzpatrick/Fitzpatrick17k-C.csv",
            concepts="fitzpatrick/skincon-fitzpatrick-c.csv",
            num_classes=9,
            augment="dermoscopy",
        ),
        DatasetSpec(
            name="fitzpatrick17k-skincon",
            images="fitzpatrick/skincon/imgs",
            metadata="fitzpatrick/skincon/train/metadata.csv",
            concepts="fitzpatrick/skincon/concepts.csv",
            num_classes=114,
            multilabel=True,
            augment="dermoscopy",
            notes="SkinCon concept annotations: 114 binary concepts, not 114 classes.",
        ),
        DatasetSpec(
            name="ddi",
            images="ddi/imgs",
            metadata="ddi/ddi_metadata.csv",
            concepts="ddi/skincon-annotations.csv",
            num_classes=24,
            augment="dermoscopy",
        ),
        DatasetSpec(
            name="mra-midas",
            images="mra-midas/imgs",
            metadata="mra-midas/metadata.csv",
            num_classes=15,
            augment="dermoscopy",
        ),
        DatasetSpec(
            name="cub200",
            images="CUB_200_2011/imgs",
            metadata="CUB_200_2011/metadata.csv",
            concepts="CUB_200_2011/concepts.csv",
            num_classes=200,
            augment="standard",
            notes="Natural images: no vertical flips. An upside-down bird is not a bird.",
        ),
        DatasetSpec(
            name="celeba",
            images="celeba/splits",
            metadata="celeba/list_attr_celeba.csv",
            num_classes=40,
            label_key="attributes",
            multilabel=True,
            augment="standard",
            notes="40 binary attributes -> multilabel BCE, not 40-way softmax.",
        ),
        DatasetSpec(
            name="scin",
            images="scin/imgs",
            metadata="scin/scin_labels.csv",
            num_classes=0,
            notes="No label schema pinned yet.",
        ),
    ]
}


def list_datasets() -> list[str]:
    return sorted(REGISTRY)


def get_spec(name: str) -> DatasetSpec:
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}. Known: {list_datasets()}")
    return REGISTRY[name]


# --------------------------------------------------------------------------
# Raw loading -- no model, no transforms, no training
# --------------------------------------------------------------------------
def _load_one(name: str, root: Path):
    """Load a single imagefolder, drop rows with a null label. No casting."""
    from datasets import load_dataset

    spec = get_spec(name)
    # drop_labels=True: labels live in metadata.csv inside the image dir. Without
    # it, imagefolder *also* invents a label from subfolder names and the two collide.
    ds = load_dataset("imagefolder", data_dir=str(root / spec.images), drop_labels=True)
    for split in list(ds):
        n = len(ds[split])
        ds[split] = ds[split].filter(lambda x: x[spec.label_key] is not None)
        dropped = n - len(ds[split])
        if dropped:
            print(f"  ! {name}/{split}: dropped {dropped} rows with no label")
    return spec, ds


def get_dataset(
    name: str | list[str],
    *,
    data_root: str | Path | None = None,
    cast_labels: bool | None = None,
):
    """Raw HF DatasetDict(s): PIL images under "image", labels under `label_key`.

    cast_labels:
      * single dataset -> casts the label column to a ClassLabel and checks the
        count against the registry (a head sized to the wrong number of classes
        trains happily and scores nonsense).
      * multiple datasets -> defaults to *not* casting.
    """
    from datasets import ClassLabel, DatasetDict, concatenate_datasets

    names = [name] if isinstance(name, str) else list(name)
    root = get_data_root(data_root)
    multi = len(names) > 1

    if cast_labels is None:
        cast_labels = not multi

    specs, dicts = [], []
    for n in names:
        spec, ds = _load_one(n, root)
        if multi:
            keep = {"image", spec.label_key}
            for split in list(ds):
                extra = [c for c in ds[split].column_names if c not in keep]
                if extra:
                    ds[split] = ds[split].remove_columns(extra)
        specs.append(spec)
        dicts.append(ds)

    label_key = specs[0].label_key

    if not multi:
        combined = dicts[0]
    else:
        splits = set().union(*[set(d.keys()) for d in dicts])
        combined = DatasetDict(
            {
                s: concatenate_datasets([d[s] for d in dicts if s in d])
                for s in splits
                if any(s in d for d in dicts)
            }
        )

    do_cast = cast_labels and not specs[0].multilabel

    if do_cast:
        values = sorted({v for split in combined for v in combined[split][label_key]})
        if not multi and specs[0].num_classes and len(values) != specs[0].num_classes:
            raise ValueError(
                f"{names[0]}: registry says {specs[0].num_classes} classes, the data has "
                f"{len(values)}. Fix one - a head sized to the wrong number of classes "
                f"trains happily and scores nonsense."
            )
        combined = combined.cast_column(label_key, ClassLabel(names=[str(v) for v in values]))

    return combined


def get_splits(
    name: str | list[str],
    *,
    model_key: str | None = None,
    augment: Augment | None = None,
    data_root: str | Path | None = None,
    cast_labels: bool | None = None,
):
    """
    (train, val, test) with transforms attached, ready for a DataLoader.

    Missing splits come back as None. The transform/augment defaults follow the
    *first* dataset's registry entry when several are combined.
    """
    names = [name] if isinstance(name, str) else list(name)
    spec0 = get_spec(names[0])
    ds = get_dataset(names, data_root=data_root, cast_labels=cast_labels)

    train = ds.get("train")
    val = ds.get("validation") or ds.get("valid")
    test = ds.get("test")

    if model_key is not None:
        aug = augment or spec0.augment
        train_tf = build_transforms(model_key, train=True, augment=aug)
        eval_tf = build_transforms(model_key, train=False)
        if train is not None:
            attach_transform(train, train_tf, label_key=spec0.label_key)
        for split in (val, test):
            if split is not None:
                attach_transform(split, eval_tf, label_key=spec0.label_key)

    return train, val, test


def processor_stats(model_key: str) -> tuple[list[float], list[float], int]:
    """
    (mean, std, size) taken from the model's *own* image processor.
    """
    from .backbone import load_image_processor
    from .registry import get_spec as get_model_spec

    proc = load_image_processor(model_key)
    mean = list(getattr(proc, "image_mean", [0.485, 0.456, 0.406]))
    std = list(getattr(proc, "image_std", [0.229, 0.224, 0.225]))

    size = get_model_spec(model_key).image_size
    for src in (getattr(proc, "crop_size", None) or {}, getattr(proc, "size", None) or {}):
        if isinstance(src, dict):
            if "height" in src:
                size = int(src["height"])
                break
            if "shortest_edge" in src:
                size = int(src["shortest_edge"])
                break
    return mean, std, size


def build_transforms(model_key: str, *, train: bool, augment: Augment = "standard"):
    """
    A torchvision v2 pipeline matched to the backbone's preprocessing.

    augment="dermoscopy" adds vertical flips and 90-degree rotations: label-preserving
    for skin lesions (there is no canonical "up") and wrong for natural images.
    """
    from torchvision.transforms import v2

    mean, std, size = processor_stats(model_key)

    if not train or augment == "none":
        return v2.Compose([
            v2.ToImage(),
            v2.Resize((size, size), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ])

    steps = [
        v2.ToImage(),
        v2.RandomResizedCrop((size, size), scale=(0.8, 1.0),
                             interpolation=v2.InterpolationMode.BICUBIC),
        v2.RandomHorizontalFlip(p=0.5),
    ]
    if augment == "dermoscopy":
        steps += [v2.RandomVerticalFlip(p=0.5), v2.RandomRotation(degrees=90)]
    steps += [
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.01),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ]
    return v2.Compose(steps)


def attach_transform(dataset, transform, *, image_key: str = "image", label_key: str = "label"):
    """
    Lazily map an HF dataset to {"pixel_values", "labels"}."""

    def apply(batch):
        return {
            "pixel_values": [transform(img.convert("RGB")) for img in batch[image_key]],
            "labels": batch[label_key],
        }

    dataset.set_transform(apply)
    return dataset


def denormalize(pixel_values: torch.Tensor, model_key: str) -> torch.Tensor:
    """Undo the normalisation so you can actually look at a tensor. (C,H,W) or (B,C,H,W)."""
    mean, std, _ = processor_stats(model_key)
    m = torch.tensor(mean).view(-1, 1, 1)
    s = torch.tensor(std).view(-1, 1, 1)
    x = pixel_values.detach().cpu()
    if x.ndim == 4:
        m, s = m.unsqueeze(0), s.unsqueeze(0)
    return (x * s + m).clamp(0, 1)

def class_balanced_sampler(dataset, label_key: str = "label") -> WeightedRandomSampler:
    """
    Inverse-frequency sampling *within* one task.

    Orthogonal to MultiTaskLoader's strategy, which balances tasks *against each other*.
    """
    import numpy as np

    labels = np.asarray(dataset.with_format("numpy")[label_key])
    if labels.ndim > 1:
        raise ValueError(
            "class_balanced_sampler is for single-label tasks; a multilabel sample has "
            "no single class. Use per-class pos_weight in the loss instead."
        )
    labels = labels.astype(int)
    counts = np.bincount(labels)
    weights = np.zeros_like(counts, dtype=np.float64)
    nz = counts > 0
    weights[nz] = 1.0 / counts[nz]
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights[labels], dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )


def task_spec(name: str, **overrides):
    """The TaskSpec implied by the registry entry (classes, multilabel)."""
    from .heads import TaskSpec

    spec = get_spec(name)
    kwargs = dict(
        name=spec.name,
        num_classes=spec.num_classes,
        multilabel=spec.multilabel,
    )
    kwargs.update(overrides)
    return TaskSpec(**kwargs)
