"""
Extract activations from a trained model to disk.

Single dataset:
    uv run python extract_activations.py --run runs/dino-derm/best \
        --datasets fitzpatrick17k --site blocks.11.resid_post

Combined datasets (embeddings pooled across all of them):
    uv run python extract_activations.py --run runs/dino-derm/best \
        --datasets fitzpatrick17k dermamnist 7ptderm --site blocks.11.resid_post
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import vitlab
from vitlab.batching import make_loader
from vitlab.datasets import get_splits
from vitlab import ActivationStore, extract_activations, seed_everything


def slugify_datasets(names: list[str]) -> str:
    return "+".join(sorted(names))


def default_out_dir(root: Path, model_key: str, datasets: list[str], site: str) -> Path:
    """acts/<model>/<datasets>/<site>."""
    return root / model_key / slugify_datasets(datasets) / site.replace("/", "_")


def _pixels_only_collate(batch):
    return {"pixel_values": torch.stack([b["pixel_values"] for b in batch])}


def build_loader(datasets, model_key, split, batch_size, workers, augment):
    """One loader over one or several datasets, transformed for `model_key`."""
    train, val, test = get_splits(
        datasets, model_key=model_key, augment=augment, cast_labels=False
    )
    chosen = {"train": train, "val": val, "test": test}[split]
    if chosen is None:
        raise SystemExit(f"datasets {datasets} have no '{split}' split")
    return make_loader(
        chosen, batch_size, shuffle=False, num_workers=workers,
        collate_fn=_pixels_only_collate,
    ), len(chosen)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="trained checkpoint dir (has config.json)")
    p.add_argument("--datasets", nargs="+", required=True,
                   help="one name, or several to combine into a single activation set")
    p.add_argument("--site", default="blocks.11.resid_post")
    p.add_argument("--tokens", default="patches", choices=["patches", "cls", "registers", "all"])
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--augment", default="none", choices=["none", "standard", "dermoscopy"],
                   help="usually 'none' for collection -- you want the real images")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=None, help="cap; default = whole dataset")
    p.add_argument("--shard-size", type=int, default=500_000, help="tokens per .pt shard")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--out-root", type=Path, default=Path("/home/voigt/data/thesis/acts"))
    p.add_argument("--out", type=Path, default=None, help="explicit output dir (overrides --out-root layout)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.data_root is not None:
        vitlab.set_data_root(args.data_root)
    seed_everything(args.seed)

    model = vitlab.load_model(args.run, device=args.device)
    print(f"model {model.spec.key}: {model.spec.n_layers} blocks, d_model={model.spec.d_model}")

    loader, n_images = build_loader(
        args.datasets, model.spec.key, args.split, args.batch_size, workers=8, augment=args.augment
    )
    out_dir = args.out or default_out_dir(args.out_root, model.spec.key, args.datasets, args.site)
    print(f"collecting {args.site} ({args.tokens}) from {n_images} images "
          f"[{'+'.join(args.datasets)}] -> {out_dir}")

    store = extract_activations(
        model.reader, loader, args.site, out_dir,
        token_select=args.tokens,
        max_tokens=args.max_tokens,
        shard_size=args.shard_size,
        device=args.device,
        dtype=torch.float16 if args.dtype == "float16" else torch.float32,
    )

    # sanity
    check = ActivationStore(out_dir)
    X = check.dataset().tensors[0]
    print(f"reload check: {tuple(X.shape)} {X.dtype}, "
          f"model={check.manifest.model_key}, site={check.manifest.site}")
    assert X.shape == (store.manifest.num_tokens, model.spec.d_model)
    print(f"\nOK -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())