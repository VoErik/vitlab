from __future__ import annotations

import argparse
from pathlib import Path

import vitlab
from vitlab.batching import MultiTaskLoader, make_loader
from vitlab.datasets import class_balanced_sampler, get_spec, get_splits, list_datasets, task_spec
from vitlab.model import ModelConfig
from vitlab.train import OptimConfig, TrainConfig, train


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="dinov2-base", help="key from vitlab.registry")
    p.add_argument("--tasks", nargs="+", required=True, choices=list_datasets())
    p.add_argument("--data-root", type=Path, default=None,
                   help="overrides vitlab/config.py for this run")
    p.add_argument("--mode", default="lora", choices=["frozen", "lora", "full"])
    p.add_argument("--pooling", default="cls", choices=["cls", "mean", "cls_mean", "attn"])
    p.add_argument("--augment", default=None, choices=["none", "standard", "dermoscopy"],
                   help="default: whatever the dataset registry says is right for it")
    p.add_argument("--balanced", action="store_true",
                   help="inverse-frequency sampling of classes within each task")
    p.add_argument("--strategy", default="temperature",
                   choices=["proportional", "uniform", "temperature", "round_robin"])
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--out", default="runs/default")
    args = p.parse_args()

    if args.data_root is not None:
        vitlab.set_data_root(args.data_root)

    tasks, train_loaders, val_loaders = [], {}, {}
    print(f"=== {args.model} | {args.mode} | {', '.join(args.tasks)}")

    for name in args.tasks:
        d = get_spec(name)
        d_train, d_val, _ = get_splits(name, model_key=args.model, augment=args.augment)

        sampler = None
        if args.balanced:
            if d.multilabel:
                print(f"  ! {name}: multilabel, skipping class balancing (undefined)")
            else:
                sampler = class_balanced_sampler(d_train, d.label_key)

        train_loaders[name] = make_loader(
            d_train, args.batch_size, sampler=sampler, num_workers=args.workers
        )
        if d_val is not None:
            val_loaders[name] = make_loader(
                d_val, args.batch_size, shuffle=False, num_workers=args.workers
            )

        tasks.append(task_spec(name, pooling=args.pooling))
        print(f"  {name:24s} {len(d_train):6d} train  "
              f"{len(d_val) if d_val is not None else 0:6d} val  "
              f"{d.num_classes:4d} classes"
              f"{'  [balanced]' if sampler else ''}"
              f"{'  [multilabel]' if d.multilabel else ''}")

    loader = MultiTaskLoader(
        train_loaders, strategy=args.strategy, temperature=args.temperature
    )
    probs = {k: round(v, 2) for k, v in loader.task_probs.items()}
    print(f"  sampling: {args.strategy} {probs}  ({len(loader)} steps/epoch)\n")

    cfg = TrainConfig(
        model=ModelConfig(model_key=args.model, tasks=tasks, backbone_mode=args.mode),
        optim=OptimConfig(backbone_lr=args.backbone_lr, head_lr=args.head_lr),
        epochs=args.epochs,
        mixed_precision=args.precision,
        output_dir=args.out,
    )
    train(cfg, loader, val_loaders or None)
    print(f"\ndone -> {args.out}/best   (reload: vitlab.load_model('{args.out}/best'))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
