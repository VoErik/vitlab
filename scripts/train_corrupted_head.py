"""Train the *corrupted* classifier for the watermark audit.

Plants the "SFS" watermark on every melanoma image of the DermaMNIST training
split (strict correlation) and trains a classification head on top of the frozen
foundation model. The head learns the watermark as a shortcut for melanoma; the
audit script then excises it.

    uv run python scripts/train_watermark_head.py \
        --model dinov2-base --mode frozen \
        --out runs/dermamnist-watermarked --epochs 30

"""

from __future__ import annotations

import argparse
from pathlib import Path

import vitlab
from vitlab.batching import MultiTaskLoader, make_loader
from vitlab.datasets import class_balanced_sampler, get_spec, task_spec
from vitlab.eval.watermark import corrupted_splits, melanoma_index
from vitlab.model import ModelConfig
from vitlab.train import OptimConfig, TrainConfig, train


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2-base", help="key from vitlab.registry")
    p.add_argument("--dataset", default="dermamnist")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--mode", default="frozen", choices=["frozen", "lora", "full"],
                   help="frozen == train only the head on the foundation model (thesis setting)")
    p.add_argument("--pooling", default="cls", choices=["cls", "mean", "cls_mean", "attn"])
    p.add_argument("--target-class", type=int, default=None,
                   help="class to correlate the watermark with (default: auto-detect melanoma)")
    p.add_argument("--wm-prob", type=float, default=1.0, help="P(watermark | target class) in train")
    p.add_argument("--wm-off-prob", type=float, default=0.0,
                   help="P(watermark | non-target class) in train. >0 breaks the perfect "
                        "correlation so a trainable backbone keeps watermark and class separable "
                        "(recommended with --mode lora/full).")
    p.add_argument("--wm-text", default="SFS")
    p.add_argument("--balanced", action="store_true",
                   help="inverse-frequency class sampling (helps the rare melanoma class)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--out", default="runs/dermamnist-watermarked")
    args = p.parse_args()

    if args.data_root is not None:
        vitlab.set_data_root(args.data_root)

    spec = get_spec(args.dataset)
    if args.target_class is None:
        target, names = melanoma_index(args.dataset, data_root=args.data_root)
        print(f"target class (watermark <-> class): {target} = {names[target]!r}")
    else:
        target = args.target_class

    wm_kwargs = {"text": args.wm_text}
    train_split, val_split, _ = corrupted_splits(
        args.dataset, model_key=args.model, target_label=target,
        train_wm_prob=args.wm_prob, train_off_prob=args.wm_off_prob, eval_watermarked=True,
        data_root=args.data_root, watermark_kwargs=wm_kwargs,
    )

    sampler = None
    if args.balanced and not spec.multilabel:
        sampler = class_balanced_sampler(train_split, spec.label_key)

    train_loader = make_loader(
        train_split, args.batch_size, sampler=sampler, num_workers=args.workers
    )
    val_loaders = {}
    if val_split is not None:
        val_loaders[args.dataset] = make_loader(
            val_split, args.batch_size, shuffle=False, num_workers=args.workers
        )

    loader = MultiTaskLoader({args.dataset: train_loader})
    tasks = [task_spec(args.dataset, pooling=args.pooling)]

    print(f"=== {args.model} | {args.mode} | corrupted {args.dataset} "
          f"(watermark {args.wm_text!r} on class {target}, p={args.wm_prob}, off={args.wm_off_prob})")
    print(f"  train {len(train_split)}  val {len(val_split) if val_split is not None else 0}  "
          f"{spec.num_classes} classes")

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