"""Extract activations from the *corrupted* model on *watermarked* images.

Needed for the trained-backbone audit: when the backbone is fine-tuned on the
corrupted data it builds a watermark feature that only fires when the mark is
present, so the SAE must be trained on watermarked activations to allocate a latent
to it. (With a frozen backbone this isn't necessary -- the base model's clean
activations already carry generic corner/edge latents the mark rides on -- so the
plain scripts/extract_activations.py is fine there.)

The corruption here mirrors the *training* distribution (same target class, same
wm_prob / wm_off_prob), so the SAE sees exactly what the backbone was trained on.

    uv run python scripts/extract_watermarked_activations.py \
        --run runs/dinov3-watermarked-full/best \
        --dataset dermamnist --site blocks.4.resid_post \
        --wm-prob 0.9 --wm-off-prob 0.05 \
        --out-root /home/voigt/data/thesis/acts_wm

Loop over layers with the shell snippet in scripts/README_watermark.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import vitlab
from vitlab import ActivationStore, extract_activations, seed_everything
from vitlab.batching import make_loader
from vitlab.eval import watermark as wm


def _pixels_only_collate(batch):
    return {"pixel_values": torch.stack([b["pixel_values"] for b in batch])}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="corrupted checkpoint dir")
    p.add_argument("--dataset", default="dermamnist")
    p.add_argument("--site", default="blocks.4.resid_post")
    p.add_argument("--tokens", default="patches", choices=["patches", "cls", "registers", "all"])
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--target-class", type=int, default=None, help="default: auto-detect melanoma")
    p.add_argument("--wm-prob", type=float, default=1.0, help="P(watermark | target class)")
    p.add_argument("--wm-off-prob", type=float, default=0.0, help="P(watermark | non-target class)")
    p.add_argument("--wm-text", default="SFS")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--shard-size", type=int, default=500_000)
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--out-root", type=Path, default=Path("/home/voigt/data/thesis/acts_wm"))
    p.add_argument("--out", type=Path, default=None, help="explicit output dir (overrides layout)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.data_root is not None:
        vitlab.set_data_root(args.data_root)
    seed_everything(args.seed)

    model = vitlab.load_model(args.run, device=args.device)
    print(f"model {model.spec.key}: {model.spec.n_layers} blocks, d_model={model.spec.d_model}")

    if args.target_class is None:
        target, names = wm.melanoma_index(args.dataset, data_root=args.data_root)
        print(f"watermark <-> class {target} ({names[target]})")
    else:
        target = args.target_class

    wm_kwargs = {"text": args.wm_text}
    train, val, test = wm.corrupted_splits(
        args.dataset, model_key=model.spec.key, target_label=target,
        train_wm_prob=args.wm_prob, train_off_prob=args.wm_off_prob,
        eval_watermarked=True, data_root=args.data_root, watermark_kwargs=wm_kwargs,
    )
    chosen = {"train": train, "val": val, "test": test}[args.split]
    if chosen is None:
        raise SystemExit(f"{args.dataset} has no '{args.split}' split")
    loader = make_loader(
        chosen, args.batch_size, shuffle=False, num_workers=8, collate_fn=_pixels_only_collate
    )

    out_dir = args.out or (args.out_root / model.spec.key / args.dataset / args.site.replace("/", "_"))
    print(f"collecting {args.site} ({args.tokens}) from {len(chosen)} watermarked images "
          f"(p={args.wm_prob}, off={args.wm_off_prob}) -> {out_dir}")

    store = extract_activations(
        model.reader, loader, args.site, out_dir,
        token_select=args.tokens, max_tokens=args.max_tokens, shard_size=args.shard_size,
        device=args.device, dtype=torch.float16 if args.dtype == "float16" else torch.float32,
    )

    check = ActivationStore(out_dir)
    X = check.dataset().tensors[0]
    print(f"reload check: {tuple(X.shape)} {X.dtype}, site={check.manifest.site}")
    print(f"\nOK -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())