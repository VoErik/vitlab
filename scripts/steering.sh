#!/usr/bin/env bash
RUN=runs/dinov3-watermarked-full/best
BANK=saes/dinov3_dermamnist_corrupt_full
ACTS=data/acts_wm

# 1) train the corrupted model WITH the backbone, imperfect correlation
uv run python scripts/train_corrupted_head.py \
    --model dinov3-base --mode full --dataset dermamnist \
    --wm-prob 0.9 --wm-off-prob 0.05 --balanced --epochs 30 \
    --out runs/dinov3-watermarked-full

MK=$(uv run python -c "import vitlab;print(vitlab.load_model('$RUN').spec.key)")
N=$(uv run python -c "import vitlab;print(vitlab.load_model('$RUN').spec.n_layers)")

# 2) extract WATERMARKED activations from the trained backbone, every layer
for ((L=0; L<N; L++)); do
  uv run python scripts/extract_watermarked_activations.py \
      --run $RUN --dataset dermamnist --site blocks.${L}.resid_post \
      --wm-prob 0.9 --wm-off-prob 0.05 --split train --tokens patches --out-root $ACTS
done

# 3) train one SAE per layer
for ((L=0; L<N; L++)); do
  SITE=blocks.${L}.resid_post
  uv run python scripts/train_sae.py --config configs/sae.yaml \
      --acts $ACTS/$MK/dermamnist/$SITE --site $SITE --out $BANK/$SITE
done

# 4) audit
uv run python scripts/auditing_sweep.py --run $RUN --bank $BANK \
    --identify-all-classes --cumulative --control-seeds 5 --n-features 16 \
    --out out/watermark/sweep_dinov3_full.json

uv run python scripts/auditing_single.py --run $RUN --bank $BANK \
    --site blocks.0.resid_post,blocks.1.resid_post \
    --identify-all-classes --restrict-corners --steer-scale feature \
    --n-features 8 --alphas 0 1 2 4 8 \
    --out out/watermark/steer_dinov3_full.json

# 5) figures
uv run python scripts/make_watermark_figure.py --results out/watermark/sweep_dinov3_full.json --out out/figures/sweep_dinov3_full.pdf
uv run python scripts/make_watermark_figure.py --results out/watermark/steer_dinov3_full.json --out out/figures/steer_dinov3_full.pdf