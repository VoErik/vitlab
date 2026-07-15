#!/usr/bin/env bash

set -euo pipefail

MODELS=(
    "vit-in21k-base"
    "clip-large"
    "dinov3-base"
    "dinov2-base" 
)

TASKS="fitzpatrick17k dermamnist 7ptderm mra-midas"

for model in "${MODELS[@]}"; do
    run_name="runs/${model}_full_epochs30"
    
    echo "Starting training for model: ${model} -> ${run_name}"
    
    uv run python scripts/train.py \
        --model "${model}" \
        --tasks ${TASKS} \
        --mode full \
        --pooling cls \
        --epochs 30 \
        --batch-size 64 \
        --out "${run_name}" \
        --balanced
        
    echo "Finished training for model: ${model}"
    echo "----------------------------------------"
done