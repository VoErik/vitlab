# vitlab

Research code for the thesis *Unsupervised Concept Extraction with Sparse Autoencoders -- A Study of Sparse Autoencoders for Concept-based Explanations in Dermatology*. It loads pinned ViT backbones, extracts their activations, trains Sparse Autoencoders (SAEs) on them to recover unsupervised visual concepts, and uses those concepts for attribution, cross-layer circuits, and model auditing. An optional web app browses the learned concept atlas.

## Install

Python 3.13, managed with [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync
uv sync --group dev # pytest, ruff
uv sync --group interp # overcomplete, nnsight
cd app/frontend && npm install # only if you want the app
```

**DINOv3** is a gated Hugging Face repo: accept the licence, then `huggingface-cli login` and `export HF_TOKEN=...`.

## Setup

Set the data root before anything else — the default in `src/vitlab/config.py` is machine-specific. Either edit `DATA_ROOT` there, call `vitlab.set_data_root(...)`, or pass `--data-root` to scripts.

```bash
uv run vitlab pin    # record HF commit SHAs into revisions.json
uv run vitlab doctor # verify hook sites against real weights
```

Backbones are in `src/vitlab/registry.py` (`dinov3-base`, `dinov2-base`, `dinov2-reg-base`, `clip-base/large`, `mae-base`, `vit-in21k-base`). Datasets are in `src/vitlab/datasets.py` (`fitzpatrick17k`, `fitzpatrick17k-skincon`, `dermamnist`, `7ptderm`, `mra-midas`, `ddi`, `scin`, `cub200`, `celeba`); see that module for the on-disk layout each expects.

For a number of datasets you need to accept their own terms and conditions. `assets/datasets`. contains the `.csv` files used that outline the respective splits of the data. For images please download from the respective provider.

## Pipeline

```bash
# 1. fine-tune the backbone (multi-task heads; --mode frozen|lora|full)
uv run python scripts/train.py --model dinov3-base \
    --tasks fitzpatrick17k dermamnist 7ptderm mra-midas \
    --mode full --epochs 30 --balanced --out runs/dinov3_full

# 2. extract activations (one site, or every layer via extract_all_layers.sh)
uv run python scripts/extract_activations.py --run runs/dinov3_full/best \
    --datasets fitzpatrick17k --site blocks.8.resid_post

# 3. train an SAE (CLI overrides the YAML)
uv run python scripts/train_sae.py --config configs/sae.yaml \
    --acts acts/dinov3-base/fitzpatrick17k/blocks.8.resid_post

# 4. evaluate
uv run python scripts/evaluate_sae.py --bank saes/dinov3_fitz \
    --acts-root acts/dinov3-base/fitzpatrick17k --aspects dictionary concepts
```

SAE types (`configs/sae.yaml`, `sae_type`): `SAE`, `GatedSAE`, `JumpSAE`, `TopKSAE`, `BatchTopKSAE`, plus matching-pursuit (`MpSAE`, `OMPSAE`) and archetypal (`RATopKSAE`, `RAJumpSAE`) variants. Matryoshka and archetypal training are toggled with `as_matryoshka` / `as_archetypal` in the config. Concept-bottleneck classifier: `scripts/train_sae_cbm.py` with `configs/cbm.yaml`.

Activation sites (`src/vitlab/activations.py`), identical across families: `embed`, `blocks.{i}.{resid_pre,attn_z,attn_out,mlp_out,resid_post}`, `final_norm`.

## Explanation

- **Attribution** — `src/vitlab/attribution/`: direct-logit, act×grad patching, and ablation. `scripts/morf.py` produces MoRF curves comparing SAE attribution against LRP (`lxt`) and GradCAM.
- **Circuits** — `src/vitlab/circuits/`: SAE feature nodes + cross-layer edges, scored by faithfulness / completeness / causality.
- **Auditing** — the planted "SFS" watermark shortcut: `scripts/train_corrupted_head.py`, then `scripts/auditing_single.py` / `auditing_sweep.py`. `scripts/steering.sh` runs the full corrupt $\rightarrow$ extract $\rightarrow$ train $\rightarrow$ audit $\rightarrow$ figure sequence.

## App

FastAPI backend (`app/backend`, :8000) + Vite/React frontend (`app/frontend`, :5173). Precompute an atlas, then launch:

```bash
uv run python scripts/app_precompute_atlas.py --model runs/dinov3_full/best \
    --bank saes/dinov3_fitz --dataset fitzpatrick17k-skincon \
    --split train --site blocks.8.resid_post
./run_app.sh
```