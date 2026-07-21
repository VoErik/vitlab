"""App configuration. No env vars -- edit these (mirrors vitlab's config.py stance).

Paths resolve lazily so the app boots and reports a clear error rather than crashing
on import if DATA_ROOT isn't set yet.
"""

from __future__ import annotations

from pathlib import Path

from vitlab.config import get_data_root

DEVICE: str = "cuda"          # resolved to cpu at runtime if cuda unavailable (state.py)

_CHECKPOINTS = "runs/dinov2-base_full_epochs30"          # fine-tuned model dirs (config.json + heads)
_BANKS = "saes"                # SAE bank trees (per-layer SAE dirs)
_ATLAS = "atlas"               # precomputed atlas artifacts
_UPLOAD_CACHE = ".app_cache"
_ARTEFACT_DIR = "thesis"


def data_root() -> Path:
    return get_data_root()


def checkpoints_dir() -> Path:
    return data_root() / _ARTEFACT_DIR / _CHECKPOINTS


def banks_dir() -> Path:
    return data_root() / _ARTEFACT_DIR / _BANKS


def atlas_dir() -> Path:
    return data_root() / _ARTEFACT_DIR / _ATLAS


def upload_cache_dir() -> Path:
    return data_root() / _ARTEFACT_DIR / _UPLOAD_CACHE
