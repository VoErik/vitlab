from __future__ import annotations

from pathlib import Path

from vitlab.config import get_data_root

DEVICE: str = "cuda"

_CHECKPOINTS = "runs/dinov2-base_full_epochs30" 
_BANKS = "saes"
_ATLAS = "atlas"
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
