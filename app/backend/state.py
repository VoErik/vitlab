from __future__ import annotations

import io
import uuid
from dataclasses import dataclass, field

import torch
from PIL import Image

import vitlab
from vitlab.backbone import load_image_processor
from vitlab.datasets import denormalize
from vitlab.sae import discover_bank

from . import config


def resolve_device() -> str:
    if config.DEVICE == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return config.DEVICE


DEVICE = resolve_device()

_models: dict[str, "vitlab.MultiTaskViT"] = {}
_banks: dict[str, object] = {}
_processors: dict[str, object] = {}


def get_model(checkpoint: str):
    """Load + cache a MultiTaskViT by checkpoint dir (absolute or under CHECKPOINTS_DIR)."""
    path = str((config.checkpoints_dir() / checkpoint) if not _is_abs(checkpoint) else checkpoint)
    if path not in _models:
        _models[path] = vitlab.load_model(path, device=DEVICE)
    return _models[path]


def get_bank(bank_path: str):
    path = str((config.banks_dir() / bank_path) if not _is_abs(bank_path) else bank_path)
    if path not in _banks:
        _banks[path] = discover_bank(path, device=DEVICE)
    return _banks[path]


def get_processor(model_key: str):
    if model_key not in _processors:
        _processors[model_key] = load_image_processor(model_key)
    return _processors[model_key]


def _is_abs(p: str) -> bool:
    from pathlib import Path
    return Path(p).is_absolute()


@dataclass
class CachedImage:
    pixel_values: torch.Tensor
    model_key: str
    shown_png: bytes
    meta: dict = field(default_factory=dict)


_images: dict[str, CachedImage] = {}


def cache_upload(pil_image: Image.Image, model_key: str) -> str:
    """Preprocess a PIL upload for `model_key`, cache tensor + denormalized PNG, return a token."""
    proc = get_processor(model_key)
    px = proc(pil_image.convert("RGB"), return_tensors="pt")["pixel_values"]
    shown = denormalize(px[0], model_key)
    png = _tensor_to_png(shown)
    token = uuid.uuid4().hex
    _images[token] = CachedImage(pixel_values=px, model_key=model_key, shown_png=png)
    return token


def get_image(token: str) -> CachedImage:
    if token not in _images:
        raise KeyError(f"unknown image token {token!r} (upload expired or never created)")
    return _images[token]


def _tensor_to_png(chw_float: torch.Tensor) -> bytes:
    arr = (chw_float.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy())
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()
