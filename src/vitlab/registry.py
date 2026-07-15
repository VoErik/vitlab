"""
Central registry of vision backbones.

Revisions live in ``revisions.json`` next to this file and are filled in by ``python -m vitlab.cli pin``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

Family = Literal["vit", "mae", "dinov2", "dinov3", "clip"]

_REVISIONS_PATH = Path(__file__).with_name("revisions.json")


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to reconstruct a backbone byte-for-byte."""

    key: str
    hf_id: str
    family: Family
    hf_class: str
    image_size: int
    patch_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_registers: int = 0
    has_cls: bool = True
    gated: bool = False # requires accepting a licence + HF_TOKEN
    revision: str | None = None
    notes: str = ""

    @property
    def n_prefix_tokens(self) -> int:
        """Tokens before the patch grid: CLS + registers."""
        return 1 + self.n_registers

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def resolved(self) -> "ModelSpec":
        """Attach the pinned revision from revisions.json."""
        rev = _load_revisions().get(self.hf_id)
        return replace(self, revision=rev)


def _load_revisions() -> dict[str, str]:
    if not _REVISIONS_PATH.exists():
        return {}
    return json.loads(_REVISIONS_PATH.read_text())


def _write_revisions(revs: dict[str, str]) -> None:
    _REVISIONS_PATH.write_text(json.dumps(revs, indent=2, sort_keys=True) + "\n")


_SPECS: list[ModelSpec] = [
    ModelSpec(
        key="dinov2-base",
        hf_id="facebook/dinov2-base",
        family="dinov2",
        hf_class="Dinov2Model",
        image_size=224,
        patch_size=14,
        d_model=768,
        n_layers=12,
        n_heads=12,
        n_registers=0,
        notes="Self-distillation. patch14 -> 256 patch tokens at 224px.",
    ),
    ModelSpec(
        key="dinov2-reg-base", # TODO: some issues here, look into it
        hf_id="facebook/dinov2-with-registers-base",
        family="dinov2",
        hf_class="Dinov2WithRegistersModel",
        image_size=224,
        patch_size=14,
        d_model=768,
        n_layers=12,
        n_heads=12,
        n_registers=4,
        notes="DINOv2 + 4 registers. Cleaner attention maps; good SAE control.",
    ),
    ModelSpec(
        key="dinov3-base",
        hf_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
        family="dinov3",
        hf_class="DINOv3ViTModel",
        image_size=224,
        patch_size=16,
        d_model=768,
        n_layers=12,
        n_heads=12,
        n_registers=4,
        gated=True,
        notes="RoPE, 4 registers, gated repo: accept the licence and set HF_TOKEN.",
    ),
    ModelSpec(
        key="clip-base", # TODO: some issues here, look into it
        hf_id="laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
        family="clip",
        hf_class="CLIPVisionModel",
        image_size=224,
        patch_size=16,
        d_model=768,
        n_layers=12,
        n_heads=12,
        notes="LAION-2B. Vision tower only; text tower is dropped.",
    ),
    ModelSpec(
        key="clip-large",
        hf_id="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        family="clip",
        hf_class="CLIPVisionModel",
        image_size=224,
        patch_size=14,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        notes="Fallback if the B/16 repo has no HF-format config.",
    ),
    ModelSpec(
        key="mae-base",
        hf_id="facebook/vit-mae-base",
        family="mae",
        hf_class="ViTMAEModel",
        image_size=224,
        patch_size=16,
        d_model=768,
        n_layers=12,
        n_heads=12,
        notes="mask_ratio forced to 0 and noise made deterministic -- see backbone.py.",
    ),
    ModelSpec(
        key="vit-in21k-base",
        hf_id="google/vit-base-patch16-224-in21k",
        family="vit",
        hf_class="ViTModel",
        image_size=224,
        patch_size=16,
        d_model=768,
        n_layers=12,
        n_heads=12,
        notes="Supervised ImageNet-21k.",
    ),
]

REGISTRY: dict[str, ModelSpec] = {s.key: s for s in _SPECS}


def get_spec(key: str) -> ModelSpec:
    if key not in REGISTRY:
        raise KeyError(f"Unknown model key {key!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[key].resolved()


def list_models() -> list[str]:
    return sorted(REGISTRY)
