from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import transformers

from .arch import ArchSpec, get_arch
from .registry import ModelSpec, get_spec


@dataclass
class BackboneOutput:
    """Token layout is normalised: [CLS] [registers...] [patches...]."""

    tokens: torch.Tensor
    cls: torch.Tensor
    patches: torch.Tensor
    registers: torch.Tensor

    def pooled(self, mode: str = "cls") -> torch.Tensor:
        if mode == "cls":
            return self.cls
        if mode == "mean":
            return self.patches.mean(dim=1)
        if mode == "cls_mean":
            return torch.cat([self.cls, self.patches.mean(dim=1)], dim=-1)
        raise ValueError(f"unknown pooling {mode!r}")


class VisionBackbone(nn.Module):
    """Loads a pinned HF vision tower and normalises its forward pass."""

    def __init__(self, spec: ModelSpec, model: nn.Module):
        super().__init__()
        self.spec = spec
        self.arch: ArchSpec = get_arch(spec.family)
        self.model = model

    @classmethod
    def from_key(
        cls,
        key: str,
        *,
        dtype: torch.dtype | None = None,
        require_pinned_revision: bool = True,
        **kwargs,
    ) -> "VisionBackbone":
        spec = get_spec(key)
        if require_pinned_revision and spec.revision is None:
            raise RuntimeError(
                f"{spec.hf_id} has no pinned revision. Run `python -m vitlab.cli pin` "
                f"once and commit revisions.json, or pass require_pinned_revision=False."
            )
        hf_cls = getattr(transformers, spec.hf_class)

        load_kwargs: dict = {"revision": spec.revision, **kwargs}
        if dtype is not None:
            load_kwargs["dtype"] = dtype
        if spec.family == "mae":
            # Turn the pretraining-time masking off at load time.
            load_kwargs["mask_ratio"] = 0.0

        try:
            model = hf_cls.from_pretrained(spec.hf_id, **load_kwargs)
        except Exception as exc:  # noqa: BLE001
            if spec.gated:
                raise RuntimeError(
                    f"Could not load gated repo {spec.hf_id}. Accept the licence on the "
                    f"model page and export HF_TOKEN, then retry."
                ) from exc
            raise
        return cls(spec, model)

    @property
    def hf(self) -> nn.Module:
        """The underlying HF module, unwrapping PEFT if present."""
        m = self.model
        # PeftModel -> LoraModel -> original
        while hasattr(m, "base_model") and not isinstance(m, transformers.PreTrainedModel):
            m = m.base_model
        m = getattr(m, "model", m) if not isinstance(m, transformers.PreTrainedModel) else m
        return m

    @property
    def blocks(self) -> nn.ModuleList:
        return self.hf.get_submodule(self.arch.blocks)

    @property
    def n_layers(self) -> int:
        return len(self.blocks)

    @property
    def d_model(self) -> int:
        return self.spec.d_model

    def submodule(self, path: str) -> nn.Module:
        return self.hf.get_submodule(path)

    def _forward_kwargs(self, pixel_values: torch.Tensor) -> dict:
        if self.spec.family != "mae":
            return {}
        b, _, h, w = pixel_values.shape
        n_patches = (h // self.spec.patch_size) * (w // self.spec.patch_size)
        # Ascending noise => argsort is the identity => patch order preserved.
        noise = (
            torch.arange(n_patches, device=pixel_values.device, dtype=torch.float32)
            .div(n_patches)
            .expand(b, n_patches)
        )
        return {"noise": noise}

    def forward(self, pixel_values: torch.Tensor) -> BackboneOutput:
        out = self.model(pixel_values, **self._forward_kwargs(pixel_values))
        tokens = out.last_hidden_state
        r = self.spec.n_registers
        return BackboneOutput(
            tokens=tokens,
            cls=tokens[:, 0],
            registers=tokens[:, 1 : 1 + r],
            patches=tokens[:, 1 + r :],
        )

    def freeze(self) -> "VisionBackbone":
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        return self

    def unfreeze(self) -> "VisionBackbone":
        for p in self.model.parameters():
            p.requires_grad_(True)
        return self


def load_image_processor(key: str):
    """The matching preprocessing, pinned to the same revision."""
    from transformers import AutoImageProcessor

    spec = get_spec(key)
    return AutoImageProcessor.from_pretrained(spec.hf_id, revision=spec.revision)
