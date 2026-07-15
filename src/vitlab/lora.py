"""LoRA via PEFT.

The HF model families name their Linear layers differently (`q_proj` vs `query`,
`fc1` vs `up_proj`, ...). Rather than maintain a mapping table, target
modules default to "auto": every Linear inside the transformer blocks.
"""
# TODO: actually maintain the table :)
# Remark: LoRA will provide a speedup, but for the shorter training runs I did so far it
# was not really noticeable until now / especially when we are running them over night anyway

from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn

from .backbone import VisionBackbone


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] | str = "auto"
    exclude: list[str] = field(default_factory=list)


def auto_target_modules(backbone: VisionBackbone, exclude: list[str] | None = None) -> list[str]:
    """Every distinct Linear suffix inside the transformer blocks."""
    exclude = exclude or []
    names: set[str] = set()
    for block in backbone.blocks:
        for name, mod in block.named_modules():
            if isinstance(mod, nn.Linear):
                leaf = name.split(".")[-1]
                if leaf not in exclude:
                    names.add(leaf)
    if not names:
        raise RuntimeError("No Linear layers found in blocks -- check the ArchSpec.")
    return sorted(names)


def apply_lora(backbone: VisionBackbone, cfg: LoraConfig) -> VisionBackbone:
    """Freeze the backbone and wrap the targeted Linears with LoRA adapters.

    Mutates and returns the backbone. Hook sites survive: PEFT swaps Linear
    modules in place and VisionBackbone.submodule() re-resolves by path.
    """
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model

    targets = (
        auto_target_modules(backbone, cfg.exclude)
        if cfg.target_modules == "auto" # TODO: see comment above
        else list(cfg.target_modules)
    )
    peft_cfg = PeftLoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=targets,
        bias="none",
    )
    backbone.model = get_peft_model(backbone.model, peft_cfg)
    return backbone


def trainable_parameter_summary(module: nn.Module) -> str:
    total = sum(p.numel() for p in module.parameters())
    train = sum(p.numel() for p in module.parameters() if p.requires_grad)
    pct = 100 * train / max(total, 1)
    return f"{train:,} / {total:,} trainable ({pct:.2f}%)"
