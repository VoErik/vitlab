from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from .backbone import BackboneOutput, VisionBackbone
from .heads import TaskHead, TaskSpec
from .activations import ActivationReader
from .lora import LoraConfig, apply_lora
from .registry import get_spec

BackboneMode = Literal["frozen", "full", "lora"]


@dataclass
class ModelConfig:
    model_key: str
    tasks: list[TaskSpec]
    backbone_mode: BackboneMode = "lora"
    lora: LoraConfig | None = None

    def __post_init__(self):
        if self.backbone_mode == "lora" and self.lora is None:
            self.lora = LoraConfig()
        self.tasks = [t if isinstance(t, TaskSpec) else TaskSpec(**t) for t in self.tasks]


class MultiTaskViT(nn.Module):
    """One backbone, N heads."""

    def __init__(self, cfg: ModelConfig, backbone: VisionBackbone):
        super().__init__()
        self.cfg = cfg
        self.spec = backbone.spec

        if cfg.backbone_mode == "frozen":
            backbone.freeze()
        elif cfg.backbone_mode == "lora":
            apply_lora(backbone, cfg.lora)

        self.encoder = backbone
        self.heads = nn.ModuleDict(
            {t.name: TaskHead(t, self.spec.d_model) for t in cfg.tasks}
        )

    @classmethod
    def from_config(cls, cfg: ModelConfig, **load_kwargs) -> "MultiTaskViT":
        backbone = VisionBackbone.from_key(cfg.model_key, **load_kwargs)
        return cls(cfg, backbone)

    @property
    def backbone(self) -> VisionBackbone:
        return self.encoder

    @property
    def reader(self) -> ActivationReader:
        return ActivationReader(self.encoder)

    @property
    def task_names(self) -> list[str]:
        return list(self.heads)

    def features(self, pixel_values: torch.Tensor) -> BackboneOutput:
        return self.encoder(pixel_values)

    def forward(
        self,
        pixel_values: torch.Tensor,
        task: str,
        labels: torch.Tensor | None = None,
    ):
        feats = self.features(pixel_values)
        head: TaskHead = self.heads[task]
        logits = head(feats)
        if labels is None:
            return logits
        return logits, head.loss(logits, labels)

    def forward_all(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.features(pixel_values)
        return {name: head(feats) for name, head in self.heads.items()}

    def param_groups(self, backbone_lr: float, head_lr: float, weight_decay: float = 0.05):

        def split(named):
            decay, no_decay = [], []
            for n, p in named:
                if not p.requires_grad:
                    continue
                (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)
            return decay, no_decay

        bd, bn = split(self.encoder.named_parameters())

        # Split head weights by whether their own penalty already carries the L2.
        plain_w, elastic_w, head_b = [], [], []
        for name, head in self.heads.items():
            w, b = split(head.named_parameters())
            (elastic_w if head.spec.alpha > 0 else plain_w).extend(w)
            head_b.extend(b)

        groups = [
            {"name": "backbone_decay", "params": bd, "lr": backbone_lr, "weight_decay": weight_decay},
            {"name": "backbone_nodecay", "params": bn, "lr": backbone_lr, "weight_decay": 0.0},
            {"name": "head_weights", "params": plain_w, "lr": head_lr, "weight_decay": weight_decay},
            {"name": "head_weights_elastic", "params": elastic_w, "lr": head_lr, "weight_decay": 0.0},
            {"name": "head_bias", "params": head_b, "lr": head_lr, "weight_decay": 0.0},
        ]
        return [g for g in groups if g["params"]]


def save_model(model: MultiTaskViT, path: str | Path) -> Path:
    """Writes config.json + heads.safetensors (+ adapter/ or backbone.safetensors)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    cfg = model.cfg
    payload = {
        "model_key": cfg.model_key,
        "hf_id": model.spec.hf_id,
        "revision": model.spec.revision,
        "backbone_mode": cfg.backbone_mode,
        "lora": asdict(cfg.lora) if cfg.lora else None,
        "tasks": [asdict(t) for t in cfg.tasks],
        "format": 1,
    }
    (path / "config.json").write_text(json.dumps(payload, indent=2) + "\n")

    save_file(
        {k: v.contiguous() for k, v in model.heads.state_dict().items()},
        path / "heads.safetensors",
    )

    if cfg.backbone_mode == "lora":
        model.backbone.model.save_pretrained(str(path / "adapter"))
    elif cfg.backbone_mode == "full":
        save_file(
            {k: v.contiguous() for k, v in model.backbone.model.state_dict().items()},
            path / "backbone.safetensors",
        )
    return path


def load_model(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    eval_mode: bool = True,
) -> MultiTaskViT:
    """Rebuild a trained model from a directory."""
    path = Path(path)
    payload = json.loads((path / "config.json").read_text())

    spec = get_spec(payload["model_key"])
    if payload.get("revision") and spec.revision and payload["revision"] != spec.revision:
        raise RuntimeError(
            f"Checkpoint was trained on {payload['hf_id']}@{payload['revision']} but the "
            f"registry now pins {spec.revision}."
        )

    tasks = [TaskSpec(**t) for t in payload["tasks"]]
    lora = LoraConfig(**payload["lora"]) if payload.get("lora") else None
    mode: BackboneMode = payload["backbone_mode"]

    backbone = VisionBackbone.from_key(payload["model_key"])
    cfg = ModelConfig(payload["model_key"], tasks, backbone_mode="frozen", lora=None)
    model = MultiTaskViT(cfg, backbone)
    model.cfg = ModelConfig(payload["model_key"], tasks, backbone_mode=mode, lora=lora)

    if mode == "lora":
        from peft import PeftModel

        model.backbone.model = PeftModel.from_pretrained(
            model.backbone.model, str(path / "adapter"), is_trainable=not eval_mode
        )
    elif mode == "full":
        model.backbone.model.load_state_dict(load_file(path / "backbone.safetensors"))

    model.heads.load_state_dict(load_file(path / "heads.safetensors"))
    model.to(device)
    if eval_mode:
        model.eval()
    return model
