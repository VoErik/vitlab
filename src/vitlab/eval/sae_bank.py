"""Wrap an overcomplete SAE with the preprocessing it was trained under."""
# TODO: move to vitlab/sae submoddule and fix paths

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F

NormKind = Literal[
    "none", "zscore", "scale", "global_rms", "global_quantile", "l2norm", "instance_norm"
]
# TODO: remove in future iteration
_ALIASES = {
    "elementwise_standardize": "zscore",
    "global_mean_norm": "scale",
    "layernorm": "instance_norm",
}

_INVERTIBLE = {"none", "zscore", "scale", "global_rms", "global_quantile"}


@dataclass
class Normalizer:
    kind: NormKind = "zscore"
    mean: torch.Tensor | None = None   # (D,)  zscore
    std: torch.Tensor | None = None    # (D,)  zscore
    scale: torch.Tensor | None = None  # scalar: scale/global_rms/global_quantile
    eps: float = 1e-6

    def __post_init__(self):
        self.kind = _ALIASES.get(self.kind, self.kind)

    def to(self, device) -> "Normalizer":
        t = lambda v: None if v is None else v.to(device)  # noqa: E731
        return Normalizer(self.kind, t(self.mean), t(self.std), t(self.scale), self.eps)

    @property
    def invertible(self) -> bool:
        return self.kind in _INVERTIBLE

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        k = self.kind
        if k == "none":
            return x
        if k == "zscore":
            return (x - self.mean) / (self.std + self.eps)
        if k in ("scale", "global_rms", "global_quantile"):
            return x / (self.scale + self.eps)
        if k == "l2norm":
            return x / (x.norm(dim=-1, keepdim=True) + self.eps) * (x.shape[-1] ** 0.5)
        if k == "instance_norm":
            m = x.mean(-1, keepdim=True)
            s = x.std(-1, keepdim=True)
            return (x - m) / (s + self.eps)
        raise ValueError(k)

    def denorm(self, x_norm: torch.Tensor) -> torch.Tensor:
        k = self.kind
        if k in ("none", "l2norm", "instance_norm"):
            return x_norm
        if k == "zscore":
            return x_norm * (self.std + self.eps) + self.mean
        if k in ("scale", "global_rms", "global_quantile"):
            return x_norm * (self.scale + self.eps)
        raise ValueError(k)

    def save(self, path: str | Path) -> None:
        tensors = {k: getattr(self, k) for k in ("mean", "std", "scale") if getattr(self, k) is not None}
        torch.save({"meta": {"kind": self.kind, "eps": self.eps}, "tensors": tensors}, path)

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if "meta" not in blob:
            if "mean" in blob and "std" in blob:
                return cls(kind="zscore", mean=blob["mean"], std=blob["std"])
            for key, kind in (("mean_norm", "scale"), ("rms_norm", "global_rms"),
                              ("quantile_0.9_norm", "global_quantile")):
                if key in blob:
                    return cls(kind=kind, scale=blob[key])
            return cls(kind="none")
        return cls(**blob["meta"], **blob["tensors"])


def fit_normalizer(activations: torch.Tensor, kind: NormKind = "zscore", *, eps: float = 1e-6) -> Normalizer:
    """Fit a Normalizer over an (N, D) activation matrix."""
    kind = _ALIASES.get(kind, kind)
    with torch.no_grad():
        if kind in ("none", "l2norm", "instance_norm"):
            return Normalizer(kind, eps=eps)
        if kind == "zscore":
            return Normalizer("zscore", mean=activations.mean(0, keepdim=True),
                              std=activations.std(0, keepdim=True).clamp_min(eps), eps=eps)
        if kind == "scale":
            return Normalizer("scale", scale=activations.norm(dim=-1).mean(), eps=eps)
        if kind == "global_rms":
            return Normalizer("global_rms", scale=(activations.pow(2).sum(-1).mean()).sqrt(), eps=eps)
        if kind == "global_quantile":
            return Normalizer("global_quantile", scale=activations.norm(dim=-1).quantile(0.9), eps=eps)
    raise ValueError(kind)


@dataclass(frozen=True)
class SAESpec:
    """Everything needed to identify and reproduce a trained SAE."""

    site: str = ""
    model_key: str = ""
    sae_class: str = ""         
    nb_concepts: int = 0
    d_model: int = 0
    top_k: int | None = None
    normalizer: str = "none"

    dataset: str = ""
    token_select: str = ""
    num_train_tokens: int = 0

    final_r2: float | None = None
    final_dead_frac: float | None = None
    epochs: int | None = None

    extra: dict = field(default_factory=dict)

    @property
    def expansion(self) -> float:
        """Dictionary size relative to input width."""
        return self.nb_concepts / self.d_model if self.d_model else 0.0

    @property
    def concept_sample_ratio(self) -> float | None:
        """nb_concepts / num_train_tokens."""
        if not self.num_train_tokens:
            return None
        return self.nb_concepts / self.num_train_tokens

    def summary(self) -> str:
        r2 = f"{self.final_r2:.3f}" if self.final_r2 is not None else "?"
        return (f"{self.sae_class}({self.nb_concepts}c/{self.d_model}d, "
                f"k={self.top_k}, {self.normalizer}) @ {self.model_key}:{self.site} "
                f"R2={r2}")

    @classmethod
    def from_meta(cls, meta: dict, logs: dict | None = None) -> "SAESpec":
        """Build from a saved meta.json."""
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        fields_in = {k: v for k, v in meta.items() if k in known}
        extra = {k: v for k, v in meta.items() if k not in known}
        if logs:
            if logs.get("r2"):
                fields_in.setdefault("final_r2", float(logs["r2"][-1]))
            if logs.get("dead_features"):
                fields_in.setdefault("final_dead_frac", float(logs["dead_features"][-1]))
            fields_in.setdefault("epochs", len(logs.get("r2", [])) or None)
        return cls(**fields_in, extra=extra)

    def to_meta(self) -> dict:
        """Flat dict for meta.json."""
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "extra"}
        d.update(self.extra)
        return d


class LayerSAE:
    """One SAE at one site, plus its Normalizer."""

    def __init__(self, sae, normalizer: Normalizer, *, site: str = "",
                 spec: "SAESpec | None" = None, device="cpu"):
        self.sae = sae.to(device).eval()
        self.normalizer = normalizer.to(device)
        self.spec = spec or SAESpec(site=site, normalizer=normalizer.kind)
        self.site = self.spec.site or site
        self.device = device

    def to(self, device) -> "LayerSAE":
        self.sae.to(device)
        self.normalizer = self.normalizer.to(device)
        self.device = device
        return self

    @property
    def dictionary(self) -> torch.Tensor:
        """(n_concepts, D) decoder atoms."""
        return self.sae.get_dictionary()

    @property
    def n_concepts(self) -> int:
        return self.dictionary.shape[0]

    def encode(self, raw: torch.Tensor) -> torch.Tensor:
        x = self.normalizer.norm(raw.to(self.device))
        out = self.sae.encode(x)
        return out[1] if isinstance(out, tuple) else out

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        x_hat_norm = self.sae.decode(codes.to(self.device))
        return self.normalizer.denorm(x_hat_norm)

    @torch.no_grad()
    def forward(self, raw: torch.Tensor):
        raw = raw.to(self.device)
        codes = self.encode(raw)
        raw_hat = self.decode(codes)
        if not self.normalizer.invertible:
            # decode() stayed in normalised space; compare there, not against raw
            error = self.normalizer.norm(raw) - self.sae.decode(codes)
        else:
            error = raw - raw_hat
        return codes, raw_hat, error

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.sae, directory / "sae.pt")
        self.normalizer.save(directory / "normalizer.pt")
        (directory / "meta.json").write_text(json.dumps({"site": self.site}) + "\n")
        return directory

    @classmethod
    def load(cls, directory: str | Path, device="cpu") -> "LayerSAE":
        directory = Path(directory)
        sae = torch.load(directory / "sae.pt", map_location=device, weights_only=False)
        if (directory / "normalizer.pt").exists():
            normalizer = Normalizer.load(directory / "normalizer.pt")
        elif (directory / "activation_stats.pt").exists(): # TODO: backwards compatibility, remove in future iteration
            normalizer = Normalizer.load(directory / "activation_stats.pt")
        else:
            normalizer = Normalizer("none")
        site = ""
        spec = None
        if (directory / "meta.json").exists():
            meta = json.loads((directory / "meta.json").read_text())
            site = meta.get("site", "")
            logs = None
            if (directory / "logs.json").exists():
                logs = json.loads((directory / "logs.json").read_text())
            spec = SAESpec.from_meta(meta, logs)
        return cls(sae, normalizer, site=site, spec=spec, device=device)


class SAEBank:
    """A set of LayerSAEs keyed by site name (e.g. "blocks.5.resid_post")."""

    def __init__(self, saes: dict[str, LayerSAE]):
        self.saes = saes

    def __getitem__(self, site: str) -> LayerSAE:
        return self.saes[site]

    def __iter__(self):
        return iter(self.saes)

    def __len__(self) -> int:
        return len(self.saes)

    @property
    def sites(self) -> list[str]:
        return list(self.saes)

    def to(self, device) -> "SAEBank":
        for s in self.saes.values():
            s.to(device)
        return self

    @torch.no_grad()
    def encode_all(self, raw_by_site: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """{site: raw acts} -> {site: codes}, for sites present in both."""
        return {site: self.saes[site].encode(x) for site, x in raw_by_site.items() if site in self.saes}

    @classmethod
    def load(cls, path_map: dict[str, str | Path], device="cpu") -> "SAEBank":
        """path_map: {site: directory}. Each dir holds sae.pt + normalizer.pt."""
        return cls({site: LayerSAE.load(p, device=device) for site, p in path_map.items()})
