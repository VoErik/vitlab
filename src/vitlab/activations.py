"""
Sites (identical names for every family):

    embed                    embeddings output
    blocks.{i}.resid_pre     residual stream entering block i
    blocks.{i}.attn_z        per-head values, input to the attention out-proj
    blocks.{i}.attn_out      what attention *adds* to the stream
    blocks.{i}.mlp_out       what the MLP *adds* to the stream
    blocks.{i}.resid_post    residual stream leaving block i
    final_norm               output of the final layernorm
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .backbone import BackboneOutput, VisionBackbone

TokenSelect = Literal["all", "patches", "cls", "registers"]


class ActivationReader:
    """Names tensor in a backbone and reads it out.

    >>> reader = ActivationReader(backbone)
    >>> acts = reader.read(pixels, "blocks.11.resid_post")   # (B, S, D)
    >>> X = reader.collect(loader, "blocks.11.resid_post")   # (N, D)
    """

    def __init__(self, backbone: VisionBackbone):
        self.backbone = backbone
        self.spec = backbone.spec
        self._paths = self._build_sites()

    def _build_sites(self) -> dict[str, tuple[str, str]]:
        a = self.backbone.arch
        sites: dict[str, tuple[str, str]] = {
            "embed": (a.embeddings, "output"),
            "final_norm": (a.final_norm, "output"),
        }
        for i in range(self.backbone.n_layers):
            blk = f"{a.blocks}.{i}"
            sites[f"blocks.{i}.resid_pre"] = (blk, "input")
            sites[f"blocks.{i}.resid_post"] = (blk, "output")
            sites[f"blocks.{i}.attn_out"] = (f"{blk}.{a.attn_out}", "output")
            sites[f"blocks.{i}.mlp_out"] = (f"{blk}.{a.mlp_out}", "output")
            sites[f"blocks.{i}.attn_z"] = (f"{blk}.{a.out_proj}", "input")
        return sites

    @property
    def sites(self) -> list[str]:
        return list(self._paths)

    def site_path(self, site: str) -> tuple[str, str]:
        if site not in self._paths:
            raise KeyError(f"unknown site {site!r}")
        return self._paths[site]

    def match(self, pattern: str) -> list[str]:
        if pattern in self._paths:
            return [pattern]
        return [s for s in self.sites if re.search(pattern, s)]

    @contextmanager
    def _capture(self, sites: list[str], store: dict[str, torch.Tensor]):
        handles = []
        for site in sites:
            path, kind = self._paths[site]
            module: nn.Module = self.backbone.submodule(path)

            if kind == "output":

                def fwd(_m, _a, out, _site=site):
                    t = out[0] if isinstance(out, tuple) else out
                    store[_site] = t.detach()

                handles.append(module.register_forward_hook(fwd))
            else:

                def pre(_m, args, kwargs, _site=site):
                    t = args[0] if args else next(iter(kwargs.values()))
                    store[_site] = t.detach()

                handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @torch.no_grad()
    def read(
        self,
        pixel_values: torch.Tensor,
        sites: str | Iterable[str],
        *,
        return_output: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[BackboneOutput, dict[str, torch.Tensor]]:
        wanted = self.match(sites) if isinstance(sites, str) else list(sites)
        store: dict[str, torch.Tensor] = {}
        with self._capture(wanted, store):
            out = self.backbone(pixel_values)
        if return_output:
            return out, store
        return store if len(wanted) > 1 else store[wanted[0]]

    def select(self, acts: torch.Tensor, mode: TokenSelect = "patches") -> torch.Tensor:
        """(B, S, D) -> (N, D), dropping CLS/registers as asked."""
        n = self.spec.n_prefix_tokens
        r = self.spec.n_registers
        if mode == "cls":
            return acts[:, 0]
        if mode == "registers":
            return acts[:, 1 : 1 + r].reshape(-1, acts.shape[-1])
        if mode == "patches":
            return acts[:, n:].reshape(-1, acts.shape[-1])
        return acts.reshape(-1, acts.shape[-1])

    @torch.no_grad()
    def collect(
        self,
        loader: DataLoader,
        site: str,
        *,
        token_select: TokenSelect = "patches",
        max_tokens: int | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """(N, d_model) activation matrix"""
        self.backbone.eval()
        chunks: list[torch.Tensor] = []
        total = 0
        for batch in loader:
            px = batch["pixel_values"] if isinstance(batch, dict) else batch[0]
            acts = self.read(px.to(device), site)
            toks = self.select(acts, token_select).to(dtype).cpu()
            if max_tokens is not None and total + toks.shape[0] > max_tokens:
                toks = toks[: max_tokens - total]
            chunks.append(toks)
            total += toks.shape[0]
            if max_tokens is not None and total >= max_tokens:
                break
        return torch.cat(chunks) if chunks else torch.empty(0, self.spec.d_model)
    

@dataclass
class ShardManifest:
    model_key: str
    site: str
    token_select: str
    d_model: int
    dtype: str
    num_tokens: int
    num_shards: int
    shard_files: list[str]
    tokens_per_image: int | None  # None for token_select="all" (varies)


class ActivationStore:
    """Loads activations saved by `extract_activations` back as a TensorDataset."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        manifest = json.loads((self.dir / "manifest.json").read_text())
        self.manifest = ShardManifest(**manifest)

    @property
    def num_tokens(self) -> int:
        return self.manifest.num_tokens

    def shards(self, *, mmap: bool = True) -> Iterator[torch.Tensor]:
        for name in self.manifest.shard_files:
            yield torch.load(self.dir / name, map_location="cpu", mmap=mmap)

    def tensor(self, *, mmap: bool = True) -> torch.Tensor:
        """All tokens as one (N, D) tensor. With mmap=True the concatenation is
        materialised in RAM; for very large sets prefer iterating `shards()` or
        wrapping shard-wise datasets."""
        parts = list(self.shards(mmap=mmap))
        return torch.cat(parts) if len(parts) > 1 else parts[0]

    def dataset(self, *, mmap: bool = True) -> TensorDataset:
        return TensorDataset(self.tensor(mmap=mmap))


@torch.no_grad()
def extract_activations(
    reader,
    loader: DataLoader,
    site: str,
    out_dir: str | Path,
    *,
    token_select: TokenSelect = "patches",
    max_tokens: int | None = None,
    shard_size: int = 500_000,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
    flush_every_batch: bool = False,
) -> ActivationStore:
    """Stream activations at `site` to disk in shards, then return a loader.

    Args:
        reader: a vitlab.ActivationReader over the backbone.
        site:   e.g. "blocks.11.resid_post". One site per call -- for several sites
                call this once each.
        token_select: "patches" (default) | "cls" | "registers" | "all".
        max_tokens: stop after roughly this many tokens; None = whole dataset.
        shard_size: tokens per file.
        dtype: fp16 by default.

    Reload with ActivationStore(out_dir).dataset() -> TensorDataset.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reader.backbone.to(device).eval()

    spec = reader.spec
    prefix = spec.n_prefix_tokens
    has_cls = spec.has_cls
    tokens_per_image = {
        "cls": 1 if has_cls else 0,
        "registers": spec.n_registers,
        "patches": None,   # filled from the first batch
        "all": None,
    }[token_select]

    buffer: list[torch.Tensor] = []
    held = 0
    total = 0
    shard_files: list[str] = []

    def flush() -> None:
        nonlocal buffer, held
        if not buffer:
            return
        shard = torch.cat(buffer)
        name = f"shard_{len(shard_files):05d}.pt"
        torch.save(shard, out_dir / name)
        shard_files.append(name)
        buffer = []
        held = 0

    for batch in loader:
        px = batch["pixel_values"] if isinstance(batch, dict) else batch[0]
        acts = reader.read(px.to(device), site)
        toks = reader.select(acts, token_select)
        toks = toks.to(dtype).cpu()

        if tokens_per_image is None and token_select == "patches":
            tokens_per_image = acts.shape[1] - prefix

        if max_tokens is not None and total + toks.shape[0] > max_tokens:
            toks = toks[: max_tokens - total]

        buffer.append(toks)
        held += toks.shape[0]
        total += toks.shape[0]

        if flush_every_batch or held >= shard_size:
            flush()

        if max_tokens is not None and total >= max_tokens:
            break

    flush()

    manifest = ShardManifest(
        model_key=spec.key,
        site=site,
        token_select=token_select,
        d_model=spec.d_model,
        dtype=str(dtype).replace("torch.", ""),
        num_tokens=total,
        num_shards=len(shard_files),
        shard_files=shard_files,
        tokens_per_image=tokens_per_image,
    )
    (out_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2) + "\n")
    print(f"wrote {total:,} tokens -> {out_dir}  ({len(shard_files)} shards, {manifest.dtype})")
    return ActivationStore(out_dir)
