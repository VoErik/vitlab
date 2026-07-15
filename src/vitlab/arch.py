"""
HuggingFace unfortunately does not standardize naming. So we need to do it here per model.

Site names (identical for every model):

    embed                    embeddings module output
    blocks.{i}.resid_pre     residual stream entering block i
    blocks.{i}.attn_z        per-head values, input to the attention out-proj
    blocks.{i}.attn_out      what the attention block *adds* to the stream
    blocks.{i}.mlp_out       what the MLP block *adds* to the stream
    blocks.{i}.resid_post    residual stream leaving block i
    final_norm               output of the final layernorm

Because attn_out/mlp_out are defined as the tensors added to the stream
(post-LayerScale where a family has one), the decomposition

    resid_post = resid_pre + attn_out + mlp_out

holds exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

from .registry import Family


@dataclass(frozen=True)
class ArchSpec:
    blocks: str        # path to the ModuleList of transformer blocks
    attn: str          # path to attention module, relative to a block
    attn_out: str      # module whose output is added to the residual stream
    mlp_out: str       # module whose output is added to the residual stream
    out_proj: str      # attention output projection (its *input* is attn_z)
    final_norm: str    # path from model root
    embeddings: str    # path from model root


ARCHS: dict[Family, ArchSpec] = {
    # transformers v5 ViT: flat `layers`, fused-name projections, ViTMLP.
    "vit": ArchSpec(
        blocks="layers",
        attn="attention",
        attn_out="attention",
        mlp_out="mlp",
        out_proj="attention.o_proj",
        final_norm="layernorm",
        embeddings="embeddings",
    ),
    # MAE shares the v5 ViT layout.
    "mae": ArchSpec(
        blocks="layers",
        attn="attention",
        attn_out="attention",
        mlp_out="mlp",
        out_proj="attention.o_proj",
        final_norm="layernorm",
        embeddings="embeddings",
    ),
    # DINOv2 kept the old `encoder.layer` layout and has LayerScale on both
    # branches -- so the residual contribution is the *post*-LayerScale tensor.
    "dinov2": ArchSpec(
        blocks="encoder.layer",
        attn="attention",
        attn_out="layer_scale1",
        mlp_out="layer_scale2",
        out_proj="attention.output.dense",
        final_norm="layernorm",
        embeddings="embeddings",
    ),
    # DINOv3: `model.layer`, RoPE passed through as extra forward args,
    # LayerScale as in v2, and o_proj naming as in v5 ViT.
    "dinov3": ArchSpec(
        blocks="model.layer",
        attn="attention",
        attn_out="layer_scale1",
        mlp_out="layer_scale2",
        out_proj="attention.o_proj",
        final_norm="norm",
        embeddings="embeddings",
    ),
    "clip": ArchSpec(
        blocks="encoder.layers",
        attn="self_attn",
        attn_out="self_attn",
        mlp_out="mlp",
        out_proj="self_attn.out_proj",
        final_norm="post_layernorm",
        embeddings="embeddings",
    ),
}


def get_arch(family: Family) -> ArchSpec:
    if family not in ARCHS:
        raise KeyError(f"No ArchSpec for family {family!r}")
    return ARCHS[family]
