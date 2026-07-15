from __future__ import annotations

from overcomplete.sae import (
    SAE,
    BatchTopKSAE,
    JumpSAE,
    MpSAE,
    OMPSAE,
    RAJumpSAE,
    RATopKSAE,
    TopKSAE,
)

from .gated import GatedSAE

SAE_CLASSES = {
    "TopKSAE": TopKSAE,
    "BatchTopKSAE": BatchTopKSAE,
    "JumpSAE": JumpSAE,
    "MpSAE": MpSAE,
    "OMPSAE": OMPSAE,
    "RAJumpSAE": RAJumpSAE,
    "RATopKSAE": RATopKSAE,
    "SAE": SAE,
    "GatedSAE": GatedSAE,
}


def build_sae(sae_type: str, *, batch_size: int = 4096, **kwargs):
    if sae_type not in SAE_CLASSES:
        raise ValueError(f"Unknown SAE type {sae_type!r}. Available: {list(SAE_CLASSES)}")
    if sae_type == "BatchTopKSAE" and "top_k" in kwargs:
        # BatchTopK selects top_k over the whole batch, not per sample.
        kwargs["top_k"] = kwargs["top_k"] * batch_size
    return SAE_CLASSES[sae_type](**kwargs)
