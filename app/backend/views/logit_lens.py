"""Logit lens: project each layer's residual through the task head.

Reuses head.pooled(BackboneOutput) + head.pre + head.classifier so the lens matches
the real head exactly (pooling may be cls/mean/attn -- do NOT assume CLS). This router
is fully implemented; it's a good second reference after classify.py.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from vitlab.backbone import BackboneOutput

from .. import state

router = APIRouter()


class LensRequest(BaseModel):
    image_token: str
    model_id: str
    task: str
    mode: str = "cls"        # "cls" (trajectory over layers) | "per_patch" (single layer)
    layer: int | None = None  # required for per_patch


def _lens_logits(head, resid, nprefix, per_patch: bool):
    """resid: (1,S,D). per_patch -> (P, C); else pooled -> (1, C)."""
    if per_patch:
        feats = resid[:, nprefix:, :]                     # (1,P,D)
        return head.classifier(head.pre(feats[0]))        # (P, C)
    bo = BackboneOutput(cls=resid[:, 0, :], patches=resid[:, nprefix:, :])
    pooled = head.pooled(bo)                               # respects spec.pooling / attn-pool
    return head.classifier(head.pre(pooled))              # (1, C)


@router.post("/logit_lens")
def logit_lens(req: LensRequest):
    model = state.get_model(req.model_id)
    if req.task not in model.task_names:
        raise HTTPException(400, f"task {req.task!r} not in {model.task_names}")
    head = model.heads[req.task]
    reader = model.reader
    nprefix = model.spec.n_prefix_tokens
    px = state.get_image(req.image_token).pixel_values.to(state.DEVICE)

    if req.mode == "cls":
        traj = []
        with torch.no_grad():
            for l in range(model.spec.n_layers):
                resid = reader.read(px, f"blocks.{l}.resid_post")
                logits = _lens_logits(head, resid, nprefix, per_patch=False)
                traj.append(F.softmax(logits[0], -1).cpu().tolist())
        return {"mode": "cls", "trajectory": traj,   # (n_layers, n_classes)
                "note": "Applying the head to earlier layers is the logit-lens approximation."}

    if req.mode == "per_patch":
        if req.layer is None:
            raise HTTPException(400, "per_patch mode requires 'layer'")
        with torch.no_grad():
            resid = reader.read(px, f"blocks.{req.layer}.resid_post")
            logits = _lens_logits(head, resid, nprefix, per_patch=True)  # (P, C)
            probs = F.softmax(logits, -1)
            conf, argmax = probs.max(-1)
        P = logits.shape[0]
        side = int(round(P ** 0.5))
        return {
            "mode": "per_patch", "layer": req.layer, "side": side,
            "argmax": argmax.cpu().tolist(),           # (P,)
            "confidence": conf.cpu().tolist(),         # (P,)
            "note": "Per-patch head read-out; interpretive, not the trained pooled path.",
        }
    raise HTTPException(400, f"unknown mode {req.mode!r}")
