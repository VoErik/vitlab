from __future__ import annotations

import torch
import torch.nn.functional as F
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from vitlab.backbone import BackboneOutput

from .. import state
from .._labels import class_names_for_task

router = APIRouter()


class LensRequest(BaseModel):
    image_token: str
    model_id: str
    task: str
    mode: str = "cls"        # "cls" (trajectory over layers) | "per_patch" (single layer)
    layer: int | None = None


def _lens_logits(head, resid, nprefix, per_patch: bool, final_norm):
    """resid: (1,S,D). per_patch -> (P, C); else pooled -> (1, C).

    `final_norm` is the backbone's final normalization module; applying it here is the
    logit-lens step that makes the last layer agree with the trained head.
    """
    resid = final_norm(resid)
    if per_patch:
        feats = resid[:, nprefix:, :]
        return head.classifier(head.pre(feats[0]))
    bo = BackboneOutput(
        tokens=resid,
        cls=resid[:, 0, :],
        registers=resid[:, 1:nprefix, :],
        patches=resid[:, nprefix:, :],
    )
    pooled = head.pooled(bo)
    return head.classifier(head.pre(pooled))


@router.post("/logit_lens")
def logit_lens(req: LensRequest):
    model = state.get_model(req.model_id)
    if req.task not in model.task_names:
        raise HTTPException(400, f"task {req.task!r} not in {model.task_names}")
    head = model.heads[req.task]
    reader = model.reader
    nprefix = model.spec.n_prefix_tokens
    final_norm = model.backbone.submodule(reader.site_path("final_norm")[0])
    n_classes = head.classifier.out_features
    names = list(class_names_for_task(req.task, n_classes))
    px = state.get_image(req.image_token).pixel_values.to(state.DEVICE)

    if req.mode == "cls":
        traj = []
        with torch.no_grad():
            for l in range(model.spec.n_layers):
                resid = reader.read(px, f"blocks.{l}.resid_post")
                logits = _lens_logits(head, resid, nprefix, per_patch=False, final_norm=final_norm)
                traj.append(F.softmax(logits[0], -1).cpu().tolist())
        return {"mode": "cls", "trajectory": traj,   # (n_layers, n_classes)
                "class_names": names,
                "note": "Each layer's residual is passed through the backbone's final norm "
                        "and the trained head; the last layer matches the real prediction."}

    if req.mode == "per_patch":
        if req.layer is None:
            raise HTTPException(400, "per_patch mode requires 'layer'")
        with torch.no_grad():
            resid = reader.read(px, f"blocks.{req.layer}.resid_post")
            logits = _lens_logits(head, resid, nprefix, per_patch=True, final_norm=final_norm)  # (P, C)
            probs = F.softmax(logits, -1)
            conf, argmax = probs.max(-1)
        P = logits.shape[0]
        side = int(round(P ** 0.5))
        return {
            "mode": "per_patch", "layer": req.layer, "side": side,
            "argmax": argmax.cpu().tolist(),           # (P,)
            "confidence": conf.cpu().tolist(),         # (P,)
            "class_names": names,
            "note": "Per-patch head read-out; interpretive, not the trained pooled path.",
        }

    raise HTTPException(400, f"unknown mode {req.mode!r} (use 'cls' or 'per_patch')")
