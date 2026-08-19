from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import Response
from PIL import Image


def png_response(png_bytes: bytes) -> Response:
    return Response(content=png_bytes, media_type="image/png")


def softmax_probs(logits: torch.Tensor) -> list[float]:
    return F.softmax(logits[0], dim=-1).detach().cpu().tolist()


def grid_to_png(grid: np.ndarray, size: int = 224, cmap: str = "viridis", alpha_over=None) -> bytes:
    """Render a (h,w) map to a viridis PNG, optionally alpha-blended over a base image.

    Kept deliberately simple; the frontend usually renders overlays on a canvas instead,
    but this is handy for server-side thumbnails.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    g = grid.astype("float32")
    g = (g - g.min()) / (np.ptp(g) + 1e-8)
    up = F.interpolate(torch.tensor(g)[None, None], size=(size, size),
                       mode="bilinear", align_corners=False)[0, 0].numpy()
    rgba = (cm.get_cmap(cmap)(up) * 255).astype("uint8")   # matplotlib<3.11 pinned
    img = Image.fromarray(rgba, mode="RGBA")
    if alpha_over is not None:
        base = alpha_over.convert("RGBA").resize((size, size))
        img.putalpha(140)
        img = Image.alpha_composite(base, img)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()
