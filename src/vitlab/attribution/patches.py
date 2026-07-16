from __future__ import annotations

import torch

from .core import _ablate_one, _resolve_task, _target_class


def patch_to_token(row: int, col: int, grid) -> int:
    """(row, col) -> flat token index, row-major. grid is int (square) or (H, W)."""
    gh, gw = (grid, grid) if isinstance(grid, int) else grid
    if not (0 <= row < gh and 0 <= col < gw):
        raise ValueError(f"({row},{col}) out of range for grid {gh}x{gw}")
    return row * gw + col


def token_to_patch(token: int, grid) -> tuple[int, int]:
    gw = grid if isinstance(grid, int) else grid[1]
    return token // gw, token % gw


def tokens_in_region(top_left, bottom_right, grid) -> list[int]:
    """All flat token indices in a rectangular patch region (corners inclusive)."""
    gw = grid if isinstance(grid, int) else grid[1]
    r0, c0 = top_left
    r1, c1 = bottom_right
    return [r * gw + c for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def ablate_feature_at_patches(model, bank, pixel_values, site, feature, patch_positions,
                              *, grid=None, task=None, target_class_idx=None, device="cuda"):
    """
    Logit drop from ablating one concept at specific patches. 
    
    `patch_positions` accepts flat indices, or (row, col) tuples if `grid` is given.
    """
    model.to(device).eval()
    task = _resolve_task(model, task)
    px = pixel_values.to(device)
    cls = _target_class(model, px, task, target_class_idx)
    prefix = model.spec.n_prefix_tokens

    def flat(p):
        if isinstance(p, tuple):
            if grid is None:
                raise ValueError("grid required for (row, col) positions")
            return patch_to_token(p[0], p[1], grid)
        return int(p)

    positions = [flat(patch_positions)] if isinstance(patch_positions, (int, tuple)) \
        else [flat(p) for p in patch_positions]

    with torch.no_grad():
        clean = model(px, task)[0, cls].item()
    return _ablate_one(model, bank, px, site, feature, cls, task, prefix, clean,
                       ablate_positions=positions)


def scan_feature_across_patches(model, bank, pixel_values, site, feature, grid, *,
                                patch_subset=None, task=None, target_class_idx=None,
                                device="cuda", verbose=False):
    """
    Ablate one concept at each patch individually -> (H, W) map of logit drops.
    """
    gh, gw = (grid, grid) if isinstance(grid, int) else grid
    n = gh * gw
    drops = torch.zeros(n)

    model.to(device).eval()
    task = _resolve_task(model, task)
    px = pixel_values.to(device)
    cls = _target_class(model, px, task, target_class_idx)
    prefix = model.spec.n_prefix_tokens
    with torch.no_grad():
        clean = model(px, task)[0, cls].item()

    positions = patch_subset if patch_subset is not None else range(n)
    for i, pos in enumerate(positions):
        if verbose and i % 50 == 0:
            print(f"  patch {i}/{len(list(positions)) if patch_subset else n}")
        drops[pos] = _ablate_one(model, bank, px, site, feature, cls, task, prefix, clean,
                                 ablate_positions=[pos])
    return drops.reshape(gh, gw)