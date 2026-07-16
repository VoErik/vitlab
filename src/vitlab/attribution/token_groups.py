from __future__ import annotations

from dataclasses import dataclass

import torch

from .core import _resolve_task, _target_class


@dataclass
class TokenGroupShare:
    """Per-layer decomposition of the decision across token groups."""

    layers: list[int]
    groups: list[str] # ["cls", "registers", "patches"]
    gradient_mass: dict[str, list[float]]
    ablation_drop: dict[str, list[float]]
    target_class: int

    def table(self) -> str:
        lines = ["layer  " + "  ".join(f"{g:>10}" for g in self.groups) + "   (grad mass | abl drop)"]
        for i, l in enumerate(self.layers):
            gm = "  ".join(f"{self.gradient_mass[g][i]:.2f}".rjust(10) for g in self.groups)
            ab = "  ".join(f"{self.ablation_drop[g][i]:+.3f}".rjust(10) for g in self.groups)
            lines.append(f"{l:>5}  {gm}    {ab}")
        return "\n".join(lines)


def _group_slices(spec):
    """{name: slice} over the token axis, from the spec's prefix layout."""
    n_reg = spec.n_registers
    groups = {"cls": slice(0, 1)}
    if n_reg > 0:
        groups["registers"] = slice(1, 1 + n_reg)
    groups["patches"] = slice(1 + n_reg, None)
    return groups


def token_group_attribution(
    model,
    pixel_values,
    *,
    task: str | None = None,
    target_class_idx: int | None = None,
    layers: list[int] | None = None,
    device: str = "cuda",
    site_kind: str = "resid_post",
) -> TokenGroupShare:
    """Per-layer share of the decision across CLS / registers / patches."""
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    task = _resolve_task(model, task)
    px = pixel_values.to(device)
    cls_idx = _target_class(model, px, task, target_class_idx)

    reader = model.reader
    spec = model.spec
    if layers is None:
        layers = list(range(spec.n_layers))
    sites = [f"blocks.{l}.{site_kind}" for l in layers]
    groups = _group_slices(spec)
    gnames = list(groups)

    # grad mass
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(site):
        def hook(_m, _in, out):
            is_tuple = isinstance(out, tuple)
            x = out[0] if is_tuple else out
            x = x.clone().requires_grad_(True)
            captured[site] = x
            return (x,) + out[1:] if is_tuple else x
        return hook

    for site in sites:
        path, _ = reader.site_path(site)
        handles.append(reader.backbone.submodule(path).register_forward_hook(make_hook(site)))
    try:
        logit = model(px, task)[0, cls_idx]
        grads = torch.autograd.grad(logit, [captured[s] for s in sites])
    finally:
        for h in handles:
            h.remove()

    grad_mass = {g: [] for g in gnames}
    for site, g in zip(sites, grads):
        x = captured[site]
        # per-token attribution mass = |sum_d x_d * grad_d|
        tok_mass = (x * g).sum(-1).abs()[0] # (S,)
        total = tok_mass.sum().clamp_min(1e-12)
        for name, sl in groups.items():
            grad_mass[name].append((tok_mass[sl].sum() / total).item())

    # mean ablate within group (for CLS mean ablate over all tokens) TODO: might be weird cuz of norms
    with torch.no_grad():
        clean = model(px, task)[0, cls_idx].item()
    abl = {g: [] for g in gnames}
    for site in sites:
        for name, sl in groups.items():
            n_tok = len(range(*sl.indices(reader.spec.n_prefix_tokens + 10_000)))
            drop = _ablate_group(model, reader, px, site, sl, task, cls_idx, clean, n_tokens=n_tok)
            abl[name].append(drop)

    return TokenGroupShare(layers=layers, groups=gnames, gradient_mass=grad_mass,
                           ablation_drop=abl, target_class=cls_idx)


@torch.no_grad()
def _ablate_group(model, reader, px, site, token_slice, task, cls_idx, clean, *, n_tokens):
    """Ablate a token group's residual at one site, return the logit drop."""
    def hook(_m, _in, out):
        is_tuple = isinstance(out, tuple)
        x = out[0] if is_tuple else out
        x = x.clone()
        grp = x[:, token_slice]
        if grp.shape[1] > 1:
            x[:, token_slice] = grp.mean(dim=1, keepdim=True)
        else:
            # single token: replace with the mean over all tokens
            x[:, token_slice] = x.mean(dim=1, keepdim=True)
        return (x,) + out[1:] if is_tuple else x

    path, _ = reader.site_path(site)
    hnd = reader.backbone.submodule(path).register_forward_hook(hook)
    try:
        new = model(px, task)[0, cls_idx].item()
    finally:
        hnd.remove()
    return clean - new