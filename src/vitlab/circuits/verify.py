from __future__ import annotations

import torch

from ..attribution.core import _resolve_task, _target_class


@torch.no_grad()
def _ablate_feature_hook(bank, site, feature, prefix, reader):
    """A forward hook that subtracts one SAE feature's contribution at `site`."""
    layer_sae = bank[site]
    W_dec = layer_sae.dictionary
    vec = W_dec[feature].detach()

    def hook(_m, _in, out):
        is_t = isinstance(out, tuple)
        x = out[0] if is_t else out
        pre, sp = x[:, :prefix], x[:, prefix:]
        B, P, D = sp.shape
        codes = layer_sae.encode(sp.reshape(B * P, D)).reshape(B, P, -1)
        act = codes[..., feature]
        delta = act.unsqueeze(-1) * vec.view(1, 1, -1)
        sp = sp - delta.to(sp.dtype)
        x_new = torch.cat([pre, sp], dim=1)
        return (x_new,) + out[1:] if is_t else x_new

    path, _ = reader.site_path(site)
    return reader.backbone.submodule(path), hook


@torch.no_grad()
def verify_nodes(model, bank, circuit, image, target_class, *, task=None, device="cuda"):
    """
    Ablate each circuit node and record the logit-margin drop on its node.

    Writes the measured drop into node.edge_importance's sibling; returns a dict
    {(layer, feature): logit_drop} and annotates each node with `.verified_drop`.
    """
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    task = _resolve_task(model, task)
    reader = model.reader
    prefix = model.spec.n_prefix_tokens
    px = image.to(device)
    cls = _target_class(model, px, task, target_class)

    def margin():
        lo = model(px, task)
        return (lo[0, cls] - lo[0].mean()).item()

    clean = margin()
    drops = {}
    for layer, nodes in circuit.nodes.items():
        site = f"blocks.{layer}.resid_post"
        if site not in bank.saes:
            continue
        for node in nodes:
            mod, hook = _ablate_feature_hook(bank, site, node.feature_idx, prefix, reader)
            h = mod.register_forward_hook(hook)
            try:
                drop = clean - margin()
            finally:
                h.remove()
            drops[(layer, node.feature_idx)] = drop
            node.verified_drop = drop
    return drops


@torch.no_grad()
def verify_edges(model, bank, circuit, image, *, task=None, device="cuda", top_edges=None):
    """
    For each edge u->d, ablate u and measure the change in d's activation.

    A real edge means removing the upstream feature moves the downstream feature.
    Returns {(u_key, d_key): delta_downstream_activation}; also stores it on the
    edge as a 4th tuple slot via a parallel dict on the circuit.
    """
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    task = _resolve_task(model, task)
    reader = model.reader
    prefix = model.spec.n_prefix_tokens
    px = image.to(device)

    def downstream_act(d_layer, d_feat):
        site = f"blocks.{d_layer}.resid_post"
        acts = reader.read(px, site)[:, prefix:, :]
        B, P, D = acts.shape
        codes = bank[site].encode(acts.reshape(B * P, D)).reshape(B, P, -1)
        return codes[..., d_feat].mean().item()

    edges = circuit.edges if top_edges is None else \
        sorted(circuit.edges, key=lambda e: -abs(e[2]))[:top_edges]

    verified = {}
    for u, d, w in edges:
        u_site = f"blocks.{u.layer}.resid_post"
        if u_site not in bank.saes:
            continue
        base = downstream_act(d.layer, d.feature_idx)
        mod, hook = _ablate_feature_hook(bank, u_site, u.feature_idx, prefix, reader)
        h = mod.register_forward_hook(hook)
        try:
            ablated = downstream_act(d.layer, d.feature_idx)
        finally:
            h.remove()
        verified[(u.key(), d.key())] = base - ablated
    circuit.verified_edges = verified
    return verified