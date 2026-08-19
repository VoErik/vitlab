from __future__ import annotations

import torch

from vitlab.attribution.core import AttributionResult, _grid, _splice_codes


@torch.no_grad()
def contrastive_dla(model, bank, px, site, task, a, b, top_k, device):
    model.to(device).eval()
    reader = model.reader
    prefix = model.spec.n_prefix_tokens
    layer_sae = bank[site].to(device)

    acts = reader.read(px, site)[:, prefix:, :]
    B, P, D = acts.shape
    codes = layer_sae.encode(acts.reshape(B * P, D)).reshape(B, P, -1)

    W_dec = layer_sae.dictionary.to(device)                       # (F, D)
    W = model.heads[task].classifier.weight.detach().to(device)   # (C, D)
    w = W[a] - W[b]                                                # (D,)  margin direction
    per_unit = W_dec @ w                                           # (F,)

    contrib = codes[0].sum(0) * per_unit                          # (F,)
    scores, idx = torch.topk(contrib, min(top_k, contrib.numel()))
    h, wg = _grid(P)
    maps = (codes[0].T.reshape(-1, h, wg))[idx].detach().cpu()
    return AttributionResult(
        features=idx.tolist(), sites=[site] * len(idx), scores=scores.tolist(),
        method="contrastive_dla", target_class=a, spatial_maps=maps,
    )


def contrastive_patching(model, bank, px, site, task, a, b, top_k, device,
                         normalize_cross_site=True):
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    prefix = model.spec.n_prefix_tokens
    sites = [site]

    cache: dict = {}
    with _splice_codes(model, bank, sites, prefix, cache):
        out = model(px, task)[0]
        logit = out[a] - out[b]                                   # the margin
        grads = torch.autograd.grad(logit, [cache[s] for s in sites])

    candidates = []
    for s, g in zip(sites, grads):
        codes = cache[s]
        eff = (codes * g).sum(dim=(0, 1))
        if normalize_cross_site and eff.abs().max() > 0:
            eff = eff / eff.abs().max()
        P = codes.shape[1]
        h, wg = _grid(P)
        act_maps = codes[0].T.reshape(-1, h, wg).detach().cpu()
        for f in range(eff.numel()):
            candidates.append((eff[f].item(), s, f, act_maps[f]))

    candidates.sort(key=lambda c: c[0], reverse=True)
    top = candidates[:top_k]
    return AttributionResult(
        features=[c[2] for c in top], sites=[c[1] for c in top], scores=[c[0] for c in top],
        method="contrastive_patching", target_class=a,
        spatial_maps=torch.stack([c[3] for c in top]) if top else torch.empty(0),
    )