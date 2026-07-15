"""Evaluation.

Two entry points, mirroring the two ways activations exist:

    evaluate_store(layer_sae, ActivationStore)   # offline: metrics from saved shards
    evaluate_bank_live(bank, reader, loader)     # live: metrics across every layer

Reconstruction/dictionary/structure metrics need only activations (no labels).
Concept metrics need labels and are run by `evaluate_concepts`.
"""

from __future__ import annotations

from dataclasses import asdict

import torch

from . import metrics as M


@torch.no_grad()
def _encode_store(layer_sae, store, *, device="cuda", chunk=8192):
    """Stream a shard set through the SAE. Returns (acts, codes, recons) on CPU."""
    layer_sae.to(device)
    acts, codes, recons = [], [], []
    for shard in store.shards(mmap=True):
        for i in range(0, shard.shape[0], chunk):
            x = shard[i : i + chunk].to(device).float()
            z = layer_sae.encode(x)
            xh = layer_sae.decode(z)
            acts.append(x.cpu()); codes.append(z.cpu()); recons.append(xh.cpu())
    return torch.cat(acts), torch.cat(codes), torch.cat(recons)


def evaluate_store(
    layer_sae,
    store,
    *,
    device: str = "cuda",
    k_max: int = 64,
    chunk: int = 8192,
    with_connectivity: bool = True,
) -> dict:
    """Full SAE-quality report for one layer, from a saved ActivationStore."""
    acts, codes, recons = _encode_store(layer_sae, store, device=device, chunk=chunk)
    D = layer_sae.dictionary.detach().cpu()

    out: dict = {"site": layer_sae.site, "num_tokens": acts.shape[0], "n_concepts": D.shape[0]}
    out.update(asdict(M.reconstruction_metrics(acts, recons, codes)))
    out.update(M.dictionary_metrics(D, k_max=k_max))
    out["negative_interference"] = M.negative_interference(codes, D)
    out["ood_score"] = M.ood_score(D, acts, chunk=chunk)
    if with_connectivity:
        ztz = M.cooccurrence(codes)
        out["connectivity"] = M.connectivity(ztz)
        n_comp, sizes = M.connectivity_components(ztz)
        out["n_components"] = n_comp
        out["largest_component"] = sizes[0] if sizes else 0
    return out


@torch.no_grad()
def evaluate_bank_live(
    bank,
    reader,
    loader,
    *,
    device: str = "cuda",
    k_max: int = 64,
    max_tokens: int | None = 200_000,
) -> dict[str, dict]:
    """Evaluate every LayerSAE in the bank in a single pass over the loader.

    One backbone forward per batch feeds all sites at once (reader.read over the bank's sites). 
    Per-site codes and activations accumulate on CPU up to `max_tokens`, then metrics run per site.
    """
    reader.backbone.to(device).eval()
    bank.to(device)
    sites = bank.sites
    prefix = reader.spec.n_prefix_tokens

    acc = {s: {"acts": [], "codes": [], "recons": [], "n": 0} for s in sites}
    done = False
    for batch in loader:
        if done:
            break
        px = (batch["pixel_values"] if isinstance(batch, dict) else batch[0]).to(device)
        store = reader.read(px, sites)
        by_site = store if isinstance(store, dict) else {sites[0]: store}
        for s in sites:
            patches = by_site[s][:, prefix:, :]
            B, P, Dm = patches.shape
            x = patches.reshape(B * P, Dm)
            z = bank[s].encode(x)
            xh = bank[s].decode(z)
            acc[s]["acts"].append(x.cpu()); acc[s]["codes"].append(z.cpu()); acc[s]["recons"].append(xh.cpu())
            acc[s]["n"] += x.shape[0]
        if max_tokens is not None and min(acc[s]["n"] for s in sites) >= max_tokens:
            done = True

    report: dict[str, dict] = {}
    for s in sites:
        acts = torch.cat(acc[s]["acts"]); codes = torch.cat(acc[s]["codes"]); recons = torch.cat(acc[s]["recons"])
        D = bank[s].dictionary.detach().cpu()
        r: dict = {"site": s, "num_tokens": acts.shape[0], "n_concepts": D.shape[0]}
        r.update(asdict(M.reconstruction_metrics(acts, recons, codes)))
        r.update(M.dictionary_metrics(D, k_max=k_max))
        r["negative_interference"] = M.negative_interference(codes, D)
        r["ood_score"] = M.ood_score(D, acts)
        report[s] = r
    return report


def evaluate_concepts(
    layer_sae,
    reader,
    loader,
    site: str,
    concept_labels,
    *,
    device: str = "cuda",
    id_key: str | None = None,
    run_fms: bool = True,
    run_purity: bool = True,
    run_probes: bool = True,
) -> dict:
    """Concept-label report for one layer: encode dataset -> align labels -> FMS, purity, probes."""
    from .concepts import align_labels_to_loader, encode_dataset, fms_metrics, linear_probes, purity_metrics

    X, ids = encode_dataset(layer_sae, reader, loader, site, device=device, id_key=id_key, return_ids=True)
    Y = align_labels_to_loader(concept_labels, ids)
    names = concept_labels.names

    out: dict = {"site": site, "n_images": X.shape[0], "n_concepts": len(names)}
    if run_fms:
        out["fms"] = fms_metrics(X, Y, names)
    if run_purity:
        out["purity"] = purity_metrics(X, Y, names)
    if run_probes:
        out["probes"] = linear_probes(X, Y, names)
    return out
