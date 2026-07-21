from __future__ import annotations

import torch

from . import metrics as M


@torch.no_grad()
def _encode_store(layer_sae, store, *, device="cuda", chunk=8192):
    """Stream a shard set through the SAE. Returns (acts, acts_norm, codes, recons,
    recons_norm) on CPU - both raw and normalized-space tensors.
    """
    layer_sae.to(device)
    acts, acts_n, codes, recons, recons_n = [], [], [], [], []
    for shard in store.shards(mmap=True):
        for i in range(0, shard.shape[0], chunk):
            x = shard[i : i + chunk].to(device).float()
            xn = layer_sae.normalizer.norm(x)
            z = layer_sae.encode(x)
            xh = layer_sae.decode(z)
            xhn = layer_sae.normalizer.norm(xh)
            acts.append(x.cpu()); acts_n.append(xn.cpu()); codes.append(z.cpu())
            recons.append(xh.cpu()); recons_n.append(xhn.cpu())
    return (torch.cat(acts), torch.cat(acts_n), torch.cat(codes),
            torch.cat(recons), torch.cat(recons_n))


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
    acts, acts_n, codes, recons, recons_n = _encode_store(layer_sae, store, device=device, chunk=chunk)
    D = layer_sae.dictionary.detach().cpu()

    out: dict = {"site": layer_sae.site, "num_tokens": acts.shape[0], "n_concepts": D.shape[0]}

    out["r2"] = M.r2_score(acts_n, recons_n)
    out["r2_raw"] = M.r2_score(acts, recons)
    out["fvu"] = M.fvu(acts_n, recons_n)
    out["l0"] = M.l0(codes)

    n_dead = int((~(codes.abs() > 1e-6).any(dim=0)).sum().item())
    out["dead_fraction"] = n_dead / D.shape[0]
    out["n_dead"] = n_dead
    out["dead_measured_over_tokens"] = int(acts.shape[0])

    out.update(M.dictionary_metrics(D, k_max=k_max))
    out["negative_interference"] = M.negative_interference(codes, D)
    out["ood_score"] = M.ood_score(D, acts_n, chunk=chunk)
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
    max_tokens: int | None = 1_000_000,
) -> dict[str, dict]:
    """
    Evaluate every LayerSAE in the bank in a single pass over the loader.
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
        norm = bank[s].normalizer
        acts_n, recons_n = norm.norm(acts), norm.norm(recons)
        D = bank[s].dictionary.detach().cpu()
        r: dict = {"site": s, "num_tokens": acts.shape[0], "n_concepts": D.shape[0]}
        r["r2"] = M.r2_score(acts_n, recons_n)         # normalized: matches training
        r["r2_raw"] = M.r2_score(acts, recons)
        r["fvu"] = M.fvu(acts_n, recons_n)
        r["l0"] = M.l0(codes)
        n_dead = int((~(codes.abs() > 1e-6).any(dim=0)).sum().item())
        r["dead_fraction"] = n_dead / D.shape[0]
        r["n_dead"] = n_dead
        r["dead_measured_over_tokens"] = int(acts.shape[0])
        r.update(M.dictionary_metrics(D, k_max=k_max))
        r["negative_interference"] = M.negative_interference(codes, D)
        r["ood_score"] = M.ood_score(D, acts_n)
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
    image_ids: list[str] | None = None,
    run_fms: bool = True,
    run_purity: bool = True,
    run_probes: bool = True,
) -> dict:
    """Concept-label report for one layer: encode dataset -> align labels -> FMS,
    purity, probes.
    """
    from .concepts import align_labels_to_loader, encode_dataset, fms_metrics, linear_probes, purity_metrics

    if image_ids is not None:
        X = encode_dataset(layer_sae, reader, loader, site, device=device, return_ids=False)
        if len(image_ids) != X.shape[0]:
            raise ValueError(
                f"image_ids has {len(image_ids)} entries but encoded {X.shape[0]} images; "
                f"the loader must be in dataset order (shuffle=False) and cover every row."
            )
        ids = image_ids
    else:
        X, ids = encode_dataset(layer_sae, reader, loader, site, device=device,
                                id_key=id_key, return_ids=True)
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


@torch.no_grad()
def collect_codes(
    layer_sae,
    store,
    *,
    device: str = "cuda",
    chunk: int = 16384,
    max_tokens: int | None = None,
    keep_acts_sample: int = 0,
):
    """Stream a store through the SAE and return the codes for the WHOLE set (or up
    to `max_tokens`), without ever holding all activations in memory at once.
    """
    layer_sae.to(device)
    code_chunks: list[torch.Tensor] = []
    acts_s: list[torch.Tensor] = []
    recons_s: list[torch.Tensor] = []
    total = 0
    for shard in store.shards(mmap=True):
        for i in range(0, shard.shape[0], chunk):
            x = shard[i : i + chunk].to(device).float()
            z = layer_sae.encode(x)
            code_chunks.append(z.cpu())
            if keep_acts_sample and total < keep_acts_sample:
                take = min(keep_acts_sample - total, x.shape[0])
                acts_s.append(x[:take].cpu())
                recons_s.append(layer_sae.decode(z[:take]).cpu())
            total += x.shape[0]
            if max_tokens is not None and total >= max_tokens:
                break
        if max_tokens is not None and total >= max_tokens:
            break

    codes = torch.cat(code_chunks)
    if max_tokens is not None and codes.shape[0] > max_tokens:
        codes = codes[:max_tokens]
    if keep_acts_sample:
        return codes, torch.cat(acts_s), torch.cat(recons_s)
    return codes