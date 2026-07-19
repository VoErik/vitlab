from __future__ import annotations

import torch


@torch.no_grad()
def compute_median_activations(model, bank, loader, n_layers, *, device="cuda",
                               max_batches=None, site_kind="resid_post"):
    """
    {layer: (F,)} per-feature median activation over a reference set.
    Averages patch-token codes per image, then takes the median over images.
    """
    from tqdm.auto import tqdm

    model.to(device).eval()
    reader = model.reader
    prefix = model.spec.n_prefix_tokens
    accum = {l: [] for l in range(n_layers)}

    for bi, batch in enumerate(tqdm(loader, desc="medians")):
        if max_batches and bi >= max_batches:
            break
        px = (batch["pixel_values"] if isinstance(batch, dict) else batch[0]).to(device)
        for l in range(n_layers):
            site = f"blocks.{l}.{site_kind}"
            if site not in bank.saes:
                continue
            acts = reader.read(px, site)[:, prefix:, :]
            B, P, D = acts.shape
            codes = bank[site].encode(acts.reshape(B * P, D)).reshape(B, P, -1)
            per_img = codes.mean(1)
            accum[l].append(per_img.to(torch.float16).cpu())

    medians = {}
    for l in range(n_layers):
        if not accum[l]:
            continue
        allz = torch.cat(accum[l], 0).float()
        medians[l] = allz.median(0).values.to(device)
        nz = (medians[l] > 0).sum().item()
        print(f"  layer {l}: {nz}/{medians[l].numel()} features have non-zero median")
    return medians


@torch.no_grad()
def collect_class_images(loader, target_class, n_images, *, device="cuda"):
    """Gather up to n_images of one class from a loader (for class-level circuits)."""
    out, got = [], 0
    for batch in loader:
        if isinstance(batch, dict):
            imgs, labels = batch["pixel_values"], batch.get("labels")
        else:
            imgs, labels = batch[0], (batch[1] if len(batch) > 1 else None)
        if labels is None:
            raise ValueError("loader yields no labels; cannot filter by class")
        mask = labels == target_class
        if mask.any():
            out.append(imgs[mask])
            got += int(mask.sum())
        if got >= n_images:
            break
    if not out:
        raise ValueError(f"no images of class {target_class} found")
    return list(torch.cat(out)[:n_images].unsqueeze(1).to(device))