from __future__ import annotations

import json
from pathlib import Path

import torch


def save_sae(
    sae,
    normalizer,
    directory: str | Path,
    *,
    site: str = "",
    model_key: str = "",
    dataset: str = "",
    token_select: str = "",
    num_train_tokens: int = 0,
    top_k: int | None = None,
    logs: dict | None = None,
    extra: dict | None = None,
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    torch.save(sae, directory / "sae.pt")
    normalizer.save(directory / "normalizer.pt")

    meta = {
        "site": site,
        "model_key": model_key,
        "sae_class": type(sae).__name__,
        "normalizer": normalizer.kind,
        "dataset": dataset,
        "token_select": token_select,
        "num_train_tokens": num_train_tokens,
    }
    if top_k is not None:
        meta["top_k"] = top_k
    elif hasattr(sae, "top_k"):
        meta["top_k"] = int(getattr(sae, "top_k"))
    if hasattr(sae, "get_dictionary"):
        try:
            D = sae.get_dictionary()
            meta["nb_concepts"], meta["d_model"] = int(D.shape[0]), int(D.shape[1])
        except Exception:  # noqa: BLE001
            pass
    if extra:
        meta.update(extra)
    (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    if logs is not None:
        clean = {k: [float(v) for v in vals] for k, vals in logs.items()}
        (directory / "logs.json").write_text(json.dumps(clean, indent=2) + "\n")
    return directory


def load_layer_sae(directory: str | Path, device: str = "cpu"):
    """Load one trained SAE as a ready-to-use LayerSAE (SAE + normalizer)."""
    from ..eval.sae_bank import LayerSAE

    return LayerSAE.load(directory, device=device)


def load_bank(path_map: dict[str, str | Path], device: str = "cpu"):
    """{site: directory} -> SAEBank."""
    from ..eval.sae_bank import SAEBank

    return SAEBank.load(path_map, device=device)


def discover_bank(root: str | Path, device: str = "cpu"):
    """Build a bank from a directory tree, keying each SAE by the `site` in its
    meta.json. Any subdirectory containing sae.pt is treated as one SAE."""
    from ..eval.sae_bank import LayerSAE, SAEBank

    root = Path(root)
    saes = {}
    for sae_file in sorted(root.rglob("sae.pt")):
        d = sae_file.parent
        meta = {}
        if (d / "meta.json").exists():
            meta = json.loads((d / "meta.json").read_text())
        site = meta.get("site") or d.name
        saes[site] = LayerSAE.load(d, device=device)
    if not saes:
        raise FileNotFoundError(f"no sae.pt found under {root}")
    return SAEBank(saes)