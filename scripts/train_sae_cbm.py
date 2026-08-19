"""
Train an SAE concept-bottleneck classifier.
# TODO: make this new version work!
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import yaml

from vitlab import seed_everything
from vitlab.sae import (
    CBMConfig,
    build_sae,
    fit_normalizer,
    save_sae,
    train_cbm_joint,
    train_cbm_separate,
    train_sae,
)
from vitlab.utils import setup_logger


def _load_embeddings(path: str):
    """Return (embeddings (N,T,D), labels (N,))."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    emb = blob.get("embeddings", blob.get("vit_tokens"))
    if emb is None:
        raise KeyError(f"{path}: expected 'embeddings' or 'vit_tokens'")
    labels = blob["labels"]
    emb = emb.float()
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels)
    return emb, labels.long()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    out_root = Path(cfg["out"])
    log = setup_logger(out_root, level_str=cfg.get("log_level", "INFO"))
    log.info(f"=== SAE-CBM ({cfg['mode']}) -> {out_root}")

    train_emb, train_lab = _load_embeddings(cfg["train_data"])
    test_emb, test_lab = (_load_embeddings(cfg["test_data"]) if cfg.get("test_data") else (None, None))
    input_dim = train_emb.shape[-1]
    log.info(f"train {tuple(train_emb.shape)}  classes {int(train_lab.max())+1}")

    cbm_cfg = CBMConfig(
        aggregation=cfg.get("aggregation", "max"),
        alpha=cfg.get("elasticnet_alpha", 1e-2),
        l1_ratio=cfg.get("elasticnet_l1_ratio", 0.5),
        clf_lr=cfg.get("classifier_lr", 1e-3),
        clf_epochs=cfg.get("classifier_epochs", 15),
    )

    seeds = cfg.get("seeds", [1, 2, 3])
    accs = []
    for seed in seeds:
        seed_everything(seed)
        log.info(f"--- seed {seed}")

        # preprocessing fit on this split (flattened tokens)
        N, T, D = train_emb.shape
        normalizer = fit_normalizer(train_emb.reshape(N * T, D), cfg.get("normalize", "zscore"))

        sae = build_sae(cfg["sae_type"], input_shape=input_dim, nb_concepts=cfg["nb_concepts"],
                        batch_size=cfg.get("sae_batch_size", 4096), device=args.device,
                        **{k: cfg[k] for k in ("top_k",) if k in cfg})

        if cfg["sae_type"] in ("TopKSAE", "BatchTopKSAE"):
            from overcomplete.sae.losses import top_k_auxiliary_loss
            criterion = top_k_auxiliary_loss
        else:
            from vitlab.sae import mse_with_l1
            from functools import partial
            criterion = partial(mse_with_l1, penalty=cfg.get("penalty", 3.0))
        sae_opt = torch.optim.Adam(sae.parameters(), lr=cfg.get("sae_lr", 3e-4))

        if cfg["mode"] == "separate":
            # phase 1: train the SAE on flattened, normalised tokens
            from torch.utils.data import DataLoader, TensorDataset
            Xn = normalizer.norm(train_emb.reshape(N * T, D))
            loader = DataLoader(TensorDataset(Xn), batch_size=cfg.get("sae_batch_size", 4096), shuffle=True)
            _, sae = train_sae(sae, loader, criterion, sae_opt,
                               nb_epochs=cfg.get("sae_epochs", 50), device=args.device, verbose=True)
            # phase 2: pool + sparse head
            head, info = train_cbm_separate(sae, train_emb, train_lab, normalizer, cbm_cfg,
                                            test_embeddings=test_emb, test_labels=test_lab, device=args.device)
        elif cfg["mode"] == "joint":
            sae, head, info = train_cbm_joint(sae, train_emb, train_lab, normalizer, criterion, sae_opt, cbm_cfg,
                                              test_embeddings=test_emb, test_labels=test_lab,
                                              batch_size=cfg.get("classifier_batchsize", 256),
                                              lambda_sae=cfg.get("sae_lambda", 1.0),
                                              lambda_clf=cfg.get("classifier_lambda", 1.0),
                                              epochs=max(cfg.get("sae_epochs", 50), cbm_cfg.clf_epochs),
                                              device=args.device)
        else:
            raise ValueError(f"unknown mode {cfg['mode']!r}")

        acc = info["test_acc"]
        accs.append(acc if acc is not None else float("nan"))
        log.info(f"seed {seed}: test_acc={acc}  head_sparsity={info['head_sparsity']:.2f}")

        run_dir = out_root / f"seed{seed}"
        save_sae(sae, normalizer, run_dir, site=cfg.get("site", ""),
                 model_key=cfg.get("backbone_name", ""), extra={"mode": cfg["mode"]})
        torch.save(head.state_dict(), run_dir / "classifier.pt")

    valid = [a for a in accs if a == a]  # drop nan
    summary = {
        "mode": cfg["mode"], "sae_type": cfg["sae_type"], "aggregation": cbm_cfg.aggregation,
        "seeds": seeds, "test_acc_per_seed": accs,
        "test_acc_mean": statistics.mean(valid) if valid else None,
        "test_acc_std": statistics.pstdev(valid) if len(valid) > 1 else 0.0,
    }
    (out_root / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info(f"done. mean test_acc={summary['test_acc_mean']} over {len(valid)} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
