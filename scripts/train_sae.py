"""Train an SAE on extracted activations.

    uv run python scripts/train_sae.py --config configs/sae.yaml
    uv run python scripts/train_sae.py --config configs/sae.yaml --acts acts/dinov2-base/fitzpatrick17k/blocks.10.resid_post

The config is a YAML; anything on the command line overrides it. A minimal config:

    acts: acts/dinov2-base/fitzpatrick17k/blocks.10.resid_post
    site: blocks.10.resid_post
    model_key: dinov2-base
    sae_type: TopKSAE
    nb_concepts: 16384
    top_k: 32
    normalize: zscore
    epochs: 50
    lr: 3e-4
    batch_size: 4096
    out: saes/dinov2_fitz_l10
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from vitlab.eval import intrinsic_dimensionality_spectral, local_id_mle
from vitlab import ActivationStore, seed_everything
from vitlab.sae import (
    MatryoshkaLossWrapper,
    build_sae,
    fit_normalizer,
    gated_sae_loss,
    mse_with_l1,
    pytorch_kmeans,
    reanimation,
    save_sae,
    train_sae,
)
from vitlab.utils import setup_logger


def load_activations(store_dir: Path, max_tokens: int | None, log) -> torch.Tensor:
    """(N, D) activation matrix from a sharded ActivationStore."""
    store = ActivationStore(store_dir)
    log.info(f"activation store: {store.manifest.num_tokens:,} tokens "
             f"from {store.manifest.model_key} @ {store.manifest.site} "
             f"({store.manifest.num_shards} shards, {store.manifest.dtype})")
    X = store.tensor().float()
    if max_tokens is not None and X.shape[0] > max_tokens:
        X = X[:max_tokens]
        log.info(f"capped to {max_tokens:,} tokens")
    return X


def build_criterion(cfg: dict, log):
    """Pick the loss. TopK/BatchTopK use overcomplete's auxiliary loss; the others
    use the ported custom losses."""
    sae_type = cfg["sae_type"]
    penalty = cfg.get("penalty", 3.0)
    if sae_type in ("TopKSAE", "BatchTopKSAE"):
        from overcomplete.sae.losses import top_k_auxiliary_loss
        log.info("loss: top_k_auxiliary_loss")
        crit = top_k_auxiliary_loss
    elif sae_type in ("SAE", "JumpSAE"):
        log.info(f"loss: mse_with_l1 (penalty={penalty})")
        crit = partial(mse_with_l1, penalty=penalty, reanim_dead_codes=cfg.get("use_reanimation", True))
    elif sae_type == "GatedSAE":
        log.info(f"loss: gated_sae_loss (penalty={penalty})")
        crit = partial(gated_sae_loss, penalty=penalty)
    else:
        log.info("loss: reanimation (mse + dead-code revival)")
        crit = reanimation
    if cfg.get("as_matryoshka"):
        dims = tuple(cfg.get("matryoshka_dims", (64, 128, 512, 1024)))
        log.info(f"wrapping loss in Matryoshka {dims}")
        crit = MatryoshkaLossWrapper(crit, nested_dims=dims)
    return crit


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--acts", type=Path, default=None, help="override cfg['acts']")
    p.add_argument("--out", type=Path, default=None, help="override cfg['out']")
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="train one SAE per layer, substituting {L} in acts/site/out")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.acts:
        cfg["acts"] = str(args.acts)
    if args.out:
        cfg["out"] = str(args.out)

    layers = args.layers if args.layers else [None]
    for layer in layers:
        run_cfg = dict(cfg)
        if layer is not None:
            for key in ("acts", "site", "out"):
                if key in run_cfg and "{L}" in str(run_cfg[key]):
                    run_cfg[key] = str(run_cfg[key]).replace("{L}", str(layer))
            run_cfg.setdefault("site", f"blocks.{layer}.resid_post")
        _train_one(run_cfg, args.device, args.seed)
    return 0


def _train_one(cfg: dict, device: str, seed: int) -> None:
    seed_everything(seed)
    out_dir = Path(cfg["out"])
    log = setup_logger(out_dir, level_str=cfg.get("log_level", "INFO"))
    log.info(f"=== training {cfg['sae_type']} -> {out_dir}")
    (out_dir / "config.yaml").write_text(yaml.dump(cfg))

    X = load_activations(Path(cfg["acts"]), cfg.get("num_samples"), log)
    input_dim = X.shape[1]

    normalize = cfg.get("normalize", "zscore")
    normalizer = fit_normalizer(X, normalize)
    Xn = normalizer.norm(X)
    log.info(f"normalize={normalizer.kind}  mean-norm {X.norm(dim=1).mean():.3f} -> {Xn.norm(dim=1).mean():.3f}")

    if cfg.get("report_intrinsic_dim", True):
        gid = intrinsic_dimensionality_spectral(Xn, max_rows=20_000)
        log.info("intrinsic dim | " + ", ".join(f"{k}: {v:.2f}" for k, v in gid.items()))
        if cfg.get("report_local_id", False):
            lid = local_id_mle(Xn, sample_size=cfg.get("lid_sample", 10_000), k=cfg.get("lid_k", 20))
            log.info(f"local id | mean {lid['local_id_mean']:.2f} median {lid['local_id_median']:.2f}")

    batch_size = cfg.get("batch_size", 4096)
    loader = DataLoader(TensorDataset(Xn), batch_size=batch_size, shuffle=True)

    sae = build_sae(
        cfg["sae_type"],
        input_shape=input_dim,
        nb_concepts=cfg["nb_concepts"],
        batch_size=batch_size,
        device=device,
        **{k: cfg[k] for k in ("top_k",) if k in cfg},
    )

    if cfg.get("as_archetypal"):
        from overcomplete.sae.archetypal_dictionary import RelaxedArchetypalDictionary
        log.info(f"archetypal: {cfg.get('num_centroids', 1024)} k-means centroids")
        points = pytorch_kmeans(Xn, cfg.get("num_centroids", 1024), device=device)
        sae.dictionary = RelaxedArchetypalDictionary(
            in_dimensions=input_dim, nb_concepts=cfg["nb_concepts"],
            points=points.to(device), delta=cfg.get("archetypal_delta", 1.0), device=device,
        )

    criterion = build_criterion(cfg, log)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.get("lr", 3e-4))
    scheduler = None
    if cfg.get("lr_scheduler", "cosine") == "cosine":
        total = len(loader) * cfg.get("epochs", 50)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg.get("lr", 3e-4), total_steps=total,
            pct_start=cfg.get("warmup_ratio", 0.05),
        )

    log.info("training...")
    logs, best = train_sae(
        sae, loader, criterion, opt, scheduler=scheduler,
        nb_epochs=cfg.get("epochs", 50), device=device,
        target_l0=cfg.get("target_l0"),
    )

    store_manifest = ActivationStore(Path(cfg["acts"])).manifest
    save_sae(best, normalizer, out_dir,
             site=cfg.get("site", ""), model_key=cfg.get("model_key", ""), logs=logs,
             dataset=cfg.get("acts", "unknown/unknown").split("/")[-2],
             token_select=store_manifest.token_select,
             num_train_tokens=int(X.shape[0]),
             top_k=cfg.get("top_k"),
             extra={"sae_type": cfg["sae_type"], "normalize": normalizer.kind})
    log.info(f"saved -> {out_dir}  (final R2 {logs['r2'][-1]:.4f}, dead {logs['dead_features'][-1]*100:.1f}%)")


if __name__ == "__main__":
    raise SystemExit(main())