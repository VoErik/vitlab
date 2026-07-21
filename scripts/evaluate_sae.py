from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import vitlab
from vitlab.eval import evaluate_concepts, evaluate_store, load_concepts
from vitlab import ActivationStore, seed_everything
from vitlab.sae import discover_bank, load_layer_sae
from vitlab.utils import setup_logger


def _acts_dir_for(site: str, acts: Path | None, acts_root: Path | None) -> Path:
    """Resolve the activation store for a site: explicit --acts wins, else
    <acts_root>/<site> (the layout extract_all_layers.sh writes)."""
    if acts is not None:
        return acts
    if acts_root is not None:
        cand = acts_root / site.replace("/", "_")
        if cand.exists():
            return cand
        return acts_root / site
    raise SystemExit(f"need --acts or --acts-root to evaluate dictionary metrics for {site}")


def eval_one(layer_sae, site, args, concepts, backbone_model, log) -> dict:
    result: dict = {"site": site, "spec": layer_sae.spec.summary()}

    if "dictionary" in args.aspects:
        store = ActivationStore(_acts_dir_for(site, args.acts, args.acts_root))
        log.info(f"[{site}] dictionary metrics over {store.manifest.num_tokens:,} tokens")
        result["dictionary"] = evaluate_store(
            layer_sae, store, device=args.device, k_max=args.k_max,
            with_connectivity=not args.no_connectivity,
        )

    if "concepts" in args.aspects:
        if backbone_model is None or concepts is None:
            raise SystemExit("--aspects concepts needs --backbone and --dataset")
        from vitlab.activations import ActivationReader
        from vitlab.batching import make_loader
        from vitlab.datasets import get_splits
        from vitlab.eval import reconstruct_image_ids

        reader = ActivationReader(backbone_model.backbone)

        split_map = {"train": 0, "val": 1, "validation": 1, "test": 2}
        split_idx = split_map.get(args.concept_split, 0)
        split_ds = get_splits(args.dataset, model_key=backbone_model.spec.key,
                              cast_labels=False)[split_idx]

        ids = reconstruct_image_ids(split_ds, args.concept_split, label_key=args.concept_label_col_override)

        def _concept_collate(rows):
            return {"pixel_values": torch.stack([r["pixel_values"] for r in rows])}

        loader = make_loader(split_ds, args.batch_size, shuffle=False,
                             num_workers=args.workers, collate_fn=_concept_collate)

        log.info(f"[{site}] concept metrics over {len(concepts.names)} concepts "
                 f"(split '{args.concept_split}', {len(ids)} images, exact id match)")
        result["concepts"] = evaluate_concepts(
            layer_sae, reader, loader, site, concepts, device=args.device,
            image_ids=ids,
        )
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sae", type=Path, help="one trained SAE directory")
    src.add_argument("--bank", type=Path, help="a tree of SAE directories (one per layer)")

    p.add_argument("--aspects", nargs="+", default=["dictionary"],
                   choices=["dictionary", "concepts"])
    p.add_argument("--acts", type=Path, default=None, help="activation store (single SAE)")
    p.add_argument("--acts-root", type=Path, default=None, help="root of per-site stores (bank)")
    p.add_argument("--backbone", type=Path, default=None, help="trained model dir (concepts)")
    p.add_argument("--dataset", default=None, help="dataset name (concepts)")
    p.add_argument("--concept-csv-id", default=None, help="override concept CSV id column")
    p.add_argument("--concept-split", default="train",
                   help="split to evaluate concepts on (also prefixes the ids): train/val/test")
    p.add_argument("--concept_label_col_override", default="disease", help="label column to use for path recon")
    p.add_argument("--k-max", type=int, default=64)
    p.add_argument("--no-connectivity", action="store_true", help="skip the co-activation graph")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None, help="output dir (default: alongside the SAE)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.data_root is not None:
        vitlab.set_data_root(args.data_root)
    seed_everything(args.seed)

    if args.sae is not None:
        ls = load_layer_sae(args.sae, device=args.device)
        bank = {ls.site or args.sae.name: ls}
        out_root = args.out or args.sae
    else:
        b = discover_bank(args.bank, device=args.device)
        bank = {site: b[site] for site in b.sites}
        out_root = args.out or args.bank

    log = setup_logger(out_root, name="vitlab.eval")
    log.info(f"evaluating {len(bank)} SAE(s): aspects={args.aspects}")

    backbone_model = vitlab.load_model(args.backbone, device=args.device) if args.backbone else None
    concepts = None
    if "concepts" in args.aspects and args.dataset:
        kw = {"id_column": args.concept_csv_id} if args.concept_csv_id else {}
        concepts = load_concepts(args.dataset, **kw)
        log.info(f"loaded {len(concepts.names)} concepts for {args.dataset}")

    all_results = {}
    for site, layer_sae in bank.items():
        res = eval_one(layer_sae, site, args, concepts, backbone_model, log)
        all_results[site] = res
        out_file = out_root / f"eval_{site.replace('/', '_')}.json"
        out_file.write_text(json.dumps(res, indent=2, default=float) + "\n")
        log.info(f"[{site}] -> {out_file}")

    print("\n=== summary")
    for site, r in sorted(all_results.items()):
        parts = [site]
        if "dictionary" in r:
            d = r["dictionary"]
            parts.append(f"R2={d['r2']:.3f} L0={d['l0']:.0f} dead={d['dead_fraction']*100:.1f}% "
                         f"eff_rank={d['effective_rank']:.0f} coh={d['coherence_mean']:.3f}")
        if "concepts" in r and "fms" in r["concepts"]:
            parts.append(f"FMS={r['concepts']['fms']['fms_strict_mean']:.3f}")
            if "probes" in r["concepts"] and "_mean_balanced_accuracy" in r["concepts"]["probes"]:
                parts.append(f"probe_acc={r['concepts']['probes']['_mean_balanced_accuracy']:.3f}")
        print("  " + "  ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())