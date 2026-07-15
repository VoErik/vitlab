from __future__ import annotations

import argparse
import sys

import torch

from .activations import ActivationReader
from .backbone import VisionBackbone
from .registry import REGISTRY, _load_revisions, _write_revisions, list_models


def cmd_pin(args) -> int:
    from huggingface_hub import HfApi

    api = HfApi()
    revs = _load_revisions()
    failed = []
    for key, spec in REGISTRY.items():
        if spec.hf_id in revs and not args.force:
            print(f"  = {key:16s} {spec.hf_id}@{revs[spec.hf_id][:12]} (kept)")
            continue
        try:
            sha = api.model_info(spec.hf_id).sha
            revs[spec.hf_id] = sha
            print(f"  + {key:16s} {spec.hf_id}@{sha[:12]}")
        except Exception as exc:  # noqa: BLE001
            failed.append((key, exc))
            print(f"  ! {key:16s} {spec.hf_id}: {type(exc).__name__} {exc}", file=sys.stderr)
    _write_revisions(revs)
    print(f"\nwrote {len(revs)} revisions")
    if failed:
        print("gated repos need `huggingface-cli login` + licence acceptance.", file=sys.stderr)
    return 0


def check_model(key: str, *, device="cpu", tol=1e-4) -> bool:
    print(f"\n=== {key}")
    backbone = VisionBackbone.from_key(key, require_pinned_revision=False)
    backbone.to(device).eval()
    reader = ActivationReader(backbone)
    spec = backbone.spec

    px = torch.randn(2, 3, spec.image_size, spec.image_size, device=device)
    out, cache = reader.read(px, reader.sites, return_output=True)

    n_patches = (spec.image_size // spec.patch_size) ** 2
    expected = spec.n_prefix_tokens + n_patches
    ok = True

    if out.tokens.shape[1] != expected:
        print(f"  ! token count {out.tokens.shape[1]} != expected {expected}")
        ok = False
    if out.patches.shape[1] != n_patches:
        print(f"  ! patch count {out.patches.shape[1]} != {n_patches}")
        ok = False

    for i in range(spec.n_layers):
        pre = cache[f"blocks.{i}.resid_pre"]
        post = cache[f"blocks.{i}.resid_post"]
        attn = cache[f"blocks.{i}.attn_out"]
        mlp = cache[f"blocks.{i}.mlp_out"]
        err = (post - (pre + attn + mlp)).abs().max().item()
        if err > tol:
            print(f"  ! block {i}: resid_post != resid_pre + attn_out + mlp_out (max err {err:.2e})")
            ok = False
        if i + 1 < spec.n_layers:
            gap = (post - cache[f"blocks.{i+1}.resid_pre"]).abs().max().item()
            if gap > tol:
                print(f"  ! block {i}: resid_post != resid_pre of block {i+1} (max err {gap:.2e})")
                ok = False

    z = cache[f"blocks.0.attn_z"]
    if z.shape[-1] != spec.d_model:
        print(f"  ! attn_z width {z.shape[-1]} != d_model {spec.d_model}")
        ok = False

    # determinism (catches the MAE shuffle)
    out2, _ = reader.read(px, ["final_norm"], return_output=True)
    if not torch.allclose(out.tokens, out2.tokens, atol=1e-5):
        print("  ! forward pass is not deterministic")
        ok = False

    print(f"  {'OK' if ok else 'FAILED'}: {spec.n_layers} blocks, {len(reader.sites)} sites, "
          f"{expected} tokens ({spec.n_registers} registers)")
    return ok


def cmd_doctor(args) -> int:
    keys = args.models or list_models()
    results = {}
    for key in keys:
        try:
            results[key] = check_model(key, device=args.device)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! could not load: {type(exc).__name__}: {exc}")
            results[key] = False
    print("\n" + "=" * 40)
    for k, v in results.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    return 0 if all(results.values()) else 1


def main() -> int:
    p = argparse.ArgumentParser(prog="vitlab")
    sub = p.add_subparsers(dest="cmd", required=True)

    pin = sub.add_parser("pin", help="resolve and record HF commit SHAs")
    pin.add_argument("--force", action="store_true")
    pin.set_defaults(func=cmd_pin)

    doc = sub.add_parser("doctor", help="verify hook sites against real weights")
    doc.add_argument("models", nargs="*")
    doc.add_argument("--device", default="cpu")
    doc.set_defaults(func=cmd_doctor)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())