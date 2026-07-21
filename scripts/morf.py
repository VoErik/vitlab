import os
import argparse
import math
import operator
import warnings
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

from datasets import ClassLabel

import lxt.explicit.rules as lxt_rules
import lxt.explicit.functional as lxt_func
from lxt.explicit.core import Composite

import matplotlib.pyplot as plt
import numpy as np

from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
    ScoreCAM,
    LayerCAM,
    EigenCAM,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from zennit.composites import LayerMapComposite
from zennit.rules import Gamma
from zennit.image import imgify

import vitlab
from vitlab.datasets import get_splits
from vitlab.sae import discover_bank
from vitlab.eval.sae_bank import SAEBank
from vitlab.attribution import (
    direct_logit_attribution,
    attribution_patching,
    attribute,
)

RoutingMode = str

def _sae_relevance(model, bank, image, sites, target_class_idx, k, routing, task, device):
    if routing == "dla":
        res = direct_logit_attribution(model, bank, image, sites[-1], task=task,
                                       target_class_idx=target_class_idx, device=device, top_k=k)
    elif routing == "free":
        res = attribute(model, bank, image, sites, task=task,
                        target_class_idx=target_class_idx, device=device, top_k=k)
    else:  # attribution patching
        res = attribution_patching(model, bank, image, sites, task=task,
                                   target_class_idx=target_class_idx, device=device, top_k=k)
    return {"spatial_maps": res.spatial_maps, "scores": torch.tensor(res.scores)}

plt.style.use("thesis.mplstyle")

warnings.filterwarnings("ignore", message="This functionality is not yet fully tested")
warnings.filterwarnings("ignore", message="Some functions have been replaced by tracing")

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str,  default="1")
parser.add_argument("--run_diagnostics", action="store_true")
parser.add_argument("--model", type=str,  default="dinov3")
parser.add_argument("--dataset", type=str,  default="dermamnist")
parser.add_argument("--run", type=str,  required=True, help="vitlab checkpoint dir")
parser.add_argument("--sae-root", type=str,  required=True, help="tree of per-layer SAEs")
parser.add_argument(
    "--routing",
    type=str,
    default="free",
    choices=["dla", "free", "attribution_patching"],
    help=(
        "SAE attribution routing mode."
    ),
)

def test_gradient_leakage(model, img, target_class_idx, layer_idx=10):
    num_prefix = model.n_prefix
    """TEST 1: Tests if prefix tokens absorb the majority of the backward pass."""
    model.eval()
    target_block = model.blocks[layer_idx].norm1

    saved_grads = {}

    def hook_bwd(module, grad_in, grad_out):
        saved_grads["grad"] = grad_out[0]

    h = target_block.register_full_backward_hook(hook_bwd)
    model.zero_grad()

    out    = model(img)
    logits = out.logits if hasattr(out, "logits") else out
    logits[0, target_class_idx].backward()
    h.remove()

    grads = saved_grads["grad"][0]
    cls_norm     = grads[0, :].norm().item()
    reg_norm     = grads[1:num_prefix, :].norm(dim=-1).sum().item()
    spatial_norm = grads[num_prefix:, :].norm(dim=-1).sum().item()
    total        = cls_norm + reg_norm + spatial_norm

    print("\n=== 1. GRADIENT LEAKAGE TEST ===")
    print(f"CLS Token Magnitude:             {cls_norm:.4f}")
    print(f"Register Tokens (Summed) Mag:    {reg_norm:.4f}")
    print(f"All 196 Spatial Tokens (Summed): {spatial_norm:.4f}")
    print(f"--> % of Gradient in Prefix:     {((cls_norm + reg_norm) / total) * 100:.2f}%")

def test_gradcam_blindness(model, img, target_class_idx, layer_idx=10):
    num_prefix = model.n_prefix
    """TEST 2: Tests if GradCAM's spatial reshape discards the highest causal weights."""
    model.eval()
    target_block = model.blocks[layer_idx].norm1

    saved = {}

    def hook_bwd(module, grad_in, grad_out):
        saved["grad"] = grad_out[0]

    h = target_block.register_full_backward_hook(hook_bwd)
    model.zero_grad()

    out = model(img)
    (out.logits if hasattr(out, "logits") else out)[0, target_class_idx].backward()
    h.remove()

    grads           = saved["grad"][0]
    spatial_weights = grads[num_prefix:, :].mean(dim=0)
    prefix_weights  = grads[:num_prefix, :].mean(dim=0)

    print("\n=== 2. GRADCAM BLINDNESS TEST ===")
    print(f"Max Channel Weight (Spatial): {spatial_weights.abs().max().item():.6f}")
    print(f"Max Channel Weight (Prefix):  {prefix_weights.abs().max().item():.6f}")
    if prefix_weights.abs().max() > spatial_weights.abs().max():
        print("--> GradCAM is blind to the highest causal weights.")

def test_information_routing(model, img, target_class_idx, layer_idx=10):
    num_prefix = model.n_prefix
    """TEST 3: Self-repair test."""
    model.eval()
    target_block = (
        model.blocks[layer_idx]
    )

    with torch.no_grad():
        out_clean  = model(img)
        prob_clean = torch.softmax(
            out_clean.logits if hasattr(out_clean, "logits") else out_clean, dim=-1
        )[0, target_class_idx].item()

    def hook_kill_summary(module, inp, out):
        acts = out[0].clone() if isinstance(out, tuple) else out.clone()
        acts[:, 1:num_prefix, :] = 0.0 if num_prefix > 1 else acts[:, 0:1, :]
        return (acts,) + out[1:] if isinstance(out, tuple) else acts

    def hook_kill_spatial(module, inp, out):
        acts = out[0].clone() if isinstance(out, tuple) else out.clone()
        acts[:, num_prefix:, :] = 0.0
        return (acts,) + out[1:] if isinstance(out, tuple) else acts

    def hook_kill_both(module, inp, out):
        acts = out[0].clone() if isinstance(out, tuple) else out.clone()
        acts[:, 1:, :] = 0.0 if num_prefix > 1 else acts
        return (acts,) + out[1:] if isinstance(out, tuple) else acts

    def hook_kill_cls_only(module, inp, out):
        acts = out[0].clone() if isinstance(out, tuple) else out.clone()
        acts[:, 0, :] = 0.0
        return (acts,) + out[1:] if isinstance(out, tuple) else acts

    def hook_kill_all_prefix(module, inp, out):
        acts = out[0].clone() if isinstance(out, tuple) else out.clone()
        acts[:, :num_prefix, :] = 0.0
        return (acts,) + out[1:] if isinstance(out, tuple) else acts

    def _prob(hook_fn):
        h = target_block.register_forward_hook(hook_fn)
        with torch.no_grad():
            out = model(img)
        h.remove()
        return torch.softmax(
            out.logits if hasattr(out, "logits") else out, dim=-1
        )[0, target_class_idx].item()

    prob_nosum = _prob(hook_kill_summary)
    prob_nospat = _prob(hook_kill_spatial)
    prob_both = _prob(hook_kill_both)

    print("\n=== 3. SELF-REPAIR TEST ===")
    print(f"Original Confidence: {prob_clean:.4f}")

    if num_prefix > 1:
        prob_nocls  = _prob(hook_kill_cls_only)
        prob_noall  = _prob(hook_kill_all_prefix)
        print(f"Confidence w/o CLS: {prob_nocls:.4f}")
        print(f"Confidence w/o Registers: {prob_nosum:.4f}")
        print(f"Confidence w/o ALL Prefix: {prob_noall:.4f}")
    else:
        print(f"Confidence w/o CLS: {prob_nosum:.4f}")

    print(f"Confidence w/o Spatial: {prob_nospat:.4f}")
    print(f"Confidence w/o Both (sanity): {prob_both:.4f}")

def run_diagnostics(model, dataloader, device, layer_idx=10, majority_class_idx: int = 5):
    print("\n" + "=" * 50)
    print("RUNNING ARCHITECTURE DIAGNOSTICS")
    print("=" * 50)
    model.eval()

    img, target_class_idx = None, None
    MAJORITY_CLASS_IDX = majority_class_idx

    for images, labels in dataloader:
        for i in range(len(images)):
            if labels[i].item() == MAJORITY_CLASS_IDX:
                continue
            candidate = images[i : i + 1].to(device)
            with torch.no_grad():
                out    = model(candidate)
                logits = out.logits if hasattr(out, "logits") else out
                pred   = logits.argmax(dim=-1).item()
                prob   = torch.softmax(logits, dim=-1)[0, pred].item()
            if prob > 0.5:
                img, target_class_idx = candidate, pred
                break
        if img is not None:
            break

    if img is None:
        print("Could not find a valid image.")
        return

    print(f"Target Class: {target_class_idx} | Layer: {layer_idx} | Prefix: {model.spec.num_prefix}")
    test_gradient_leakage(model, img, target_class_idx, layer_idx)
    test_gradcam_blindness(model, img, target_class_idx, layer_idx)
    test_information_routing(model, img, target_class_idx, layer_idx)
    print("\n" + "=" * 50)

def get_attn_lrp_attr(model, img, device, gamma_lin=0.25, gamma_conv=0.25):
    img.requires_grad_(True)
    if img.grad is not None:
        img.grad.zero_()
    model.eval()

    zennit_composite = LayerMapComposite([
        (nn.Linear, Gamma(gamma_lin)),
        (nn.Conv2d,  Gamma(gamma_conv)),
    ])
    layer_map = {
        torch.matmul: lxt_func.matmul,
        F.softmax:    lxt_func.softmax,
        operator.add: lxt_func.add2,
        torch.add:    lxt_func.add2,
        nn.LayerNorm: lxt_rules.IdentityRule,
        nn.GELU:      lxt_rules.IdentityRule,
        nn.Dropout:   lxt_rules.IdentityRule,
    }
    dummy_input = {"x": torch.randn(1, 3, 224, 224).to(device)}
    composite   = Composite(layer_map=layer_map, zennit_composite=zennit_composite)
    composite.register(model, dummy_inputs=dummy_input)

    output       = model(img)
    target_class = output.argmax(dim=1).item()
    output[0, target_class].backward()

    heatmap = img.grad.sum(dim=1)
    heatmap *= output[0, target_class].abs() / (heatmap.sum() + 1e-8)
    composite.remove()
    return heatmap

def vit_reshape_transform(tensor, num_prefix_tokens=5):
    spatial_tokens = tensor.size(1) - num_prefix_tokens
    grid_size      = int(math.sqrt(spatial_tokens))
    if grid_size * grid_size != spatial_tokens:
        raise ValueError(
            f"Spatial tokens ({spatial_tokens}) is not a perfect square. "
            f"Check num_prefix_tokens!"
        )
    result = tensor[:, num_prefix_tokens:, :]
    result = result.reshape(tensor.size(0), grid_size, grid_size, tensor.size(2))
    return result.permute(0, 3, 1, 2)

def get_vit_cam(model, img, device, layer=10, variant="eigencam",
                aug_smooth=True, prefix_tokens=4):
    model.eval()
    img_tensor = img.to(device)
    output = model(img_tensor)
    target_class = output.argmax(dim=1).item()
    targets = [ClassifierOutputTarget(target_class)]

    target_layers = [model.blocks[layer].norm1]

    _MAP = {
        "gradcam":    GradCAM,
        "gradcam++":  GradCAMPlusPlus,
        "scorecam":   ScoreCAM,
        "layercam":   LayerCAM,
        "eigencam":   EigenCAM,
    }
    cam_cls = _MAP.get(variant.lower())
    if cam_cls is None:
        raise ValueError(f"Unknown CAM variant: {variant}")

    cam = cam_cls(
        model=model,
        target_layers=target_layers,
        reshape_transform=partial(vit_reshape_transform,
                                  num_prefix_tokens=prefix_tokens),
    )
    if variant.lower() in ("gradcam", "gradcam++"):
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets,
                            aug_smooth=aug_smooth)[0, :]
    else:
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]

    return torch.tensor(grayscale_cam).unsqueeze(0)

def show(img, heatmap_lrp, heatmap_gradcam, heatmap_sae):
    img_np = img.detach().cpu()[0].permute(1, 2, 0)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    hm_lrp = imgify(heatmap_lrp.detach().cpu().numpy(), symmetric=False)
    hm_gc  = imgify(heatmap_gradcam.detach().cpu().numpy(), symmetric=False)
    hm_sae = imgify(heatmap_sae.detach().cpu().numpy(), symmetric=False)

    fig, axes = plt.subplots(1, 4, figsize=(8, 4))
    for ax, hm, title in zip(
        axes,
        [img_np, hm_lrp, hm_gc, hm_sae],
        ["Original", "Attention LRP", "GradCAM", "SAE Attribution"],
    ):
        ax.imshow(hm)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

def heatmap_to_patches(heatmap, patch_size=16):
    B, H, W = heatmap.shape
    h_p, w_p = H // patch_size, W // patch_size
    return (
        heatmap.view(B, h_p, patch_size, w_p, patch_size)
        .sum(dim=(2, 4))
        .view(B, -1)
    )

def topk_patches(patch_relevance, k=5, largest=True):
    values, indices = torch.topk(patch_relevance[0], k, largest=largest)
    return indices.detach().cpu().numpy().astype(int), values

def get_dataset_mean(dataloader, device="cuda"):
    print("Calculating dataset mean...")
    mean = torch.zeros(3).to(device)
    count = 0
    for images, _ in dataloader:
        images = images.to(device)
        mean += images.mean(dim=(2, 3)).sum(dim=0)
        count += images.size(0)
    return (mean / count).view(3, 1, 1)

def plot_patch_grid(patch_relevance, grid_size=14):
    grid = patch_relevance.view(grid_size, grid_size).detach().cpu()
    plt.imshow(grid / (grid.max() + 1e-8), cmap="hot")
    plt.colorbar()
    plt.axis("off")
    plt.show()

def create_sae_heatmap(
    sae_results: dict, k: int, target_shape: Tuple[int, int], device: str = "cuda"
):
    all_maps   = sae_results["spatial_maps"]
    all_scores = sae_results["scores"]
    if len(all_maps) == 0:
        return torch.zeros(target_shape, device=device)
    k = min(k, all_maps.shape[0])
    weights = all_scores[:k].view(-1, 1, 1).to(all_maps.device)
    combined = (all_maps[:k] * weights).sum(0).unsqueeze(0).unsqueeze(0).float()
    return F.interpolate(combined, size=target_shape, mode="bicubic",
                         align_corners=False).squeeze(0).squeeze(0)

def impute_road_diffusion(img, mask, noise_std=0.1, steps=20):
    mean_val = img.mean(dim=(2, 3), keepdim=True)
    img_filled = img * mask + mean_val * (1 - mask)

    k = torch.tensor([
        [1/12, 1/6, 1/12],
        [1/6,  0.0, 1/6],
        [1/12, 1/6, 1/12],
    ], device=img.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    for _ in range(steps):
        smoothed   = F.conv2d(img_filled, k, padding=1, groups=3)
        img_filled = img * mask + smoothed * (1 - mask)

    noise = torch.randn_like(img) * noise_std
    img_filled = img_filled * mask + (img_filled + noise) * (1 - mask)
    return img_filled

def plot_ablation_sanity_check(
    img, indices_dict, patch_size, device,
    save_path="sanity_check.jpg",
    plot_pcts=[0.0, 0.1, 0.2, 0.5, 1.0],
):
    print(f"Generating sanity check: {save_path}")
    fig, axes = plt.subplots(
        len(indices_dict), len(plot_pcts),
        figsize=(2.5 * len(plot_pcts), 2.5 * len(indices_dict)),
    )
    if len(indices_dict) == 1:
        axes = np.expand_dims(axes, 0)

    for row, (method, idx_list) in enumerate(indices_dict.items()):
        total = len(idx_list)
        grid_w = img.shape[3] // patch_size

        for col, p in enumerate(plot_pcts):
            ax = axes[row, col]
            mask = torch.ones((1, 1, img.shape[2], img.shape[3]), device=device)

            for idx in idx_list[: int(p * total)]:
                r, c = idx // grid_w, idx % grid_w
                mask[:, :, r * patch_size:(r + 1) * patch_size,
                     c * patch_size:(c + 1) * patch_size] = 0.0

            imputed = impute_road_diffusion(img, mask, steps=15)
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            disp = np.clip(
                (imputed * std + mean)[0].detach().cpu().permute(1, 2, 0).numpy(), 0, 1
            )
            ax.imshow(disp)
            if row == 0:
                ax.set_title(f"{int(p*100)}% Removed", fontsize=14, fontweight="bold", pad=10)
            if col == 0:
                ax.set_ylabel(method, fontsize=14, fontweight="bold", labelpad=10)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, format="jpg", dpi=150, bbox_inches="tight")
    plt.close()

def benchmark_patch_morf_inner(
    model, img, indices_dict, patch_size, init_prob, target,
    perturbation_tensor, device, percentages, use_road=True,
):
    results = {}
    B, C, H, W = img.shape
    grid_w = W // patch_size

    for method, idx_list in indices_dict.items():
        curve = [init_prob]
        total_patches = len(idx_list)

        with torch.no_grad():
            for p in percentages[1:]:
                num_ablate = int(p * total_patches)
                current_mask = torch.ones((B, 1, H, W), device=device)
                curr_mean = img.clone()

                for idx in idx_list[:num_ablate]:
                    r, c = idx // grid_w, idx % grid_w
                    y1, y2 = r * patch_size, (r + 1) * patch_size
                    x1, x2 = c * patch_size, (c + 1) * patch_size
                    if use_road:
                        current_mask[:, :, y1:y2, x1:x2] = 0.0
                    else:
                        curr_mean[:, :, y1:y2, x1:x2] = perturbation_tensor

                curr_img = (
                    impute_road_diffusion(img, current_mask, steps=15)
                    if use_road
                    else curr_mean
                )

                out = model(curr_img, task)
                prob = torch.softmax(
                    out.logits if hasattr(out, "logits") else out, dim=-1
                )[0, target].item()
                curve.append(prob)

        results[method] = curve

    return results, init_prob

def benchmark_dataset_morf(
    model,
    bank: SAEBank,
    layers: List[int],
    dataloader,
    method_funcs: Dict[str, Any],
    dataset_mean: torch.Tensor,
    patch_size: int = 14,
    max_samples: Optional[int] = None,
    device: str = "cuda",
    sae_k_values: List[int] = [1, 5, 10, 20, 50],
    mode: str = "morf",
    percentages: Optional[List[float]] = None,
    routing: RoutingMode = "dla",
    majority_class_idx: int = 5
):
    """
    MoRF / LeRF patch-perturbation benchmark.

    Parameters
    ----------
    model : _ForwardWrapper
        image -> logits module wrapping a vitlab MultiTaskViT (fixed task). Used
        for the baseline methods (LRP, GradCAM) and the perturbation forward passes.
        SAE attribution reads activations through the underlying reader/spec.
    bank : SAEBank
        String-keyed bank; must hold a LayerSAE at every "blocks.{l}.resid_post"
        for l in `layers`.
    layers : list[int]
        ViT block indices that have trained SAEs.
    routing : RoutingMode
        SAE attribution routing mode.  "dragnet" (default) is fastest for
        large-scale benchmarks; no extra forward passes per candidate.
    """
    if percentages is None:
        percentages = [i / 10 for i in range(11)]

    model.eval()
    task = model.task_names[0]      # single-head model for this benchmark
    sites = [f"blocks.{l}.resid_post" for l in layers]

    sae_keys = [f"SAE@{k}" for k in sae_k_values]
    all_method_names = list(method_funcs.keys()) + sae_keys
    aggregated_curves = {name: [] for name in all_method_names}
    aopc_scores = {name: [] for name in all_method_names}

    count = 0
    MAJORITY_CLASS_IDX = majority_class_idx
    max_k = max(sae_k_values)

    if dataset_mean.ndim == 1:
        dataset_mean = dataset_mean.view(3, 1, 1)
    dataset_mean = dataset_mean.to(device)

    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Benchmarking")):
        if max_samples and count >= max_samples:
            break

        images = images.to(device)

        for i in range(len(images)):
            if max_samples and count >= max_samples:
                break
            if labels[i].item() == MAJORITY_CLASS_IDX:
                continue

            img = images[i : i + 1]

            with torch.no_grad():
                out = model(img)
                logits = out.logits if hasattr(out, "logits") else out
                target_idx = logits.argmax(dim=-1).item()
                initial_prob = torch.softmax(logits, dim=-1)[0, target_idx].item()

            if initial_prob < 0.01:
                continue

            heatmaps = {}
            for name, func in method_funcs.items():
                heatmaps[name] = func(model, img, device)

            sae_res = _sae_relevance(
                model, bank, img, sites,
                target_class_idx = target_idx,
                k = max_k,
                routing = routing,
                task = task,
                device = device,
            )

            for k_val in sae_k_values:
                heatmaps[f"SAE@{k_val}"] = create_sae_heatmap(
                    sae_results = sae_res,
                    k = k_val,
                    target_shape = (img.shape[2], img.shape[3]),
                    device = device,
                )

            patch_indices = {}
            _, _, H, W = img.shape
            total_patches = (H // patch_size) * (W // patch_size)

            for name, hm in heatmaps.items():
                if isinstance(hm, np.ndarray):
                    hm = torch.from_numpy(hm)
                hm = hm.to(device).float()
                if hm.ndim == 2:
                    hm = hm.unsqueeze(0)
                elif hm.ndim == 4:
                    hm = hm.squeeze(1)

                patch_rel = heatmap_to_patches(hm, patch_size=patch_size)
                is_morf = mode == "morf"
                indices, _ = topk_patches(patch_rel, k=total_patches, largest=is_morf)
                patch_indices[name] = indices

            if count % 50 == 0:
                plot_ablation_sanity_check(
                    img, patch_indices, patch_size, device,
                    save_path=f"./out/sanity_check_{mode}_img{count}.jpg",
                )

            img_curves, _ = benchmark_patch_morf_inner(
                model, img, patch_indices, patch_size,
                initial_prob, target_idx, dataset_mean, device, percentages,
            )

            for name, curve in img_curves.items():
                norm_curve = np.array(curve) / (initial_prob + 1e-8)
                aggregated_curves[name].append(norm_curve)
                aopc_scores[name].append(norm_curve[0] - norm_curve[1:])

            count += 1

    final_metrics = {}
    header = "Higher is Better" if mode == "morf" else "Lower is Better"
    print(f"\n--- {mode.upper()} Results ({header}) ---")
    print(f"{'Method':<15} | {'@10%':<10} | {'@20%':<10} | {'@50%':<10} | {'@100%':<10}")
    print("-" * 65)

    for name in aggregated_curves:
        avg_drops = np.mean(np.array(aopc_scores[name]), axis=0)
        score_at_10 = np.mean(avg_drops[:2])  if len(avg_drops) > 1  else 0
        score_at_20 = np.mean(avg_drops[:4])  if len(avg_drops) > 3  else 0
        score_at_50 = np.mean(avg_drops[:10]) if len(avg_drops) > 9  else 0
        score_at_100 = np.mean(avg_drops[:20]) if len(avg_drops) > 19 else 0

        final_metrics[name] = {
            "percentages": percentages,
            "aopc@10%":    score_at_10,
            "aopc@20%":    score_at_20,
            "aopc@50%":    score_at_50,
            "aopc@100%":   score_at_100,
            "curve_mean":  np.mean(aggregated_curves[name], axis=0),
            "curve_std":   np.std(aggregated_curves[name], axis=0),
        }
        print(f"{name:<15} | {score_at_10:.4f}     | {score_at_20:.4f}     | "
              f"{score_at_50:.4f}     | {score_at_100:.4f}")

    return final_metrics

def plot_results_with_error(metrics, mode="MoRF", model_type="dinov3"):
    plt.figure(figsize=(10, 7))
    standard_colors = {"LRP": "black", "GradCAM": "gray"}
    sae_keys = sorted(
        [k for k in metrics if "SAE" in k],
        key=lambda x: int(x.split("@")[1]) if "@" in x else 0,
    )
    sae_colors = {}
    if sae_keys:
        cmap = plt.get_cmap("viridis")
        colors = cmap(np.linspace(0, 0.9, len(sae_keys)))
        sae_colors = {k: colors[i] for i, k in enumerate(sae_keys)}

    x_axis = (
        np.array(next(iter(metrics.values()))["percentages"]) * 100
    )

    for name, data in metrics.items():
        mean_c = data["curve_mean"]
        sem_c = data["curve_std"] / np.sqrt(50)
        if "SAE" in name:
            color, lw, zo, marker = sae_colors[name], 2.5, 10, None
        else:
            color  = standard_colors.get(name, "gray")
            lw, zo = 2, 5
            marker = "o" if "LRP" in name else "s"

        label = f"{name} (AOPC@50%: {data.get('aopc@50%', 0):.3f})"
        plt.plot(x_axis, mean_c, label=label, color=color,
                 linewidth=lw, marker=marker, markersize=4, zorder=zo)
        plt.fill_between(x_axis, mean_c - sem_c, mean_c + sem_c,
                         color=color, alpha=0.1, zorder=zo - 1)

    title = "MoRF" if mode == "morf" else "LeRF"
    plt.title(f"{title} Ablation", fontsize=14)
    plt.xlabel("Percentage of Image Ablated (%)", fontsize=12)
    plt.ylabel("Normalized Class Probability", fontsize=12)
    plt.axhline(1.0, color="black", linestyle="--", alpha=0.3)
    plt.xlim(0, 100)
    plt.grid(True, alpha=0.2)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(f"./out/pdfs/{mode}_{model_type}_percentage.pdf")

def plot_morf_lerf_combined(morf_data, lerf_data, num_samples=None):
    """
    Plots MoRF and LeRF side-by-side with consistent coloring.
    
    Args:
        morf_data: Metrics dictionary for MoRF
        lerf_data: Metrics dictionary for LeRF
        num_samples: (int) Number of images processed. 
                     If None, plots Standard Deviation instead of Standard Error.
    """
    plt.style.use("../thesis.mplstyle")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    
    all_keys = set(morf_data.keys()) | set(lerf_data.keys())
    sae_keys = [k for k in all_keys if 'SAE' in k]
    
    try:
        sae_keys.sort(key=lambda x: int(x.split('@')[1]) if '@' in x else 0)
    except:
        sae_keys.sort()
        
    sae_colors = {}
    if len(sae_keys) > 0:
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 0.9, len(sae_keys)))
        for i, key in enumerate(sae_keys):
            sae_colors[key] = colors[i]
            
    standard_colors = {'LRP': '#873e23', 'GradCAM': '#e28743'}

    def plot_subplot(ax, metrics, mode):
        is_morf = (mode == 'MoRF')
        
        for name, data in metrics.items():
            mean_curve = data['curve_mean']
            std_curve = data['curve_std']

            if 'percentages' in data:
                x = np.array(data['percentages']) * 100
            else:
                x = np.linspace(0, 100, len(mean_curve)) 
            
            if num_samples:
                error_curve = std_curve / np.sqrt(num_samples)
            else:
                error_curve = std_curve 
            
            if 'SAE' in name:
                color = sae_colors.get(name, 'red')
                lw = 1.0
                zorder = 10
                marker = None
                linestyle="solid"
                alpha=0.7
            else:
                color = standard_colors.get(name, 'blue')
                lw = 1.5
                zorder = 5
                marker = None
                linestyle="dashed"
                alpha=1.0

            label = name
            
            dynamic_markevery = max(1, len(x) // 5)
            
            ax.plot(x, mean_curve, label=label, color=color, linestyle=linestyle, alpha=alpha, 
                    linewidth=lw, marker=marker, markevery=dynamic_markevery, markersize=5, zorder=zorder)
            
            ax.fill_between(x, mean_curve - error_curve, mean_curve + error_curve, 
                            color=color, alpha=0.1, zorder=zorder-1)

        title = "MoRF" if is_morf else "LeRF"
        
        ax.set_title(f"{title}", fontsize=12, fontweight='bold')
        ax.set_xlabel("% Patches Removed", fontsize=12)
        
        if is_morf:
            ax.set_ylabel("Normalized Class Probability", fontsize=12)
        
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        
        if is_morf:
            ax.legend(fontsize=9, loc='upper right')
        
        ax.axhline(1.0, color='black', linestyle=':', alpha=0.3)


    plot_subplot(axes[0], morf_data, "MoRF")
    plot_subplot(axes[1], lerf_data, "LeRF")
    plt.tight_layout()
    plt.savefig("../out/pdfs/morf-lerf-base.pdf")
    plt.show()

def generate_latex_table(metrics_dict, mode="morf"):
    caption_text = (
        "Most Relevant First (MoRF)" if mode.lower() == "morf"
        else "Least Relevant First (LeRF)"
    )
    trend_text = "Higher is better" if mode.lower() == "morf" else "Lower is better"

    print(f"\n% ====== LaTeX Table for {mode.upper()} ======\n")
    print(r"\begin{table}[h!]")
    print(r"\centering")
    print(r"\begin{tabularx}{\textwidth}{X X X X X}")
    print(r"\toprule")
    print(r"\textbf{Method} & \textbf{@10\%} & \textbf{@20\%} & \textbf{@50\%} & \textbf{@100\%} \\")
    print(r"\midrule")
    for name, data in metrics_dict.items():
        v10, v20  = data.get("aopc@10%", 0), data.get("aopc@20%", 0)
        v50, v100 = data.get("aopc@50%", 0), data.get("aopc@100%", 0)
        print(f"{name} & {v10:.3f} & {v20:.3f} & {v50:.3f} & {v100:.3f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabularx}")
    print(f"\\caption{{AOPC for {caption_text}. {trend_text}.}}")
    print(f"\\label{{tab:aopc_{mode}}}")
    print(r"\end{table}")

def create_label_map_2(dataset):
    print("Building label map...")
    if hasattr(dataset, "features") and "label" in dataset.features:
        feat = dataset.features["label"]
        if isinstance(feat, ClassLabel):
            return {name: i for i, name in enumerate(feat.names)}
    unique = (
        set(dataset.unique("label"))
        if hasattr(dataset, "unique")
        else {dataset[i]["label"] for i in range(len(dataset))}
    )
    return {label: idx for idx, label in enumerate(sorted(unique))}

class _ForwardWrapper(nn.Module):
    def __init__(self, mtv, task, processor):
        super().__init__()
        self.mtv = mtv
        self.task = task
        self.spec = mtv.spec
        self.n_prefix = mtv.spec.n_prefix_tokens
        self.patch_size = mtv.spec.patch_size
        self.blocks = mtv.backbone.blocks
        self._processor = processor

    def forward(self, pixel_values):
        return self.mtv(pixel_values, self.task)

    def preprocess(self, pil_image):
        out = self._processor(pil_image, return_tensors="pt")["pixel_values"][0]
        return out


def _build_sites_and_layers(bank, n_layers):
    """Layers with a trained SAE, and their site strings."""
    layers = sorted(l for l in range(n_layers) if f"blocks.{l}.resid_post" in bank.saes)
    return layers, [f"blocks.{l}.resid_post" for l in layers]


if __name__ == "__main__":
    args   = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    MODEL_TYPE       = args.model
    DATASET          = args.dataset
    ROUTING          = args.routing
    PERCENTAGES      = [i / 20 for i in range(21)]   # 0%, 5%, 10%, ..., 100%
    MAX_SAMPLES      = None

    print("\n=== Loading model (backbone + head) ===")
    model_mtv = vitlab.load_model(args.run, device=device)
    from vitlab.backbone import load_image_processor
    processor = load_image_processor(model_mtv.spec.key)
    task = args.dataset if args.dataset in model_mtv.task_names else model_mtv.task_names[0]
    model = _ForwardWrapper(model_mtv, task, processor).to(device).eval()
    NUM_PREFIX_TOKENS = model.n_prefix
    PATCH_SIZE = model.patch_size

    print("\n=== Loading SAEs ===")
    bank = discover_bank(args.sae_root, device=device)
    LAYERS, _ = _build_sites_and_layers(bank, model_mtv.spec.n_layers)
    print(f"  SAEs at layers: {LAYERS}")

    print("\n=== Loading Dataset ===")
    _, _, test = get_splits(DATASET, model_key=model_mtv.spec.key)

    def collate_fn(batch):
        return (
            torch.stack([item["pixel_values"] for item in batch]),
            torch.stack([item["labels"] if torch.is_tensor(item["labels"])
                         else torch.tensor(item["labels"]) for item in batch]),
        )

    dl = torch.utils.data.DataLoader(
        test, batch_size=1, shuffle=False, collate_fn=collate_fn,
    )

    if args.run_diagnostics:
        print("Running diagnostics for", MODEL_TYPE)
        run_diagnostics(model, dl, device, layer_idx=10)
        import sys; sys.exit(0)

    ds_mean = get_dataset_mean(dl, device=device)

    methods = {
        "LRP":     lambda m, img, dev: get_attn_lrp_attr(m, img, dev),
        "GradCAM": lambda m, img, dev: get_vit_cam(
            m, img, dev, prefix_tokens=NUM_PREFIX_TOKENS
        ),
    }

    os.makedirs("./out/pdfs", exist_ok=True)

    for mode in ["morf", "lerf"]:
        print(f"\n{'='*40}")
        print(f"   STARTING: {mode.upper()}  (routing={ROUTING})")
        print(f"{'='*40}")

        res = benchmark_dataset_morf(
            model        = model,
            bank         = bank,
            layers       = LAYERS,
            dataloader   = dl,
            method_funcs = methods,
            dataset_mean = ds_mean,
            patch_size   = PATCH_SIZE,
            max_samples  = MAX_SAMPLES,
            device       = device,
            mode         = mode,
            percentages  = PERCENTAGES,
            routing      = ROUTING,
        )

        torch.save(res, f"./out/{mode}-{DATASET}_{MODEL_TYPE}_{ROUTING}.pt")
        plot_results_with_error(res, mode, MODEL_TYPE)
        generate_latex_table(res, mode=mode)
        print(f"Finished {mode.upper()} — results saved to ./out/")