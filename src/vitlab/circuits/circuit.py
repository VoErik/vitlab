"""
Follows the SAE-circuit methodology: attribute the decision to SAE features
(nodes) and to feature->feature interactions across layers (edges), using
attribution patching on the codes with a median-activation baseline, normalised
by per-layer gradient norm so early layers are not inflated by skip-connection
gradient accumulation.

Node importance   I(u)   = mean_t[ (dm/dx_t . W_dec[u]) * (z_{u,t} - median_u) ]
Edge importance   I(u->d) = jacobian(d,u) * mean_t[ (dm/dz_{d,t}) * (z_{u,t} - median_u) ]
                  with jacobian(d,u) = W_enc[d] . W_dec[u]

Two gradient modes:
  standard    one backward pass per image; each layer gets its own accumulated
              gradient (early layers larger due to skip connections).
  libragrad   the final-layer gradient is reused for all layers, removing the
              skip-connection accumulation that inflates early-layer importance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

@dataclass
class CircuitNode:
    layer: int
    feature_idx: int
    activation: float
    node_importance: float = 0.0
    edge_importance: float = 0.0
    label: str = ""

    def key(self) -> tuple[int, int]:
        return (self.layer, self.feature_idx)

    def __repr__(self):
        s = f"L{self.layer}#{self.feature_idx} [node={self.node_importance:.4f}, edge={self.edge_importance:.4f}]"
        return s + (f" '{self.label}'" if self.label else "")


@dataclass
class Circuit:
    nodes: dict[int, list[CircuitNode]] = field(default_factory=dict)
    edges: list[tuple[CircuitNode, CircuitNode, float]] = field(default_factory=list)

    def summary(self) -> None:
        n_nodes = sum(len(v) for v in self.nodes.values())
        print(f"Circuit: {n_nodes} nodes, {len(self.edges)} edges\n")
        for layer in sorted(self.nodes):
            print(f"  Layer {layer}:")
            for n in sorted(self.nodes[layer], key=lambda x: -abs(x.node_importance)):
                print(f"    {n}")
        if self.edges:
            print("\n  Top edges:")
            for u, d, w in sorted(self.edges, key=lambda e: -abs(e[2]))[:10]:
                print(f"    L{u.layer}#{u.feature_idx} -> L{d.layer}#{d.feature_idx}  w={w:.4f}")

    def prune(self, edge_percentile: float = 50.0, min_node_importance: float = 0.0) -> "Circuit":
        """
        Drop weak edges (below the percentile) and any node left disconnected,
        always keeping the final-layer seed nodes.
        """
        import numpy as np

        pruned = Circuit()
        if not self.edges:
            pruned.nodes = {k: list(v) for k, v in self.nodes.items()}
            return pruned

        threshold = float(np.percentile([abs(w) for _, _, w in self.edges], edge_percentile))
        pruned.edges = [(u, d, w) for u, d, w in self.edges if abs(w) >= threshold]

        connected = set()
        for u, d, _ in pruned.edges:
            connected.add(u.key()); connected.add(d.key())
        for n in self.nodes[max(self.nodes)]:
            connected.add(n.key())

        for layer, nodes in self.nodes.items():
            kept = [n for n in nodes if n.key() in connected and abs(n.node_importance) >= min_node_importance]
            if kept:
                pruned.nodes[layer] = kept
        return pruned

    def save(self, path, *, metadata=None) -> None:
        """Save to a human-readable JSON file."""
        from .io import save_circuit
        save_circuit(self, path, metadata=metadata)

    @classmethod
    def load(cls, path) -> "Circuit":
        """Load a circuit saved with .save()."""
        from .io import load_circuit
        return load_circuit(path)

    def plot(self, *, save=None, show=False, **kwargs):
        from ..viz.circuit_viz import visualize_circuit
        return visualize_circuit(self, save=save, show=show, **kwargs)

    def to_html(self, model, bank, dataloader, path, **kwargs):
        from ..viz.circuit_html import save_circuit_html
        save_circuit_html(self, model, bank, dataloader, str(path), **kwargs)


def _decoder_weight(layer_sae) -> torch.Tensor:
    """(F, D) decoder atoms."""
    return layer_sae.dictionary

# TODO: this must be kept updated with overcomplete
def _encoder_weight(layer_sae) -> torch.Tensor:
    """(F, D) encoder weight. overcomplete's MLPEncoder exposes this at
    encoder.final_block[0].weight; fall back to searching for the (F, D) linear."""
    sae = layer_sae.sae
    try:
        w = sae.encoder.final_block[0].weight
        if w.shape[0] == layer_sae.n_concepts:
            return w
    except (AttributeError, IndexError):
        pass
    import torch.nn as nn
    cand = None
    for m in sae.encoder.modules():
        if isinstance(m, nn.Linear) and m.weight.shape[0] == layer_sae.n_concepts:
            cand = m.weight
    if cand is None:
        raise RuntimeError("could not locate SAE encoder weight (F, D) for edge importance")
    return cand

class AttributionPatcher:
    """
    Computes median-baselined act x grad importance for SAE features and their
    cross-layer interactions.
    """

    def __init__(self, model, bank, medians: dict[int, torch.Tensor], *, task=None,
                 site_kind="resid_post", device="cuda"):
        self.model = model.to(device).eval()
        self.bank = bank.to(device)
        self.medians = medians
        self.device = device
        self.site_kind = site_kind
        self.prefix = model.spec.n_prefix_tokens
        from ..attribution.core import _resolve_task
        self.task = _resolve_task(model, task)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def site(self, layer: int) -> str:
        return f"blocks.{layer}.{self.site_kind}"

    def objective(self, logits, target_class) -> torch.Tensor:
        """logit(target) - mean(logits)."""
        return logits[0, target_class] - logits[0].mean()

    def _grads(self, image, target_class, layers, *, libragrad=False):
        """
        Returns {layer: grad (1,S,D)} and {layer: patch-grad norm}. 
        In libragrad mode every layer shares the final layer's gradient.
        """
        reader = self.model.reader
        captured = {}
        handles = []

        def mk(site):
            def hook(_m, _in, out):
                is_t = isinstance(out, tuple)
                x = (out[0] if is_t else out).clone().requires_grad_(True)
                captured[site] = x
                return (x,) + out[1:] if is_t else x
            return hook

        sites = {l: self.site(l) for l in layers}
        for l, s in sites.items():
            path, _ = reader.site_path(s)
            handles.append(reader.backbone.submodule(path).register_forward_hook(mk(s)))
        try:
            image_g = image.detach().to(self.device)
            logits = self.model(image_g, self.task)
            m = self.objective(logits, target_class)
            if libragrad:
                final = max(layers)
                g_final = torch.autograd.grad(m, captured[sites[final]])[0]
                grads = {l: g_final for l in layers}
            else:
                gs = torch.autograd.grad(m, [captured[sites[l]] for l in layers])
                grads = {l: g for l, g in zip(layers, gs)}
        finally:
            for h in handles:
                h.remove()

        norms = {l: grads[l][:, self.prefix:, :].norm().item() + 1e-8 for l in layers}
        self._last_residuals = {l: captured[sites[l]].detach() for l in layers}
        return grads, norms

    def _codes(self, layer: int, residual: torch.Tensor) -> torch.Tensor:
        """(T, F) codes for the patch tokens of a cached residual."""
        sp = residual[:, self.prefix:, :].squeeze(0)
        return self.bank[self.site(layer)].encode(sp)

class CircuitDiscovery:
    def __init__(self, patcher: AttributionPatcher, n_layers: int, *, seed_layer: int | None = None):
        """
        seed_layer: the deepest present layer whose paatch tokens still influence the
        logit. Under CLS pooling the final block's patches feed nothing downstream,
        so their gradient and node importance is zero; seeding there yields an
        all-zero circuit. Default None auto-detects the deepest non-dead layer.
        """
        self.patcher = patcher
        self.n_layers = n_layers
        self.seed_layer = seed_layer
        self.layers = sorted(
            l for l in range(n_layers) if patcher.site(l) in patcher.bank.saes
        )
        if not self.layers:
            raise ValueError(
                f"bank has no SAE at any blocks.*.{patcher.site_kind} site for a "
                f"{n_layers}-layer model; circuits need per-layer SAEs."
            )

    def _detect_seed_layer(self, images, target_class) -> int:
        """Deepest present layer with non-zero patch gradient."""
        import warnings

        p = self.patcher
        _, norms = p._grads(images[0], target_class, self.layers, libragrad=False)
        for l in reversed(self.layers):
            if norms[l] > 1e-6:
                deeper = [x for x in self.layers if x > l]
                if deeper:
                    warnings.warn(
                        f"layers {deeper} have zero patch gradient (CLS-pooled head "
                        f"reads only the CLS token there); seeding the circuit at layer "
                        f"{l}, the deepest layer whose patches still drive the logit.",
                        stacklevel=3,
                    )
                return l
        raise RuntimeError("no present layer has non-zero patch gradient; is head mean-pooled?")

    def compute_importance_matrices(self, images, target_class, *, use_libragrad=False):
        """
        Average node importance (F,), edge importance (F_l, F_d), and mean codes
        over a set of images.
        """
        from tqdm.auto import tqdm

        p = self.patcher
        layers = self.layers
        W_dec = {l: _decoder_weight(p.bank[p.site(l)]).detach().cpu() for l in layers}
        W_enc = {l: _encoder_weight(p.bank[p.site(l)]).detach().cpu() for l in layers}
        med = {l: p.medians[l].cpu() for l in layers}
        pairs = list(zip(layers[:-1], layers[1:]))

        node_acc, edge_acc, z_acc = {}, {}, {}
        mode = "LibraGrad" if use_libragrad else "standard"
        for img in tqdm(images, desc=f"importance ({mode})"):
            grads, norms = p._grads(img, target_class, layers, libragrad=use_libragrad)
            resid = p._last_residuals
            g_cpu = {l: grads[l].cpu() for l in layers}
            z_cpu = {l: p._codes(l, resid[l]).cpu() for l in layers}

            for l in layers:
                gp = g_cpu[l][:, p.prefix:, :].squeeze(0) # (T, D)
                grad_z = gp @ W_dec[l].T # (T, F)
                delta = z_cpu[l] - med[l].unsqueeze(0) # (T, F)
                ni = (grad_z * delta).mean(0) / norms[l] # (F,)
                node_acc[l] = node_acc.get(l, torch.zeros_like(ni)) + ni
                zc = z_cpu[l].mean(0)
                z_acc[l] = z_acc.get(l, torch.zeros_like(zc)) + zc

            for l, ld in pairs:
                gd = g_cpu[ld][:, p.prefix:, :].squeeze(0)
                grad_zd = gd @ W_dec[ld].T # (T, F_d)
                delta_l = z_cpu[l] - med[l].unsqueeze(0) # (T, F_l)
                T = grad_zd.shape[0]
                cross = (delta_l.T @ grad_zd) / T # (F_l, F_d)
                jac = W_enc[ld] @ W_dec[l].T # (F_d, F_l)
                I = (jac.T * cross) / (norms[l] * norms[ld]) ** 0.5
                edge_acc[l] = edge_acc.get(l, torch.zeros_like(I)) + I

        n = len(images)
        dev = p.device
        node_imp = {l: (v / n).to(dev) for l, v in node_acc.items()}
        edge_imp = {l: (v / n).to(dev) for l, v in edge_acc.items()}
        z_avg = {l: (v / n).to(dev) for l, v in z_acc.items()}
        return node_imp, edge_imp, z_avg

    def build_circuit_from_matrices(self, node_imp, edge_imp, z_avg, top_k, *,
                                    seed_layer=None, annotations=None):
        """
        Greedy top-k per layer from precomputed matrices, seeded at `seed_layer`
        and walking up by aggregated edge importance.
        """
        annotations = annotations or {}
        circuit = Circuit()
        present = self.layers
        seed = seed_layer if seed_layer is not None else present[-1]
        chain = [l for l in present if l <= seed]
        if top_k == 0:
            circuit.nodes = {l: [] for l in chain}
            return circuit

        final = chain[-1]
        k = min(top_k, node_imp[final].shape[0])
        top_final = node_imp[final].abs().topk(k).indices.tolist()
        circuit.nodes[final] = [
            CircuitNode(final, i, z_avg[final][i].item(), node_imp[final][i].item(), 0.0,
                        annotations.get((final, i), "")) for i in top_final
        ]

        for idx in range(len(chain) - 2, -1, -1):
            layer = chain[idx]
            down_layer = chain[idx + 1]
            downstream = circuit.nodes[down_layer]
            d_idxs = [n.feature_idx for n in downstream]
            # edge_imp is keyed by the UPSTREAM layer of each consecutive pair
            emat = edge_imp.get(layer)
            agg = emat[:, d_idxs].sum(1) if (emat is not None and d_idxs) else node_imp[layer]
            k = min(top_k, agg.shape[0])
            top_up = agg.abs().topk(k).indices.tolist()
            circuit.nodes[layer] = [
                CircuitNode(layer, i, z_avg[layer][i].item(), node_imp[layer][i].item(),
                            agg[i].item(), annotations.get((layer, i), "")) for i in top_up
            ]
            for u in circuit.nodes[layer]:
                for d in downstream:
                    circuit.edges.append((u, d, edge_imp[layer][u.feature_idx, d.feature_idx].item()))
        return circuit

    def discover_aggregated(self, images, target_class, *, top_k=3, use_libragrad=False,
                            annotations=None):
        """
        Importance over a set of images, then a top-k circuit. Averaging over images gives a class-level circuit.
        """
        seed = self.seed_layer
        if seed is None:
            seed = self._detect_seed_layer(images, target_class)
        ni, ei, za = self.compute_importance_matrices(images, target_class, use_libragrad=use_libragrad)
        return self.build_circuit_from_matrices(ni, ei, za, top_k, seed_layer=seed, annotations=annotations)