"""Causal evaluation of a discovered circuit (Kim et al.).

A circuit is only meaningful if it actually captures the model's computation.
Three causal metrics measure that, all via mid-forward SAE patching with the
error term frozen from a clean pass:

  faithfulness  keep ONLY the circuit's features (everything else -> median).
                High = the circuit alone reproduces the decision.
  completeness  remove the circuit's features (-> median), keep the rest.
                High = the decision breaks without the circuit, i.e. nothing
                outside the circuit can compensate.
  causality     ablate each layer's circuit features, measure the activation drop
                in downstream circuit features. High = the edges carry real signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

@dataclass
class CircuitEvaluation:
    faithfulness: float
    completeness: float
    causality: float
    m_full: float
    m_circuit: float
    m_empty: float
    m_complement: float
    per_layer_causality: dict = field(default_factory=dict)


class CircuitEvaluator:
    """Faithfulness / completeness / causality for a circuit on a MultiTaskViT."""

    def __init__(self, model, bank, medians, target_class, n_layers, *, task=None,
                 site_kind="resid_post", device="cuda"):
        self.model = model.to(device).eval()
        self.bank = bank.to(device)
        self.medians = medians
        self.target_class = target_class
        self.n_layers = n_layers
        self.site_kind = site_kind
        self.device = device
        self.prefix = model.spec.n_prefix_tokens
        from ..attribution.core import _resolve_task
        self.task = _resolve_task(model, task)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def site(self, layer):
        return f"blocks.{layer}.{self.site_kind}"

    def _objective(self, logits):
        return (logits[0, self.target_class] - logits[0].mean()).item()

    @torch.no_grad()
    def _clean_cache(self, image):
        """Per layer: (codes, error) with error frozen at the clean reconstruction."""
        reader = self.model.reader
        cache = {}
        for layer in range(self.n_layers):
            site = self.site(layer)
            if site not in self.bank.saes:
                continue
            acts = reader.read(image.to(self.device), site)[:, self.prefix:, :]
            B, P, D = acts.shape
            ls = self.bank[site]
            codes = ls.encode(acts.reshape(B * P, D))
            recon = ls.decode(codes)
            error = acts.reshape(B * P, D) - recon # frozen clean error
            cache[layer] = (codes, error, (B, P, D))
        return cache

    def _patched_residual(self, layer, circuit, mode, cache):
        """Compute the patched patch-residual for one layer from frozen codes."""
        codes, error, (B, P, D) = cache[layer]
        med = self.medians[layer]
        ls = self.bank[self.site(layer)]

        if mode == "empty":
            z = med.unsqueeze(0).expand_as(codes).clone()
        elif mode == "faithfulness":
            feats = [n.feature_idx for n in circuit.nodes.get(layer, [])]
            z = med.unsqueeze(0).expand_as(codes).clone()
            if feats:
                z[:, feats] = codes[:, feats]
        elif mode == "completeness":
            feats = [n.feature_idx for n in circuit.nodes.get(layer, [])]
            z = codes.clone()
            if feats:
                z[:, feats] = med[feats].unsqueeze(0)
        else:
            raise ValueError(mode)
        return (ls.decode(z) + error).reshape(B, P, D)

    @torch.no_grad()
    def _run_patched(self, image, circuit, mode):
        """Second pass injecting pre-computed patched residuals."""
        cache = self._clean_cache(image)
        layers = (list(cache) if mode == "empty" else
                  [l for l in circuit.nodes if l in cache])
        patched = {l: self._patched_residual(l, circuit, mode, cache) for l in layers}

        reader = self.model.reader
        handles = []

        def mk(layer):
            patch = patched[layer].to(self.device)

            def hook(_m, _in, out):
                is_t = isinstance(out, tuple)
                x = (out[0] if is_t else out).clone()
                x[:, self.prefix:, :] = patch.to(x.dtype)
                return (x,) + out[1:] if is_t else x
            return hook

        for l in layers:
            path, _ = reader.site_path(self.site(l))
            handles.append(reader.backbone.submodule(path).register_forward_hook(mk(l)))
        try:
            logits = self.model(image.to(self.device), self.task)
        finally:
            for h in handles:
                h.remove()
        return logits

    @torch.no_grad()
    def _causality_for_layer(self, image, circuit, source_layer, cache):
        """Ablate source_layer's circuit features; mean downstream activation drop."""
        reader = self.model.reader
        downstream = [l for l in sorted(circuit.nodes) if l > source_layer and l in cache]
        if not downstream:
            return 0.0

        def down_acts():
            out = {}
            for dl in downstream:
                codes, _, (B, P, D) = cache[dl]
                for n in circuit.nodes[dl]:
                    out[(dl, n.feature_idx)] = codes[:, n.feature_idx].mean().item()
            return out

        base = down_acts()
        src_idxs = [n.feature_idx for n in circuit.nodes[source_layer]]
        med = self.medians[source_layer]
        ls = self.bank[self.site(source_layer)]

        def hook(_m, _in, out):
            is_t = isinstance(out, tuple)
            x = (out[0] if is_t else out).clone()
            sp = x[:, self.prefix:, :]
            B, P, D = sp.shape
            z = ls.encode(sp.reshape(B * P, D))
            recon = ls.decode(z)
            err = sp.reshape(B * P, D) - recon
            z[:, src_idxs] = med[src_idxs].unsqueeze(0)
            x[:, self.prefix:, :] = (ls.decode(z) + err).reshape(B, P, D).to(x.dtype)
            return (x,) + out[1:] if is_t else x

        path, _ = reader.site_path(self.site(source_layer))
        h = reader.backbone.submodule(path).register_forward_hook(hook)
        try:
            abl_cache = self._clean_cache(image)
        finally:
            h.remove()

        drops = []
        for dl in downstream:
            codes, _, _ = abl_cache[dl]
            for n in circuit.nodes[dl]:
                drops.append(base[(dl, n.feature_idx)] - codes[:, n.feature_idx].mean().item())
        return sum(drops) / len(drops) if drops else 0.0

    def evaluate(self, image, circuit) -> CircuitEvaluation:
        with torch.no_grad():
            m_full = self._objective(self.model(image.to(self.device), self.task))
        m_empty = self._objective(self._run_patched(image, circuit, "empty"))
        m_circuit = self._objective(self._run_patched(image, circuit, "faithfulness"))
        m_compl = self._objective(self._run_patched(image, circuit, "completeness"))

        denom = m_full - m_empty + 1e-8
        faith = min((m_circuit - m_empty) / denom, 1.0)
        comp = 1.0 - min((m_compl - m_empty) / denom, 1.0)

        cache = self._clean_cache(image)
        final_l = max(circuit.nodes)
        per_layer = {}
        for layer in sorted(circuit.nodes):
            if layer == final_l or not circuit.nodes[layer]:
                continue
            per_layer[layer] = self._causality_for_layer(image, circuit, layer, cache)
        causality = sum(per_layer.values()) / len(per_layer) if per_layer else 0.0

        return CircuitEvaluation(faith, comp, causality, m_full, m_circuit, m_empty, m_compl, per_layer)

    def evaluate_over_k(self, images, discoverer, *, k_values=None, max_features=None,
                        importance_matrices=None, annotations=None, seed_layer=None):
        """Sweep circuit size; return (results, auc_faith, auc_comp)."""
        from tqdm.auto import tqdm

        if importance_matrices is None:
            ni, ei, za = discoverer.compute_importance_matrices(images, self.target_class)
        else:
            ni, ei, za = importance_matrices
        if seed_layer is None:
            seed_layer = discoverer._detect_seed_layer(images, self.target_class)
        if max_features is None:
            max_features = ni[seed_layer].numel()
        if k_values is None:
            fracs = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
            k_values = sorted({max(1, round(f * max_features)) for f in fracs})

        results = {}
        for k in tqdm(k_values, desc="k-sweep"):
            circ = discoverer.build_circuit_from_matrices(ni, ei, za, top_k=k,
                                                          seed_layer=seed_layer, annotations=annotations)
            fv, cv = [], []
            for img in images:
                with torch.no_grad():
                    m_full = self._objective(self.model(img.to(self.device), self.task))
                m_empty = self._objective(self._run_patched(img, circ, "empty"))
                m_circ = self._objective(self._run_patched(img, circ, "faithfulness"))
                m_comp = self._objective(self._run_patched(img, circ, "completeness"))
                d = m_full - m_empty + 1e-8
                fv.append(min(max((m_circ - m_empty) / d, 0.0), 1.0))
                cv.append(1.0 - min(max((m_comp - m_empty) / d, 0.0), 1.0))
            results[k] = {"faithfulness": sum(fv) / len(fv), "completeness": sum(cv) / len(cv)}

        xs = [k / max_features for k in k_values]
        auc_f = _trapezoid(xs, [results[k]["faithfulness"] for k in k_values])
        auc_c = _trapezoid(xs, [results[k]["completeness"] for k in k_values])
        return results, auc_f, auc_c


def _trapezoid(xs, ys):
    if len(xs) < 2:
        return ys[0] if ys else 0.0
    return sum((xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(len(xs) - 1))