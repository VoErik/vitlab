from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from .circuit import Circuit, CircuitNode


def save_circuit(circuit, path, metadata=None):
    data = {
        "metadata": metadata or {},
        "nodes": {str(l): [asdict(n) for n in nodes] for l, nodes in circuit.nodes.items()},
        "edges": [{"u_layer": u.layer, "u_feat": u.feature_idx, "d_layer": d.layer,
                   "d_feat": d.feature_idx, "weight": float(w)} for u, d, w in circuit.edges],
    }
    Path(path).write_text(json.dumps(data, indent=2))


def load_circuit(path) -> Circuit:
    data = json.loads(Path(path).read_text())
    circuit = Circuit()
    lookup = {}
    for layer_str, node_list in data["nodes"].items():
        layer = int(layer_str)
        circuit.nodes[layer] = [CircuitNode(**nd) for nd in node_list]
        for nd in circuit.nodes[layer]:
            lookup[nd.key()] = nd
    for e in data["edges"]:
        u = lookup.get((e["u_layer"], e["u_feat"]))
        d = lookup.get((e["d_layer"], e["d_feat"]))
        if u and d:
            circuit.edges.append((u, d, e["weight"]))
    return circuit


def save_importance_matrices(node_imp, edge_imp, z_avg, path):
    torch.save({"node_imp": {l: t.cpu() for l, t in node_imp.items()},
                "edge_imp": {l: t.cpu() for l, t in edge_imp.items()},
                "z_avg": {l: t.cpu() for l, t in z_avg.items()}}, path)


def load_importance_matrices(path, device="cpu"):
    d = torch.load(path, map_location=device, weights_only=False)
    to = lambda x: {int(l): t.to(device) for l, t in x.items()}  # noqa: E731
    return to(d["node_imp"]), to(d["edge_imp"]), to(d["z_avg"])