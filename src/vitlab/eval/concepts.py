from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class ConceptLabels:
    Y: np.ndarray # (N_images, n_concepts) binary
    names: list[str]
    image_ids: list[str]

    def __len__(self) -> int:
        return self.Y.shape[0]

    @property
    def n_concepts(self) -> int:
        return len(self.names)


def reconstruct_image_ids(
    dataset,
    split: str,
    *,
    label_key: str = "label",
    path_col: str = "filepath",
) -> list[str]:
    """
    Reconstruct the concept-CSV ids for a dataset's rows, in dataset order.

    The concept CSV keys each row as ``{split}/{class}/{filename}`` (e.g.
    ``train/basal_cell_carcinoma/ba-ce-ca_f3_218_bb3d0878.jpg``):

        split    -- passed in (which split this dataset is)
        class    -- the `label` column, whitespace joined by "_"
                    ("basal cell carcinoma" -> "basal_cell_carcinoma")
        filename -- basename of the `filepath` column (the original-dataset relic
                    path differs in its directory part but shares the filename)

    Read directly from the raw columns (bypassing the image transform), so pairing
    with an in-order (shuffle=False) encode gives an exact, unambiguous join.
    """
    cols = getattr(dataset, "column_names", [])
    for c in (path_col, label_key):
        if c not in cols:
            raise ValueError(
                f"dataset has no {c!r} column (needed to rebuild concept ids); "
                f"columns are {list(cols)[:12]}..."
            )
    fmt = dataset.format
    dataset.set_format(type=None, columns=[path_col, label_key])
    paths = list(dataset[path_col])
    labels = list(dataset[label_key])
    dataset.set_format(**fmt)

    ids = []
    for p, lab in zip(paths, labels):
        cls = "_".join(str(lab).split())
        ids.append(f"{split}/{cls}/{Path(str(p)).name}")
    return ids


def load_concepts(
    dataset_name: str,
    *,
    data_root: str | Path | None = None,
    concept_prefix: str = "has_",
    id_column: str | None = None,
) -> ConceptLabels:
    """
    Read the concept CSV declared for `dataset_name` in the registry.

    Concept columns are every column starting with `concept_prefix` (default
    "has_"). The id column (whatever keys a row to an image filename) is 
    auto-detected unless you pass `id_column`.
    """
    import pandas as pd

    from ..config import get_data_root
    from ..datasets import get_spec

    spec = get_spec(dataset_name)
    if spec.concepts is None:
        raise ValueError(f"{dataset_name} has no concept CSV in the registry")
    csv_path = get_data_root(data_root) / spec.concepts
    df = pd.read_csv(csv_path)

    concept_cols = sorted(c for c in df.columns if c.startswith(concept_prefix))
    if not concept_cols:
        raise ValueError(
            f"no columns starting with {concept_prefix!r} in {csv_path.name}. "
            f"Columns: {list(df.columns)[:12]}..."
        )

    if id_column is None:
        for cand in ("file_name", "filename", "image", "image_id", "id", "name"):
            if cand in df.columns:
                id_column = cand
                break
    ids = df[id_column].astype(str).tolist() if id_column else [str(i) for i in range(len(df))]

    Y = df[concept_cols].to_numpy().astype(np.float32)
    names = [c[len(concept_prefix):] for c in concept_cols]
    return ConceptLabels(Y=Y, names=names, image_ids=ids)


def align_labels_to_loader(labels: ConceptLabels, image_ids_in_order: list[str]) -> np.ndarray:
    """
    Reorder label rows to match the order a loader yielded images.
    """
    index = {img_id: i for i, img_id in enumerate(labels.image_ids)}
    missing = [i for i in image_ids_in_order if i not in index]
    if missing:
        raise KeyError(
            f"{len(missing)} image ids have no concept row (e.g. {missing[:3]}). "
            f"Check id_column / that the CSV covers this split."
        )
    rows = [index[i] for i in image_ids_in_order]
    return labels.Y[rows]


@torch.no_grad()
def encode_dataset(
    layer_sae,
    reader,
    loader,
    site: str,
    *,
    device: str = "cuda",
    id_key: str | None = None,
    return_ids: bool = False,
):
    """
    Mean-pooled per-image SAE codes over a dataset.
    """
    reader.backbone.to(device).eval()
    layer_sae.to(device)
    prefix = reader.spec.n_prefix_tokens

    codes_out: list[np.ndarray] = []
    ids_out: list[str] = []
    counter = 0

    for batch in loader:
        px = batch["pixel_values"] if isinstance(batch, dict) else batch[0]
        px = px.to(device)
        B = px.shape[0]

        acts = reader.read(px, site)
        patches = acts[:, prefix:, :]
        Bp, P, Dm = patches.shape
        codes = layer_sae.encode(patches.reshape(Bp * P, Dm))
        pooled = codes.reshape(Bp, P, -1).mean(1)
        codes_out.append(pooled.cpu().numpy())

        if return_ids:
            if isinstance(batch, dict) and id_key and id_key in batch:
                ids_out.extend(str(x) for x in batch[id_key])
            else:
                ids_out.extend(str(counter + i) for i in range(B))
        counter += B

    X = np.concatenate(codes_out, axis=0)
    return (X, ids_out) if return_ids else X


def _feature_capacity(X, y, seed=42):
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(max_depth=1, random_state=seed, class_weight="balanced")
    tree.fit(X, y)
    pred = tree.predict(X)
    best = int(tree.tree_.feature[0]) if tree.tree_.node_count > 1 else 0
    return balanced_accuracy_score(y, pred), f1_score(y, pred, zero_division=0), best


def _local_disentanglement(X, y, base, best_idx, use_f1, seed=42):
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.tree import DecisionTreeClassifier

    Xl = X.copy()
    Xl[:, best_idx] = 0
    tree = DecisionTreeClassifier(max_depth=1, random_state=seed, class_weight="balanced")
    tree.fit(Xl, y)
    pred = tree.predict(Xl)
    score = f1_score(y, pred, zero_division=0) if use_f1 else balanced_accuracy_score(y, pred)
    return max(0.0, 2 * (base - score))


def _global_disentanglement(X, y, base, max_steps, use_f1, seed=42):
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.tree import DecisionTreeClassifier

    cum = 0.0
    for depth in range(2, max_steps + 1):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=seed, class_weight="balanced")
        tree.fit(X, y)
        pred = tree.predict(X)
        score = f1_score(y, pred, zero_division=0) if use_f1 else balanced_accuracy_score(y, pred)
        cum += max(0.0, score - base)
    return 1.0 - cum / max(1, max_steps - 1)


def fms_metrics(X, Y, concept_names, *, max_global_steps=5, use_f1=False) -> dict:
    """Feature Monosemanticity Score per concept + means. X: (N, F) codes,
    Y: (N, n_concepts) binary."""
    out = {"concepts": {}, "fms_strict_mean": 0.0, "fms_relaxed_mean": 0.0}
    strict, relaxed = [], []
    for i, name in enumerate(concept_names):
        y = Y[:, i]
        if len(np.unique(y)) < 2:
            continue
        acc0, f1_0, best = _feature_capacity(X, y)
        factor = f1_0 if use_f1 else acc0
        local = _local_disentanglement(X, y, factor, best, use_f1)
        glob = _global_disentanglement(X, y, factor, max_global_steps, use_f1)
        score_strict = factor * (local + glob) / 2
        out["concepts"][name] = {
            "accs_0": acc0, "f1_0": f1_0, "fms_local": local,
            "fms_global": glob, "score_strict": score_strict, "best_feature_idx": best,
        }
        strict.append(score_strict)
        relaxed.append(f1_0)
    out["fms_strict_mean"] = float(np.mean(strict)) if strict else 0.0
    out["fms_relaxed_mean"] = float(np.mean(relaxed)) if relaxed else 0.0
    return out

def purity_metrics(X, Y, concept_names, *, active_eps: float = 1e-3) -> dict:
    """Per-concept best-feature precision/F1 (depth-1 tree) + global feature
    entropy over concept co-occurrence."""
    from scipy.stats import entropy as scipy_entropy
    from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score
    from sklearn.tree import DecisionTreeClassifier

    out: dict = {"concepts": {}}
    for i, name in enumerate(concept_names):
        y = Y[:, i]
        if y.sum() == 0:
            continue
        tree = DecisionTreeClassifier(max_depth=1, random_state=42, class_weight="balanced")
        tree.fit(X, y)
        best = int(tree.tree_.feature[0]) if tree.tree_.node_count > 1 else 0
        thr = tree.tree_.threshold[0] if tree.tree_.node_count > 1 else 0.0
        pred = (X[:, best] > thr).astype(int)
        if np.mean(pred == y) < 0.5:
            pred = 1 - pred
        out["concepts"][name] = {
            "best_feature": best,
            "precision": float(precision_score(y, pred, zero_division=0)),
            "f1_score": float(f1_score(y, pred, zero_division=0)),
            "balanced_acc": float(balanced_accuracy_score(y, pred)),
        }

    Xb = (X > active_eps).astype(np.float32)
    activity = Xb.sum(0)
    active = activity > 0
    if active.sum() == 0:
        out["feature_entropy_median_active"] = float("nan")
        return out
    cooc = Xb[:, active].T @ Y
    probs = cooc / (cooc.sum(1, keepdims=True) + 1e-8)
    ent = scipy_entropy(probs, axis=1)
    out["feature_entropy_median_active"] = float(np.median(ent))
    return out

def linear_probes(X, Y, concept_names, *, test_size=0.2, seed=42) -> dict:
    """Per-concept logistic-regression decodability from the codes."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    results: dict = {}
    for i, name in enumerate(concept_names):
        y = Y[:, i]
        vals, counts = np.unique(y, return_counts=True)
        if len(vals) < 2 or counts.min() < 2:
            continue
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, solver="liblinear")
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        results[name] = {
            "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
            "f1_score": float(f1_score(yte, pred, average="macro")),
        }
    if results:
        results["_mean_balanced_accuracy"] = float(
            np.mean([r["balanced_accuracy"] for k, r in results.items() if not k.startswith("_")])
        )
    return results