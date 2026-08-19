from __future__ import annotations

import json
from functools import lru_cache

_NAME_COLUMN_CANDIDATES = (
    "class_name", "classname", "label_name", "labelname", "label_str", "label_text",
    "class", "category", "diagnosis", "dx", "lesion", "target_name", "name",
)


@lru_cache(maxsize=64)
def class_names_for_task(task: str, n: int) -> tuple[str, ...]:
    """``n`` class names in classifier-index order, or string indices on any miss."""
    try:
        names = _names_from_metadata(task)
    except Exception as exc:
        print(f"[labels] task {task!r}: falling back to indices ({exc})")
        names = []

    if len(names) == n:
        return tuple(names)
    if names:
        print(f"[labels] task {task!r}: metadata yields {len(names)} names but the head has "
              f"{n}; using indices to avoid a misaligned legend.")
    return tuple(str(i) for i in range(n))


def _names_from_metadata(task: str) -> list[str]:
    import pandas as pd
    from pandas.api.types import is_numeric_dtype

    from vitlab.datasets import get_spec as get_dataset_spec

    spec = get_dataset_spec(task)  # KeyError if task isn't a registered dataset
    if spec.multilabel:
        return []

    csv_path = spec.path("metadata")
    df = pd.read_csv(csv_path)
    col = spec.label_key
    if col not in df.columns:
        raise KeyError(f"label column {col!r} not in {csv_path.name} "
                       f"(columns: {list(df.columns)[:12]})")

    series = df[col].dropna()
    numeric_like = is_numeric_dtype(series) or bool(series.astype(str).str.fullmatch(r"-?\d+").all())
    if numeric_like:
        distinct = sorted(set(series.tolist()), key=lambda v: float(v))
    else:
        distinct = sorted(set(series.tolist()))

    # (1) textual labels are already the names
    if not numeric_like:
        return [str(v) for v in distinct]

    # (2) sibling text column that names each class 1:1
    name_col = _find_name_column(df, col)
    if name_col is not None:
        mapping = df.dropna(subset=[col, name_col]).groupby(col)[name_col].first()
        if all(v in mapping.index for v in distinct):
            print(f"[labels] task {task!r}: mapped {col!r} -> {name_col!r} for class names")
            return [str(mapping.loc[v]) for v in distinct]

    # (3) class_names.json sidecar next to the metadata CSV
    sidecar = _sidecar_names(csv_path.parent, distinct)
    if sidecar is not None:
        print(f"[labels] task {task!r}: used class_names.json sidecar")
        return sidecar

    print(f"[labels] task {task!r}: numeric label column {col!r} and no name source found. "
          f"Columns present: {list(df.columns)}. Add a name column, or drop a "
          f"class_names.json next to the metadata CSV, and I'll pick it up.")
    return [str(v) for v in distinct]


def _find_name_column(df, label_col: str) -> str | None:
    from pandas.api.types import is_numeric_dtype

    by_lower = {c.lower(): c for c in df.columns}

    def usable(cand: str) -> bool:
        if cand == label_col or is_numeric_dtype(df[cand]):
            return False
        per_label = df.dropna(subset=[label_col, cand]).groupby(label_col)[cand].nunique()
        return len(per_label) > 0 and per_label.max() == 1

    for key in _NAME_COLUMN_CANDIDATES:
        if key in by_lower and usable(by_lower[key]):
            return by_lower[key]
    for cand in df.columns:
        if usable(cand):
            return cand
    return None


def _sidecar_names(folder, distinct) -> list[str] | None:
    p = folder / "class_names.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except Exception:
        return None
    if isinstance(data, list) and len(data) == len(distinct):
        return [str(x) for x in data]
    if isinstance(data, dict):
        m = {str(k): v for k, v in data.items()}
        if all(str(v) in m for v in distinct):
            return [str(m[str(v)]) for v in distinct]
    return None