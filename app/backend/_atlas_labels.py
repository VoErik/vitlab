from __future__ import annotations

import json
from pathlib import Path

from . import config


def _labels_path(atlas_id: str) -> Path:
    return config.atlas_dir() / atlas_id / "labels.json"


def _normalize(v) -> dict[str, str]:
    if isinstance(v, str):
        return {"name": v.strip(), "note": ""}
    if isinstance(v, dict):
        return {"name": str(v.get("name", "")).strip(), "note": str(v.get("note", "")).strip()}
    return {"name": "", "note": ""}


def load_labels(atlas_id: str) -> dict[str, dict]:
    """Return {feature_id(str): {"name","note"}}."""
    p = _labels_path(atlas_id)
    if p.exists():
        try:
            data = json.loads(p.read_text())
            return {str(k): _normalize(v) for k, v in data.items()}
        except Exception:
            return {}
    return {}


def save_label(atlas_id: str, feature: int, name: str, note: str = "") -> dict[str, dict]:
    """Set (or, when name and note are both empty, clear) one feature; returns the map."""
    labels = load_labels(atlas_id)
    key = str(int(feature))
    name = (name or "").strip()
    note = (note or "").strip()
    if name or note:
        labels[key] = {"name": name, "note": note}
    else:
        labels.pop(key, None)
    p = _labels_path(atlas_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(labels, indent=2))
    return labels