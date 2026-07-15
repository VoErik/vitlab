from __future__ import annotations

from pathlib import Path

DATA_ROOT: Path = Path("/home/voigt/data/")

def set_data_root(path: str | Path) -> Path:
    """Override DATA_ROOT for this process."""
    global DATA_ROOT
    DATA_ROOT = Path(path)
    return DATA_ROOT


def get_data_root(data_root: str | Path | None = None) -> Path:
    """Resolve the root: explicit argument wins, otherwise the module default."""
    root = Path(data_root) if data_root is not None else DATA_ROOT
    if not root.exists():
        raise FileNotFoundError(
            f"DATA_ROOT does not exist: {root}\n"
            f"Edit DATA_ROOT in vitlab/config.py, or call vitlab.set_data_root(...), "
            f"or pass data_root= to this function."
        )
    return root
