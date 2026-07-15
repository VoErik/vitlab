from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

def seed_everything(seed: int = 0, *, deterministic: bool = False) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def load_dotenv(
    path: str | Path | None = None,
    *,
    override: bool = False,
    search_parents: bool = True,
) -> dict[str, str]:
    """
    Populate os.environ from a .env file. Returns the keys it set.
    By default it does NOT override variables already in the environment. Set `override=True` to flip that.
    """
    if path is None:
        path = _find_dotenv() if search_parents else Path(".env")
    else:
        path = Path(path)

    if path is None or not path.exists():
        return {}

    set_keys: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
            set_keys[key] = value
    return set_keys


def _find_dotenv(start: Path | None = None, filename: str = ".env") -> Path | None:
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / filename
        if candidate.exists():
            return candidate
    return None


