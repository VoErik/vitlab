from __future__ import annotations

import logging
from pathlib import Path

_LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}


def setup_logger(run_dir: str | Path, *, name: str = "vitlab", level_str: str = "INFO",
                 filename: str = "run.log") -> logging.Logger:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    level = _LEVELS.get(str(level_str).upper(), logging.INFO)

    logger = logging.getLogger(f"{name}.{run_dir}")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(run_dir / filename)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger
