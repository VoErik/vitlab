from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

_STYLE_PATH = Path(__file__).with_name("thesis.mplstyle") # TODO: read from project root


def use_style() -> None:
    plt.style.use(str(_STYLE_PATH))


def finish(fig, save: str | Path | None = None, *, show: bool = False):
    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save)
    if show:
        plt.show()
    return fig
