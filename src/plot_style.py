"""Shared figure style for the HAR notebooks (the same look as 01_eda).

    import plot_style as ps
    ps.apply_style()
    save = ps.saver("../figures/<notebook-name>")
    fig, ax = plt.subplots(figsize=(ps.COL_W, 2.3)); ...; save(fig, "name")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Two-series charts use blue/orange; heatmaps use a one-hue ramp;
# correlations use a blue-gray-red diverging map (gray = 0).
BLUE, ORANGE = "#2a78d6", "#eb6834"
LIGHT_GRAY = "#d3d1c7"
INK, INK2 = "#1f1f1e", "#5f5e58"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#2a78d6", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"])

COL_W, PAGE_W = 3.5, 7.16  # IEEE single- and double-column widths (inches)

SHORT = {"WALKING": "Walk", "WALKING_UPSTAIRS": "Upstairs", "WALKING_DOWNSTAIRS": "Downstairs",
         "SITTING": "Sit", "STANDING": "Stand", "LAYING": "Laying"}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 8, "axes.titlesize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False,
    })


def saver(fig_dir: str | Path):
    """Return save(fig, name), which writes name.pdf (for the paper) and name.png into fig_dir."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    def save(fig, name: str) -> None:
        for ext in ("pdf", "png"):
            fig.savefig(fig_dir / f"{name}.{ext}", bbox_inches="tight")

    return save
