from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito / colorblind-safe palette.
PALETTE = {
    "direct": "#0072B2",       # blue
    "exploratory": "#D55E00",  # vermillion
    "randomized": "#7F7F7F",
    "reference": "#000000",
    "improvement": "#009E73",  # bluish green
}

METHOD_LABELS = {
    False: "Direct optimization",
    True: "Exploratory + optimization",
    "direct": "Direct optimization",
    "exploratory": "Exploratory + optimization",
}


def set_paper_style(font_size: float = 8.0, base_width_in: float = 3.35) -> None:
    """Set publication-oriented matplotlib defaults.

    Defaults target a single-column manuscript figure. All plots are exported as
    vector PDF/SVG plus high-DPI PNG by the figure saving helper.
    """
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.constrained_layout.use": True,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: Iterable[str] = ("pdf", "svg", "png")) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir / f"{stem}.{fmt}", bbox_inches="tight", facecolor="white")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
