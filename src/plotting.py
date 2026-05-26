"""
Shared matplotlib style for all notebooks in this project.

Why a separate module:
    Calling `set_style()` at the top of each notebook gives consistent look
    across the three notebooks (01_data_exploration, 02_price_model_calibration,
    03_option_valuation), so the project reads as one document. The PALETTE
    dict gives semantic color names — `PALETTE["highlight"]` always means
    "anomaly / annotation", whatever year I revisit this.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


# Semantic color palette. Colorblind-friendly and distinguishable in greyscale
# (useful if the report ever gets printed).
PALETTE: dict[str, str] = {
    "primary":   "#1f4e79",  # deep blue — main observed series
    "secondary": "#c97d24",  # warm orange — fitted / model overlay
    "season":    "#2e7d4f",  # green — seasonal component
    "highlight": "#b03a2e",  # red — anomalies, annotations, density refs
    "neutral":   "#7f7f7f",  # grey — context lines, zero markers
}


def set_style() -> None:
    """
    Apply consistent matplotlib rcParams for the project.

    Call once near the top of each notebook (after importing matplotlib).
    Idempotent — safe to call multiple times.
    """
    plt.rcParams.update({
        # Sizes and DPI — readable both inline in Jupyter and when saved to PNG
        "figure.figsize":  (10, 4.5),
        "figure.dpi":      100,
        "savefig.dpi":     150,
        "savefig.bbox":    "tight",

        # Typography — keep defaults but bump sizes slightly for readability
        "font.size":       10,
        "axes.titlesize":  12,
        "axes.titleweight": "bold",
        "axes.labelsize":  10,

        # Spines: drop top + right for a cleaner look (à la Tufte)
        "axes.spines.top":   False,
        "axes.spines.right": False,

        # Grid: subtle dashed grid, gridded plots are easier to read
        "axes.grid":      True,
        "grid.alpha":     0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,

        # Legends: no frame, slightly smaller text
        "legend.frameon":  False,
        "legend.fontsize": 9,
    })
