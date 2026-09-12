"""
viz_style.py
------------
One place where every figure in this project gets its colors and chrome, so
plots read as a single system instead of as matplotlib defaults.

The categorical order is a validated colorblind-safe sequence: adjacent pairs
clear a CVD Delta-E floor in both light and dark modes, and the first three
slots clear it for ALL pairs, which is the requirement for scatter and
trajectory plots where any two series can end up adjacent. Assign slots in
fixed order and never cycle past the list.

Status colors (good/critical/...) are reserved for outcomes and threat state.
They are never used as "just another series" color, and anything they encode is
also carried by a label so meaning never rests on hue alone.
"""

import matplotlib as mpl

# --- categorical series, in fixed assignment order -------------------------
SERIES = ["#2a78d6",  # 1 blue
          "#eb6834",  # 2 orange
          "#1baf7a",  # 3 aqua
          "#eda100",  # 4 yellow
          "#e87ba4",  # 5 magenta
          "#008300",  # 6 green
          "#4a3aa7",  # 7 violet
          "#e34948"]  # 8 red

# --- reserved status colors -------------------------------------------------
GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"

# --- surfaces and ink -------------------------------------------------------
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Semantic roles used across this project's figures
ROLE_COLORS = {
    "ESCORT": SERIES[0],
    "JAMMER": SERIES[1],
    "DECOY": SERIES[2],
    "INTERCEPT": SERIES[3],
}
MOTHERSHIP = INK
THREAT = CRITICAL


def apply():
    """Install the project chart style. Call once before plotting."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "medium",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",        # solid hairlines; dashed grids read as thresholds
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_SECONDARY,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "figure.dpi": 150,
    })
