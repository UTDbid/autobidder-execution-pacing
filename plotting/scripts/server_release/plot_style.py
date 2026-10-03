from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived"
MAIN = ROOT / "figures" / "main"
EC = ROOT / "figures" / "ec"
QA = ROOT / "qa"

INK = "#20262E"
MUTED = "#68737D"
GRID = "#D9E0E6"
NEG = "#A64B4B"
POS = "#267A68"
REF_COLORS = {"traffic_aware": "#356EA8", "traffic": "#356EA8", "pid": "#D97721", "dual": "#2A8C82", "smoothed_controller": "#7A63A8", "response_aware": "#B45A86"}
REF_LABELS = {"traffic_aware": "Traffic-aware", "traffic": "Traffic-aware", "pid": "PID pacing", "dual": "Dual pacing", "smoothed": "Smoothed", "response": "Response-aware", "smoothed_controller": "Smoothed-controller", "response_aware": "Response-aware"}
REF_MARKERS = {"traffic_aware": "o", "traffic": "o", "pid": "s", "dual": "D", "smoothed_controller": "^", "response_aware": "v"}
REF_STYLES = {"traffic_aware": "-", "traffic": "-", "pid": "--", "dual": "-.", "smoothed_controller": ":", "response_aware": (0, (3, 1, 1, 1))}


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10.8,
        "axes.labelsize": 11.2,
        "axes.titlesize": 11.2,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 10.3,
        "legend.frameon": False,
        "lines.linewidth": 1.7,
        "lines.markersize": 5.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })
    MAIN.mkdir(parents=True, exist_ok=True)
    EC.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)


def clean(ax, grid_axis="y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)


def panel(ax, label: str, subtitle: str = "") -> None:
    """Register one centered panel subtitle above the plotted panel."""
    ax._panel_heading = f"({label.lower()}) {subtitle}".strip()


def place_panel_subtitles(fig) -> None:
    fig.canvas.draw()
    gap = 7 / 72 / fig.get_size_inches()[1]
    for ax in fig.axes:
        heading = getattr(ax, "_panel_heading", None)
        if heading is None:
            continue
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2, box.y1 + gap, heading,
                 ha="center", va="bottom", fontsize=10.6, color=INK)


def save(fig, stem: str, section: str) -> None:
    out = MAIN if section == "main" else EC
    place_panel_subtitles(fig)
    suffix = "_above" if section == "ec" else "_below"
    for ext in ["pdf", "svg"]:
        fig.savefig(out / f"{stem}{suffix}.{ext}", bbox_inches="tight", pad_inches=0.07)
    fig.savefig(out / f"{stem}{suffix}.png", dpi=320, bbox_inches="tight", pad_inches=0.07)
    plt.close(fig)
