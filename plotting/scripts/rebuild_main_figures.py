"""Rebuild the eight main-text figures for the POMS manuscript.

Chart contract
--------------
The main-text figures communicate structure and comparative patterns.  They
therefore display point estimates or paired means only; confidence intervals,
seed dispersion, and episode uncertainty remain in tables and the
E-Companion.  All figures use Times New Roman, the same pacing palette, and
non-color encodings (line style and marker) wherever pacing series appear.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "main"
DATA = ROOT / "data" / "derived" / "main"
LOAD = ROOT / "data" / "load"
CALIBRATION = ROOT / "data" / "server_checked"

INK = "#1F2933"
MUTED = "#6B7785"
GRID = "#D9E0E6"
BLUE = "#356EA8"
ORANGE = "#D97721"
TEAL = "#2A8C82"
LIGHT_BLUE = "#EAF1F8"
LIGHT_ORANGE = "#FBF0E4"
LIGHT_TEAL = "#E7F2F0"

REFS = ["traffic_aware", "pid", "dual"]
REFS_MACRO = ["traffic", "pid", "dual"]
LABEL = {
    "traffic_aware": "Traffic-aware",
    "traffic": "Traffic-aware",
    "pid": "PID pacing",
    "dual": "Dual pacing",
}
COLOR = {
    "traffic_aware": BLUE,
    "traffic": BLUE,
    "pid": ORANGE,
    "dual": TEAL,
}
STYLE = {
    "traffic_aware": "-",
    "traffic": "-",
    "pid": "--",
    "dual": "-.",
}
MARKER = {
    "traffic_aware": "o",
    "traffic": "o",
    "pid": "s",
    "dual": "D",
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11.5,
            "axes.titlesize": 13.0,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 12.5,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.unicode_minus": True,
        }
    )


def clean(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.tick_params(direction="out", length=3.5, colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.9)
        ax.set_axisbelow(True)


def panel(ax: plt.Axes, letter: str, subtitle: str, *, fontsize: float = 12.5) -> None:
    """Register a centered panel subtitle above the plot area."""
    ax._panel_heading = (f"({letter.lower()}) {subtitle}", fontsize)


def place_panel_subtitles(fig: plt.Figure) -> None:
    fig.canvas.draw()
    gap_px = fig.dpi * 11 / 72  # eleven printed points above the plot area
    inverse = fig.transFigure.inverted()
    for ax in fig.axes:
        heading = getattr(ax, "_panel_heading", None)
        if heading is None:
            continue
        center_px, top_px = ax.transAxes.transform((0.5, 1.0))
        x_fig, y_fig = inverse.transform((center_px, top_px + gap_px))
        fig.text(x_fig, y_fig, heading[0], ha="center", va="bottom",
                 fontsize=heading[1], fontweight="normal", color=INK)


def save(fig: plt.Figure, filename: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    place_panel_subtitles(fig)
    stem = Path(filename).stem
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    lines: list[str],
    *,
    face: str = "white",
    edge: str = "#7A8793",
    title_color: str = INK,
    line_fontsize: float = 9.5,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.1,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.025 * width,
        y + height - 0.20 * height,
        title,
        ha="left",
        va="center",
        fontsize=11.3,
        fontweight="bold",
        color=title_color,
    )
    if len(lines) == 1:
        positions = [0.62]
    elif len(lines) == 2:
        positions = [0.52, 0.78]
    else:
        positions = np.linspace(0.43, 0.90, len(lines))
    for line, position in zip(lines, positions):
        ax.text(
            x + 0.05 * width,
            y + height - position * height,
            line,
            ha="left",
            va="center",
            fontsize=line_fontsize,
            color=INK,
        )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], **kwargs) -> None:
    defaults = dict(
        arrowstyle="-|>", mutation_scale=11, linewidth=1.0, color=MUTED,
        connectionstyle="arc3,rad=0.0"
    )
    defaults.update(kwargs)
    ax.add_patch(FancyArrowPatch(start, end, **defaults))


def figure1() -> None:
    """Layered control architecture; no research-roadmap content."""
    fig, ax = plt.subplots(figsize=(10.6, 5.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(
        ax,
        (0.03, 0.62),
        0.25,
        0.25,
        "Campaign-level control",
        ["Traffic-aware / PID / dual", "Budget and delivery state", r"Reference action  $a^{ref}_{it}$"],
        face=LIGHT_BLUE,
        edge=BLUE,
    )
    add_box(
        ax,
        (0.03, 0.26),
        0.25,
        0.25,
        "Opportunity-level autobidder",
        ["Current opportunity and campaign state", "Native bidder recommendation", r"Native action  $a^{raw}_{it}$"],
        face=LIGHT_TEAL,
        edge=TEAL,
    )

    add_box(
        ax,
        (0.35, 0.37),
        0.32,
        0.36,
        "Execution boundary",
        [
            r"Permission width  $\kappa$",
            r"$a^{exec}_{it}=\mathrm{clip}\!\left(a^{raw}_{it},\,(1-\kappa)a^{ref}_{it},\,(1+\kappa)a^{ref}_{it}\right)$",
            "Native action passes through inside the boundary",
        ],
        face=LIGHT_ORANGE,
        edge=ORANGE,
        title_color="#A65E17",
        line_fontsize=8.2,
    )

    add_box(
        ax,
        (0.74, 0.51),
        0.22,
        0.21,
        "Shared auction",
        [r"Submitted bid uses  $a^{exec}_{it}$", "Allocation and payment"],
        face="white",
        edge="#7A8793",
    )
    add_box(
        ax,
        (0.74, 0.18),
        0.22,
        0.21,
        "Realized outcomes",
        ["Impressions, cost, conversions", "Remaining budget and delivery state"],
        face="#F5F6F7",
        edge="#7A8793",
    )

    arrow(ax, (0.28, 0.745), (0.36, 0.62))
    arrow(ax, (0.28, 0.385), (0.36, 0.49))
    arrow(ax, (0.65, 0.55), (0.74, 0.615), color=BLUE)
    arrow(ax, (0.85, 0.51), (0.85, 0.39))
    arrow(
        ax,
        (0.76, 0.18),
        (0.20, 0.26),
        connectionstyle="arc3,rad=-0.19",
        color=MUTED,
    )
    arrow(
        ax,
        (0.81, 0.18),
        (0.20, 0.62),
        connectionstyle="arc3,rad=-0.30",
        color=MUTED,
    )
    ax.text(
        0.44,
        0.055,
        "Closed-loop feedback updates bidder and campaign states",
        ha="center",
        va="center",
        fontsize=9.5,
        color=MUTED,
        style="italic",
    )
    save(fig, "figure1_layered_control.pdf")


def figure2() -> None:
    """Research questions and evidence design without redundant numbering."""
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(
        ax,
        (0.36, 0.77),
        0.28,
        0.16,
        "Specialized control layers",
        ["Autobidder: opportunity information", "Campaign control: intertemporal delivery"],
        face="#F5F7F8",
        edge="#536574",
    )
    add_box(
        ax,
        (0.36, 0.48),
        0.28,
        0.16,
        "Execution coordination",
        ["Coordination rule", "Execution width"],
        face=LIGHT_ORANGE,
        edge=ORANGE,
    )
    add_box(
        ax,
        (0.36, 0.19),
        0.28,
        0.16,
        "Shared market",
        ["Adoption scale", "Competition, prices, revenue, spillovers"],
        face=LIGHT_TEAL,
        edge=TEAL,
    )
    arrow(ax, (0.50, 0.77), (0.50, 0.64))
    arrow(ax, (0.50, 0.48), (0.50, 0.35))

    add_box(
        ax,
        (0.02, 0.48),
        0.30,
        0.28,
        "RQ1: Campaign-level coordination",
        [
            "How do coordination rule and authority",
            "shape the value-delivery frontier?",
            "Campaign value and delivery",
        ],
        face="white",
        edge=BLUE,
        title_color="#214E74",
    )
    arrow(ax, (0.36, 0.56), (0.32, 0.60), color=BLUE)

    add_box(
        ax,
        (0.68, 0.51),
        0.30,
        0.43,
        "RQ2: Operating conditions",
        [
            "When and why does discretion create value?",
            "Information and selection",
            "Does bidder information improve opportunity selection?",
            "Campaign load and redeployability",
            "Does load leave room to redeploy released budget?",
        ],
        face="white",
        edge=TEAL,
        title_color="#165E59",
        line_fontsize=8.7,
    )
    arrow(ax, (0.64, 0.84), (0.68, 0.84), color=TEAL)

    add_box(
        ax,
        (0.68, 0.07),
        0.30,
        0.29,
        "RQ3: Marketplace deployment",
        [
            "How do execution width and adoption scale",
            "affect campaign and market outcomes?",
            "Market-wide deployment",
        ],
        face="white",
        edge=ORANGE,
        title_color="#A65E17",
    )
    arrow(ax, (0.64, 0.27), (0.68, 0.27), color=ORANGE)

    save(fig, "figure2_research_design.pdf")


def figure3() -> None:
    d = pd.read_csv(DATA / "figure2a_campaign_frontier.csv")
    width_order = ["0.0", "0.3", "0.5", "0.8", "1.0", "1.2", "Raw"]
    for ref in REFS:
        assert set(d.loc[d.reference.eq(ref), "width"]) == set(width_order), ref
    width_x = {w: i for i, w in enumerate(width_order)}

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.15))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.82, wspace=0.28)
    a, b = axes

    for ref in REFS:
        q = d[d.reference.eq(ref)].copy()
        q["x"] = q.width.map(width_x)
        q = q.sort_values("x")
        finite = q[q.width.ne("Raw")]
        a.plot(
            finite.x,
            finite.mean_relative_value_delta_pct,
            color=COLOR[ref],
            linestyle=STYLE[ref],
            marker=MARKER[ref],
            markerfacecolor="white",
            markeredgewidth=1.2,
            label=LABEL[ref],
        )
        raw = q[q.width.eq("Raw")]
        a.scatter(
            raw.x,
            raw.mean_relative_value_delta_pct,
            color=COLOR[ref],
            marker=MARKER[ref],
            facecolors="white",
            s=42,
            linewidth=1.2,
            zorder=4,
        )
        wide = q[q.width.eq("0.8")]
        a.scatter(wide.x, wide.mean_relative_value_delta_pct, color=COLOR[ref], s=48, zorder=5)

        b.plot(
            finite.mean_underdelivery_delta_pp,
            finite.mean_relative_value_delta_pct,
            color=COLOR[ref],
            linestyle=STYLE[ref],
            marker=MARKER[ref],
            markerfacecolor="white",
            markeredgewidth=1.2,
        )
        b.scatter(
            raw.mean_underdelivery_delta_pp,
            raw.mean_relative_value_delta_pct,
            color=COLOR[ref],
            marker=MARKER[ref],
            facecolors="white",
            s=42,
            linewidth=1.2,
            zorder=4,
        )
        b.scatter(
            wide.mean_underdelivery_delta_pp,
            wide.mean_relative_value_delta_pct,
            color=COLOR[ref],
            s=48,
            zorder=5,
        )

    a.axhline(0, color=INK, linewidth=0.75)
    a.axvline(width_x["1.0"], color=MUTED, linewidth=0.8, linestyle=":")
    a.set_xticks(range(len(width_order)))
    a.set_xticklabels(["0", "0.3", "0.5", "0.8", "1.0", "1.2", "Raw"])
    a.set_xlabel(r"Execution width $\kappa$")
    a.set_ylabel("Relative advertiser value change (%)")
    clean(a)
    panel(a, "a", "Advertiser value (execution width)")

    b.axhline(0, color=INK, linewidth=0.75)
    b.set_xlabel("Change in underdelivery (percentage points)")
    b.set_ylabel("Relative advertiser value change (%)")
    clean(b)
    panel(b, "b", "Value–delivery paths")

    # Label only the three structurally important locations on one trajectory.
    annotations = [
        ("traffic_aware", "0.8", "Wide", (4, 6)),
        ("traffic_aware", "1.0", r"$\kappa=1$", (6, -13)),
        ("dual", "Raw", "Raw", (6, -10)),
    ]
    for ref, width, text, offset in annotations:
        row = d[(d.reference.eq(ref)) & (d.width.eq(width))].iloc[0]
        b.annotate(
            text,
            (row.mean_underdelivery_delta_pp, row.mean_relative_value_delta_pct),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            color=INK,
        )

    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    save(fig, "fig2_campaign_frontier.pdf")


def figure4() -> None:
    direction = pd.read_csv(DATA / "figure4a_directional.csv")
    tv = pd.read_csv(DATA / "figure4b_t9_true_value.csv")
    eff = pd.read_csv(DATA / "figure4c_t9_purchase_efficiency.csv")
    direction["reference"] = direction.reference.replace({"traffic": "traffic_aware"})

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.10))
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.22, top=0.79, wspace=0.34)
    a, b, c = axes

    conds = ["down_only", "up_only", "symmetric"]
    x = np.arange(len(conds))
    bar_w = 0.23
    for i, ref in enumerate(REFS):
        q = direction[direction.reference.eq(ref)].set_index("condition").reindex(conds)
        a.bar(
            x + (i - 1) * bar_w,
            q.value_delta / 1000,
            width=bar_w,
            color=COLOR[ref],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.5,
            label=LABEL[ref],
        )
    a.axhline(0, color=INK, linewidth=0.75)
    a.set_xticks(x)
    a.set_xticklabels(["Down only", "Up only", "Symmetric"], rotation=17, ha="right")
    a.set_ylabel(r"Advertiser value change ($\times 10^3$)")
    clean(a)
    panel(a, "a", "Directional execution (AuctionNet)")

    views = ["C1", "C2", "C3", "C4"]
    x = np.arange(len(views))
    offsets = {"traffic_aware": -0.16, "pid": 0.0, "dual": 0.16}
    for ax, data, col, ylabel in [
        (b, tv, "true_value_delta", r"True expected-value change ($\times 10^3$)"),
        (c, eff, "exclusive_efficiency_delta", "Wide-only minus Base-only\nthreshold-normalized truth value"),
    ]:
        for ref in REFS:
            q = data[data.reference.eq(ref)].set_index("information_view").reindex(views)
            y = q[col].to_numpy() / 1000 if col == "true_value_delta" else q[col].to_numpy()
            ax.scatter(
                x + offsets[ref],
                y,
                color=COLOR[ref],
                marker=MARKER[ref],
                facecolors="white",
                linewidth=1.3,
                s=42,
                zorder=3,
            )
        ax.axhline(0, color=INK, linewidth=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels(views)
        ax.set_xlabel("Information view")
        ax.set_ylabel(ylabel)
        clean(ax)
    panel(b, "b", "True campaign value (T9Sim)")
    panel(c, "c", "Marginal purchase quality (T9Sim)")

    handles = [
        plt.Line2D(
            [], [], color=COLOR[r], marker=MARKER[r], linestyle=STYLE[r],
            markerfacecolor="white", label=LABEL[r]
        )
        for r in REFS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    save(fig, "fig3_directional_discretion.pdf")


def figure5() -> None:
    supply = pd.read_csv(DATA / "figure5a_auctionnet_supply.csv")
    pressure = pd.read_csv(DATA / "figure5b_ipinyou_pressure.csv")

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 4.10))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.24, top=0.83, wspace=0.28)
    order = ["500", "2000", "8000", "full"]
    for ref in REFS:
        q = supply[supply.reference.eq(ref)].copy()
        q["order"] = q.opportunity_supply.astype(str).map({v: i for i, v in enumerate(order)})
        q = q.sort_values("order")
        a.plot(
            range(4),
            q.value_delta / 1000,
            color=COLOR[ref],
            linestyle=STYLE[ref],
            marker=MARKER[ref],
            markerfacecolor="white",
            markeredgewidth=1.2,
            label=LABEL[ref],
        )
    a.axhline(0, color=INK, linewidth=0.75)
    a.set_xticks(range(4))
    a.set_xticklabels(["500", "2,000", "8,000", "Full"])
    a.set_xlabel("Available opportunities")
    a.set_ylabel(r"Advertiser value change ($\times 10^3$)")
    clean(a)
    panel(a, "a", "Supply pressure (AuctionNet)")

    for ref in REFS:
        q = pressure[pressure.reference.eq(ref)].sort_values("budget_fraction")
        b.plot(
            100 * q.budget_fraction,
            q.clicks_delta,
            color=COLOR[ref],
            linestyle=STYLE[ref],
            marker=MARKER[ref],
            markerfacecolor="white",
            markeredgewidth=1.2,
        )
    b.axhline(0, color=INK, linewidth=0.75)
    b.set_xscale("log", base=2)
    b.set_xticks([3.125, 12.5, 50])
    b.set_xticklabels(["3.125%", "12.5%", "50%"])
    b.set_xlabel("Budget / logged spend")
    b.set_ylabel("Mean click change per campaign")
    clean(b)
    panel(b, "b", "Spending pressure (iPinYou)")

    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    save(fig, "fig4_supply_budget_pressure.pdf")


def figure6() -> None:
    d = pd.read_csv(LOAD / "development_load_quintiles.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 4.10))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.22, top=0.86, wspace=0.28)
    x = d.load_quintile.to_numpy()
    y = d.mean_value_gain_kappa_0_8.to_numpy() / 1000
    colors = [BLUE if v >= 0 else "#A8BECE" for v in y]
    a.bar(x, y, color=colors, edgecolor=INK, linewidth=0.55, width=0.58)
    a.axhline(0, color=INK, linewidth=0.8)
    for xi, yi in zip(x, y):
        a.text(xi, yi + (0.12 if yi >= 0 else -0.12), f"{yi:+.2f}", ha="center", va="bottom" if yi >= 0 else "top", fontsize=9.2, color=INK)
    a.set_xticks(x)
    a.set_ylim(-1.7, 4.3)
    a.set_xlabel("Campaign-load quintile (low to high)")
    a.set_ylabel(r"Mean value change at $\kappa=0.8$ ($\times 10^3$)")
    clean(a)
    panel(a, "a", r"Value effect (fixed $\kappa=0.8$)")

    # Quintile 4 has a genuine selected width of zero; quintile 5 has no
    # service-feasible scenario and is intentionally left unmarked.
    feasible = d[d.load_quintile.le(4)].copy()
    for row in feasible.itertuples():
        open_marker = row.load_quintile == 4
        if not open_marker:
            b.vlines(row.load_quintile, 0, row.mean_selected_kappa, color="#A8C4D8", linewidth=2.0)
        b.scatter(
            row.load_quintile,
            row.mean_selected_kappa,
            s=54,
            facecolors="white" if open_marker else BLUE,
            edgecolors=INK,
            linewidth=1.0,
            zorder=3,
        )
        b.text(row.load_quintile, row.mean_selected_kappa + 0.045, f"{row.mean_selected_kappa:.2f}", ha="center", fontsize=9.2, color=INK)
    b.text(5, 0.16, "No feasible\nscenarios", ha="center", va="center", fontsize=9.5, color=MUTED)
    b.set_xlim(0.55, 5.45)
    b.set_ylim(-0.05, 0.88)
    b.set_xticks(x)
    b.set_xlabel("Campaign-load quintile (low to high)")
    b.set_ylabel(r"Mean selected width $\kappa$")
    clean(b)
    panel(b, "b", "Selected width (10-pp delivery screen)")
    save(fig, "fig5_campaign_load.pdf")




def figure7() -> None:
    """Line maps and exact later-period means from server-verified CSV files."""
    selected = pd.read_csv(CALIBRATION / "development_selection_maps.csv")
    later = pd.read_csv(CALIBRATION / "later_frozen_policy_comparison.csv")
    selected = selected[(selected.reference == "pid") & (selected.incremental_cap_pp == 10.0)]
    later = later[later.reference == "pid"]
    bidders = {"CQL": "cql", "IQL": "iql", "SemBid": "sembid"}
    supplies = {"Full": (BLUE, "-", "o"), "8k": (ORANGE, "--", "s"),
                "4k": (TEAL, "-.", "D"), "2k": (MUTED, ":", "^")}
    supply_cap = {"Full": "Full", "8k": "8000", "4k": "4000", "2k": "2000"}
    fixed = [("Fixed 0.0", "Base"), ("Fixed 0.3", "0.3"),
             ("Fixed 0.5", "0.5"), ("Fixed 0.8", "Wide")]
    scenario_policy = "Development near-best 10pp"

    fig = plt.figure(figsize=(9.4, 11.8))
    gs = fig.add_gridspec(3, 2, hspace=0.50, wspace=0.30,
                          left=0.09, right=0.97, bottom=0.09, top=0.89)
    top_axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    bottom_axes = [fig.add_subplot(gs[i, 1]) for i in range(3)]

    for idx, ((name, bidder), ax) in enumerate(zip(bidders.items(), top_axes)):
        subset = selected[selected.bidder == bidder]
        for supply, (color, linestyle, marker) in supplies.items():
            line = subset[subset.supply_cap.astype(str) == supply_cap[supply]].sort_values("budget_multiplier")
            assert len(line) == 9, (bidder, supply, len(line))
            ax.plot(line.budget_multiplier, line.selected_kappa, label=supply,
                    color=color, linestyle=linestyle, marker=marker, linewidth=1.8,
                    markersize=5.5, markerfacecolor="white")
        ax.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5])
        ax.set_xlim(0.47, 1.53)
        ax.set_ylim(-0.05, 1.07)
        ax.set_yticks([0, 0.3, 0.5, 0.8, 1.0])
        ax.tick_params(labelsize=12.0)
        ax.set_xlabel("Budget multiplier")
        if idx == 0:
            ax.set_ylabel(r"Selected execution width $\kappa$")
        clean(ax)
        panel(ax, chr(ord("a") + 2 * idx), f"Selected width ({name}–PID)", fontsize=12.5)

    handles, labels = top_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.53, 0.99),
               frameon=False, title="Opportunity supply", fontsize=12.5, title_fontsize=12.5)

    for idx, ((name, bidder), ax) in enumerate(zip(bidders.items(), bottom_axes)):
        subset = later[later.bidder == bidder]
        means = subset.groupby("policy")[["extra_ud_pp", "value_gain"]].mean()
        assert all(subset[subset.policy == policy].shape[0] == 20 for policy, _ in fixed + [(scenario_policy, "")])
        xs = [means.loc[policy, "extra_ud_pp"] for policy, _ in fixed]
        ys = [means.loc[policy, "value_gain"] / 1000 for policy, _ in fixed]
        ax.plot(xs, ys, color=BLUE, marker="o", markerfacecolor="white", markeredgewidth=1.2)
        for x, y, (_, label) in zip(xs, ys, fixed):
            offset = (4, 5) if label != "Wide" else (-28, 6)
            if bidder == "sembid" and label == "Base":
                offset = (0, -16)
            ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=11.5, color=INK)
        sx = means.loc[scenario_policy, "extra_ud_pp"]
        sy = means.loc[scenario_policy, "value_gain"] / 1000
        ax.scatter(sx, sy, s=58, color=ORANGE, marker="s", zorder=4)
        policy_offset = (5, -5) if bidder == "iql" else (5, 7)
        ax.annotate("Scenario-specific\npolicy", (sx, sy), xytext=policy_offset,
                    textcoords="offset points", fontsize=11.5, fontweight="bold",
                    va="top" if bidder == "iql" else "bottom", color=INK)
        ax.axvline(10, color=MUTED, linestyle=":", linewidth=1.0)
        ax.axhline(0, color=MUTED, linestyle="--", linewidth=0.8)
        ax.set_xlim(-0.5, 25)
        ax.set_ylim(-0.65, 1.22)
        ax.set_xlabel("Mean extra underdelivery (pp)")
        if idx == 0:
            ax.set_ylabel(r"Advertiser value gain ($\times 10^3$)")
        ax.tick_params(labelsize=12.0)
        clean(ax)
        panel(ax, chr(ord("b") + 2 * idx), f"Value–delivery comparison ({name}–PID)", fontsize=12.5)

    save(fig, "fig6_scenario_calibration.pdf")


def figure8() -> None:
    """Symmetric width/adoption design around the focal (0.8, 50%) policy."""
    paths = pd.read_csv(DATA / "figure3c_rollout_paths.csv")
    # Panels A and B use the common all-zero-width market as the baseline and
    # compare the two zero-origin rollout widths observed at 50% adoption.
    # Panels C and D hold the target width at 0.8 and vary the same rollout's
    # adoption rate.  Thus all four panels share one baseline and focal point.
    slice50 = paths[np.isclose(paths.alpha, 0.5)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.50))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.13, top=0.88, wspace=0.26, hspace=0.50)
    a, b, c, d = axes.flat

    for ref in REFS_MACRO:
        x = [0.0]
        value = [0.0]
        revenue = [0.0]
        for path, target in [("0_to_0.3", 0.3), ("0_to_0.8", 0.8)]:
            row = slice50[(slice50.reference.eq(ref)) & (slice50.rollout_path.eq(path))].iloc[0]
            x.append(target)
            value.append(row.mean_relative_total_base_expected_value_pct)
            revenue.append(row.mean_relative_platform_revenue_pct)
        for ax, y in [(a, value), (b, revenue)]:
            ax.plot(
                x,
                y,
                color=COLOR[ref],
                linestyle=STYLE[ref],
                marker=MARKER[ref],
                markerfacecolor="white",
                markeredgewidth=1.2,
                label=LABEL[ref] if ax is a else None,
            )
            ax.scatter([0.8], [y[-1]], color=COLOR[ref], s=50, zorder=5)

        q = paths[(paths.reference.eq(ref)) & (paths.rollout_path.eq("0_to_0.8"))].sort_values("alpha")
        xx = np.r_[0.0, 100 * q.alpha.to_numpy()]
        yy_value = np.r_[0.0, q.mean_relative_total_base_expected_value_pct.to_numpy()]
        yy_revenue = np.r_[0.0, q.mean_relative_platform_revenue_pct.to_numpy()]
        for ax, y in [(c, yy_value), (d, yy_revenue)]:
            ax.plot(
                xx,
                y,
                color=COLOR[ref],
                linestyle=STYLE[ref],
                marker=MARKER[ref],
                markerfacecolor="white",
                markeredgewidth=1.2,
            )
            focal = np.where(np.isclose(xx, 50))[0][0]
            ax.scatter([xx[focal]], [y[focal]], color=COLOR[ref], s=50, zorder=5)

    for ax in (a, b):
        ax.axhline(0, color=INK, linewidth=0.75)
        ax.set_xticks([0.0, 0.3, 0.8])
        ax.set_xlim(-0.04, 0.84)
        ax.set_xlabel(r"Adopter execution width $\kappa$")
        clean(ax)
    a.set_ylabel("Advertiser value change (%)")
    b.set_ylabel("Current auction revenue change (%)")

    for ax in (c, d):
        ax.axhline(0, color=INK, linewidth=0.75)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlim(-3, 103)
        ax.set_xlabel("Adoption rate (%)")
        clean(ax)
    c.set_ylabel("Advertiser value change (%)")
    d.set_ylabel("Current auction revenue change (%)")
    for ax in (a, c):
        ax.set_ylim(0, 250)
        ax.set_yticks([0, 50, 100, 150, 200, 250])
    for ax in (b, d):
        ax.set_ylim(-40, 5)
        ax.set_yticks([-40, -30, -20, -10, 0])
    panel(a, "a", "Advertiser value (50% adoption)")
    panel(b, "b", "Current auction revenue (50% adoption)")
    panel(c, "c", r"Advertiser value ($\kappa=0.8$)")
    panel(d, "d", r"Current auction revenue ($\kappa=0.8$)")
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))
    save(fig, "fig7_market_adoption.pdf")


def main() -> None:
    configure()
    figure3()
    figure4()
    figure5()
    figure6()
    figure7()
    figure8()
    print(f"Wrote six main-text figures to {OUT}")


if __name__ == "__main__":
    main()
