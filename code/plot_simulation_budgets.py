"""Draw the main paper's two-budget summary from the archived analysis tables."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from analyze_simulations import BUD, COMPACT_LABELS, ROOT, SCENES, write_grouped_table


INPUT = ROOT / "results/budget_refresh"
METHODS = {BUD: ("BUD", "#2166ac", "o"),
           "PC1-X": ("PC1-X", "#a5520d", "^"),
           "SRS": ("SRS", "#535d68", "s")}
def plot(output, planned_summary, planned_budget, reference_summary=INPUT / "paper_simulation"):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.right": False, "axes.spines.top": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 4.65), sharey=True)
    fig.subplots_adjust(left=.264, right=.986, top=.924, bottom=.173, wspace=.20)
    rows = np.arange(len(SCENES))
    bounds = [[], [], []]

    def points(panel, values, offset, method, filled):
        center, lower, upper = values
        _, color, marker = METHODS[method]
        axes[panel].errorbar(
            center, rows + offset, xerr=[center - lower, upper - center], fmt=marker,
            markerfacecolor=color if filled else "white", markeredgecolor=color,
            markeredgewidth=.85, markersize=3.7, ecolor=color, elinewidth=.73,
            capsize=1.35, zorder=3)
        bounds[panel].extend([float(np.min(lower)), float(np.max(upper))])

    for budget, base, offset in ((120, reference_summary, -.23),
                                (planned_budget, planned_summary, .23)):
        summary = pd.read_csv(base / "simulation_summary.csv").set_index(["method", "scene"])
        ratios = pd.read_csv(base / "plot_ratios.csv").set_index(["method", "metric", "scene"]).sort_index()
        assert summary.index.is_unique and ratios.index.is_unique
        filled = budget == 120
        for method, shift in ((BUD, -.08), ("PC1-X", .08)):
            rmse = ratios.loc[(method, "rmse"), :].loc[list(SCENES)]
            points(0, rmse[["ratio", "lower", "upper"]].to_numpy().T, offset + shift, method, filled)
        bud = summary.loc[BUD].loc[list(SCENES)]
        points(1, bud[["coverage", "coverage_lower", "coverage_upper"]].to_numpy().T, offset, BUD, filled)
        for method, shift in zip(METHODS, (-.14, 0, .14)):
            width = summary.loc[method].loc[list(SCENES), ["width", "width_lower", "width_upper"]]
            points(2, width.to_numpy().T / 2, offset + shift, method, filled)

    axes[0].set_yticks(rows, COMPACT_LABELS)
    for i, (ax, reference, xlabel, ticks) in enumerate(zip(
        axes, (1, .95, .15),
        ("RMSE / SRS", "BUD coverage", "Mean half-width"),
        ((.6, .8, 1), (.91, .93, .95, .97), (.05, .10, .15, .20, .25))
    )):
        ax.axvline(reference, color="#606060", linestyle=(0, (3, 3)), linewidth=.85)
        ax.axhline(7.5, color="#b8c0c8", linewidth=.7)
        ax.grid(axis="x", color="#e7e9ec", linewidth=.65)
        ax.set_axisbelow(True)
        ax.set_ylim(11.59, -.59)
        ax.set_xlabel(xlabel)
        ax.set_xticks(ticks)
        lo, hi = min(*bounds[i], *ticks), max(*bounds[i], *ticks)
        pad = .025 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        ax.tick_params(axis="both", length=2.6)
        ax.text(0, 1.017, "ABC"[i], transform=ax.transAxes, fontweight="bold", fontsize=10)
    handles = [
        Line2D([], [], marker=marker, color=color, linestyle="none", label=label, markersize=4.5)
        for label, color, marker in METHODS.values()
    ]
    for budget, face in ((120, "#30343a"), (planned_budget, "white")):
        handles.append(Line2D([], [], marker="o", markerfacecolor=face, color="#30343a",
                              linestyle="none", label=f"{budget} per group", markersize=4.5))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.535, .01), ncol=5,
               frameon=False, handletextpad=.4, columnspacing=1.15, handlelength=1)
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=INPUT / "figures/simulation_budgets.pdf")
    parser.add_argument("--planned-summary", type=Path, default=INPUT / "paper_planned")
    parser.add_argument("--reference-summary", type=Path, default=INPUT / "paper_simulation")
    parser.add_argument("--planned-budget", type=int, default=384)
    parser.add_argument("--planned-table", type=Path, help="Write the displayed planned-budget results.")
    parser.add_argument("--reference-table", type=Path, help="Write all three methods' 120-per-group results.")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot(args.out, args.planned_summary, args.planned_budget, args.reference_summary)
    if args.planned_table:
        summary = pd.read_csv(args.planned_summary / "simulation_summary.csv")
        write_grouped_table(summary, args.planned_table)
    if args.reference_table:
        summary = pd.read_csv(args.reference_summary / "simulation_summary.csv")
        write_grouped_table(summary, args.reference_table)
