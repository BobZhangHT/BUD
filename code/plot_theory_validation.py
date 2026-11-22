"""Draw the three S11 figures from the retained regular-case experiments."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results/theory_validation"
FIGURE_NAMES = ("theory_geometry.pdf", "theory_bootstrap.pdf", "theory_budget.pdf")
BLUE, GREEN, ORANGE = "#2166ac", "#11766c", "#a5520d"
GRAY, GRID = "#58636d", "#e7ebef"
SERIES = (
    ("arm0", "Group 0", BLUE, "o", "-"),
    ("arm1", "Group 1", ORANGE, "^", "--"),
    ("contrast", "Group difference", GREEN, "D", "-."),
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def panel(ax, letter, title, grid="y"):
    ax.set_title(f"{letter}   {title}", loc="left", fontweight="bold", pad=8)
    ax.grid(axis=grid, color=GRID, lw=.7)
    ax.set_axisbelow(True)


def reference_line(ax, axis, value):
    draw = ax.axhline if axis == "y" else ax.axvline
    draw(value, color=GRAY, linestyle=(0, (3, 3)), linewidth=1, zorder=1)


def plot_geometry(study, destination):
    result = read_json(study / "geometry.json")
    frames = result["frames"]
    cohorts = [pd.DataFrame(frame["trials"], columns=result["frame_columns"]) for frame in frames]
    labels = [f"$N={frame['N']}$, $p={frame['p']}$" for frame in frames]
    positions = np.arange(len(frames))
    fig, (scatter, variance) = plt.subplots(2, 1, figsize=(6.4, 5.55))
    fig.subplots_adjust(left=.21, right=.975, top=.93, bottom=.17, hspace=.65)

    panel(scatter, "A", "Scatter inequalities")
    reference_line(scatter, "x", 1)
    reference_line(scatter, "y", 1)
    markers = (("o", BLUE), ("^", ORANGE), ("s", GREEN))
    for draws, (marker, color) in zip(cohorts, markers, strict=True):
        initial = draws["initial_scatter"]
        bound = draws["initial_transport"] + draws["discarded_variance"]
        x = initial / bound
        y = draws["final_scatter"] / initial
        assert x.le(1 + 1e-12).all() and y.le(1 + 1e-12).all()
        scatter.scatter(x, y, marker=marker, color=color, s=11, alpha=.22, linewidths=0)
    scatter.set(xlim=(.59, 1.025), ylim=(.705, 1.026), xticks=[.6, .7, .8, .9, 1],
                yticks=[.75, .85, .95, 1], xlabel="Initial scatter / bound",
                ylabel="Refined / initial scatter")
    scatter.legend(handles=[
        Line2D([], [], marker=marker, color=color, linestyle="none", label=label, markersize=5.5)
        for label, (marker, color) in zip(labels, markers, strict=True)
    ], loc="lower center", bbox_to_anchor=(.5, -.52), ncol=3)

    panel(variance, "B", "Design variance checks", grid="x")
    for denominator, offset, color in (
        ("final_bound_projection_plus_noise", -.16, BLUE),
        ("srs_variance_projection_plus_noise", .16, ORANGE),
    ):
        ratios = [draws["final_variance_projection_plus_noise"] / draws[denominator] for draws in cohorts]
        assert max(r.max() for r in ratios) <= 1 + 1e-12
        variance.boxplot(ratios, positions=positions + offset, orientation="horizontal", widths=.24,
                         whis=(0, 100), manage_ticks=False, patch_artist=True,
                         boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 1.2},
                         medianprops={"color": color, "linewidth": 1.6},
                         whiskerprops={"color": color}, capprops={"color": color})
    reference_line(variance, "x", 1)
    variance.set(xlim=(0, 1.06), xticks=np.arange(0, 1.01, .2),
                 xlabel="Exact BUD variance divided by reference")
    variance.legend(handles=[
        Line2D([], [], color=BLUE, label="Variance bound"),
        Line2D([], [], color=ORANGE, label="SRS variance"),
    ], loc="lower center", bbox_to_anchor=(.5, -.52), ncol=2)
    variance.set_yticks(positions, labels)
    variance.set_ylim(2.55, -.55)
    fig.savefig(destination)
    plt.close(fig)


def plot_bootstrap(study, destination):
    summary = pd.read_csv(study / "bootstrap/summary.csv")
    draws = pd.read_csv(study / "bootstrap/draws.csv")
    same_law = read_json(study / "evidence_checks.json")["mean_same_law_cdf_distance"]
    sizes = (8, 16, 32, 64, 128)
    fig = plt.figure(figsize=(6.4, 5.6))
    grid = fig.add_gridspec(2, 2, left=.13, right=.975, top=.935, bottom=.095,
                           hspace=.69, wspace=.45, height_ratios=(1, 1.05))
    trend, qq, coverage = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    panel(trend, "A", "Conditional bootstrap approximation")
    panel(qq, "B", "Normal approximation")
    panel(coverage, "C", "Interval coverage")
    probabilities = np.linspace(.01, .99, 99)
    theoretical = norm.ppf(probabilities)
    qq.plot([-2.55, 2.55], [-2.55, 2.55], color=GRAY, ls=":", lw=1)
    reference_line(coverage, "y", 95)
    for i, (target, label, color, marker, linestyle) in enumerate(SERIES):
        selected = summary.loc[summary["target"] == target].set_index("H_per_arm").loc[list(sizes)]
        largest = selected.loc[128]
        trend.plot(sizes, selected["conditional_ks_mean"], color=color, marker=marker,
                   linestyle=linestyle, markersize=5, label=label)
        errors = draws.loc[(draws["target"] == target) & (draws["H_per_arm"] == 128), "error"]
        standardized = errors * np.sqrt(largest["m"] / largest["mV"])
        qq.plot(theoretical, np.quantile(standardized, probabilities), color=color, linestyle=linestyle, lw=1.35)
        level = 100 * largest["basic_coverage"]
        half_length = 100 * 1.96 * largest["basic_coverage_mcse"]
        coverage.errorbar(i, level, yerr=half_length, marker=marker, color=color,
                          markersize=5.5, capsize=4, linewidth=1.25)
        coverage.annotate(f"{level:.1f}", (i, level + half_length),
                          xytext=(0, 6), textcoords="offset points", ha="center", color=color)
    benchmark = np.mean([same_law[target] for target, *_ in SERIES])
    trend.plot(128, benchmark, linestyle="none", marker="x", color=GRAY,
               markersize=6, label="Same-law benchmark")
    trend.set(xscale="log", xlim=(6.7, 151), ylim=(.021, .111),
              xlabel="Cells per group", ylabel="Mean CDF distance")
    trend.set_xticks(sizes, sizes)
    trend.minorticks_off()
    trend.legend(loc="upper right", ncol=1)
    qq.set(xlim=(-2.55, 2.55), ylim=(-2.55, 2.55), xticks=[-2, -1, 0, 1, 2],
           yticks=[-2, -1, 0, 1, 2], xlabel="Normal quantile", ylabel="Design-error quantile")
    coverage.set(xticks=range(3), xticklabels=["Group 0", "Group 1", "Group\n difference"],
                 xlim=(-.5, 2.5), ylim=(92.2, 97.3), yticks=[93, 94, 95, 96, 97],
                 ylabel="Coverage (%)")
    fig.savefig(destination)
    plt.close(fig)


def plot_budget(study, destination):
    budget_dir = study / "budget_corrected"
    report = read_json(budget_dir / "budget.json")
    pilot = report["projections"]["matched"]
    target = report["target"]
    selected = pilot["chosen"]
    candidates = pd.DataFrame(pilot["candidates"]).set_index("m")
    cohorts = pd.read_csv(budget_dir / "budget_cohorts.csv")
    cohorts = cohorts.loc[(cohorts["regime"] == "matched") & (cohorts["role"] == "planned")]
    frequencies = read_json(study / "evidence_checks.json")["matched_pilot_budget_counts"]
    assert selected == candidates.index[candidates["passes"]][0]
    assert candidates["max_projected_empty"].le(report["beta"]).all()
    assert len(cohorts) == 50 and cohorts["m"].eq(selected).all()
    fig = plt.figure(figsize=(6.4, 5.35))
    grid = fig.add_gridspec(2, 2, left=.14, right=.965, top=.93, bottom=.13,
                           hspace=.73, wspace=.49, height_ratios=(1, 1))
    planning = fig.add_subplot(grid[0, :])
    precision, repeat = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    panel(planning, "A", "Pilot-based budget selection")
    panel(precision, "B", f"Precision at $m={selected}$")
    panel(repeat, "C", "Repeated pilots")

    projected = candidates["max_projected_half_width"]
    planning.plot(candidates.index, projected, color=BLUE, marker="o", lw=1.4, markersize=5)
    reference_line(planning, "y", target)
    planning.plot(selected, projected.loc[selected], marker="o", color=BLUE,
                  markerfacecolor="white", markeredgewidth=1.8, markersize=8)
    planning.annotate(f"Selected $m={selected}$", xy=(selected, projected.loc[selected]),
                      xytext=(selected + 6, target + .18), color=BLUE, ha="left",
                      arrowprops={"arrowstyle": "-", "color": BLUE, "lw": 1})
    planning.set(xticks=candidates.index, xlim=(9, 84), ylim=(.16, .77),
                 xlabel="Candidate phase-II size", ylabel="Projected half-width")
    planning.text(83, target + .013, f"Target {target:.2f}", ha="right", color=GRAY)

    precision.scatter(cohorts["exact_half_width"], cohorts["half_width"],
                      color=BLUE, alpha=.68, s=19, edgecolors="none")
    precision.plot([.23, .33], [.23, .33], color=GRAY, ls=":", lw=1)
    reference_line(precision, "x", target)
    reference_line(precision, "y", target)
    precision.set(xlim=(.23, .33), ylim=(.23, .33), xticks=[.24, .28, .32],
                  yticks=[.24, .28, .32], xlabel="Exact linearized\nhalf-width",
                  ylabel="Mean bootstrap half-width")

    sizes, counts = zip(*sorted((int(size), count) for size, count in frequencies.items()))
    shades = [BLUE if size == selected else "#707d88" for size in sizes]
    bars = repeat.bar(range(len(sizes)), counts, color=shades, width=.48)
    repeat.bar_label(bars, padding=4)
    repeat.set(xticks=range(len(sizes)), xticklabels=sizes, ylim=(0, 820),
               yticks=[0, 200, 400, 600, 800], xlabel="Selected size",
               ylabel="Number of pilots")
    fig.savefig(destination)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "pdf.fonttype": 42,
    })
    for draw, name in zip((plot_geometry, plot_bootstrap, plot_budget), FIGURE_NAMES):
        draw(args.study, args.out / name)


if __name__ == "__main__":
    main()
