"""Analyze simulation draws and draw the manuscript's budget comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("SRS", "PC1-X", "BUD-fullspace-onepass")
BUD = METHODS[-1]
LABELS = {"SRS": "SRS", "PC1-X": "First PC", BUD: "BUD"}
COLORS = {"SRS": "#7B8490", "PC1-X": "#D18424", BUD: "#2166AC"}
SCENES = (
    "val_pc1",
    "val_multivariate",
    "val_banana",
    "val_saddle",
    "val_mixture",
    "val_nuisance",
    "val_lowvar",
    "val_null",
    "banana",
    "saddle3d",
    "gaussian",
    "multimodal",
)
SCENE_LABELS = (
    "Leading factor",
    "Several prognostic factors",
    "Banana, 2D",
    "Saddle, 6D",
    "Unequal mixture",
    "Many nuisance variables",
    "Low-variance signal",
    "Null signal",
    "Curved trajectories",
    "Saddle trajectories",
    "Correlated trajectories",
    "Mixture trajectories",
)
COMPACT_LABELS = (
    "Leading factor",
    "Several factors",
    "Banana (2D)",
    "Saddle (6D)",
    "Unequal mixture",
    "Nuisance variables",
    "Low-variance signal",
    "Null signal",
    "Curved, varying effect",
    "Saddle, varying effect",
    "Correlated, varying effect",
    "Mixture, varying effect",
)


def write_grouped_table(summary, path):
    """Write all three prespecified methods for every frozen scenario."""
    rows = [
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrrrrr@{}}",
        r"\toprule",
        r"& \multicolumn{3}{c}{BUD} & \multicolumn{3}{c}{PC1-X} & \multicolumn{3}{c}{SRS} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}",
        r"Setting & RMSE & Width & Cov. & RMSE & Width & Cov. & RMSE & Width & Cov. \\",
        r"\midrule",
    ]
    labels = COMPACT_LABELS[:8] + (
        "Curved (TV)",
        "Saddle (TV)",
        "Correlated (TV)",
        "Mixture (TV)",
    )
    for scene, label in zip(SCENES, labels):
        values = (
            summary[summary.scene.eq(scene)]
            .set_index("method")
            .loc[[BUD, "PC1-X", "SRS"], ["rmse", "width", "coverage"]]
        )
        cells = [
            f"{rmse:.3f} & {width:.3f} & {coverage * 100:.1f}"
            for rmse, width, coverage in values.itertuples(index=False, name=None)
        ]
        rows.append(label + " & " + " & ".join(cells) + r" \\")
    rows.extend([r"\bottomrule", r"\end{tabular*}"])
    path.write_text("\n".join(rows))


def _holm(p_values):
    values, order, adjusted, last = (
        np.asarray(p_values),
        np.argsort(p_values),
        np.empty(len(p_values)),
        0.0,
    )
    for rank, index in enumerate(order):
        last = max(last, min(1.0, values[index] * (len(values) - rank)))
        adjusted[index] = last
    return adjusted


def _input(source, stem):
    for suffix in (".csv", ".csv.gz"):
        path = source / f"{stem}{suffix}"
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"Missing {stem}.csv or {stem}.csv.gz in {source}")


def analyze(
    confirmation, output, figures, *, cohorts=64, draws=20, planned_table=False
):
    """Summarize complete-cohort blocks; never treat phase-II draws as independent."""
    source, output, figures = ROOT / confirmation, ROOT / output, ROOT / figures
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    raw = _input(source, "raw")
    raw = raw[raw.method.isin(METHODS)].copy()
    # One block contains all draws from the same complete cohort.
    raw["squared_error"] = raw.error**2
    assert set(raw.scene) == set(SCENES)
    assert len(raw) == len(SCENES) * cohorts * draws * len(METHODS)
    assert not raw.duplicated(["scene", "cohort", "draw", "method"]).any()
    assert raw.groupby(["scene", "method"]).cohort.nunique().eq(cohorts).all()
    assert raw.groupby(["scene", "cohort", "method"]).size().eq(draws).all()
    assert (
        raw.groupby(["scene", "cohort", "draw"]).method.nunique().eq(len(METHODS)).all()
    )
    assert np.max(np.abs(raw.error - (raw.estimate - raw.truth))) < 1e-12
    assert raw[["error", "coverage", "width", "boot_variance"]].notna().all().all()
    raw.sort_values(["scene", "cohort", "draw", "method"]).to_csv(
        output / "main_three_method_raw.csv.gz", index=False
    )
    blocks = raw.groupby(["scene", "cohort", "method"], as_index=False).agg(
        mse=("squared_error", "mean"),
        bias=("error", "mean"),
        coverage=("coverage", "mean"),
        width=("width", "mean"),
        boot_variance=("boot_variance", "mean"),
        min_sample_risk=("sample_risk_tau_min", "min"),
    )
    blocks.to_csv(output / "cohort_metrics.csv", index=False)
    summary = []
    for (scene, method), values in blocks.groupby(["scene", "method"]):
        critical = student_t.ppf(0.975, len(values) - 1)
        row = {
            "scene": scene,
            "method": method,
            "cohorts": len(values),
            "draws_per_cohort": draws,
            "repetitions": draws * len(values),
            "rmse": float(np.sqrt(values.mse.mean())),
        }
        for metric in ("mse", "bias", "coverage", "width", "boot_variance"):
            estimate, mcse = (
                float(values[metric].mean()),
                float(values[metric].std(ddof=1) / np.sqrt(len(values))),
            )
            row.update(
                {
                    metric: estimate,
                    f"{metric}_mcse": mcse,
                    f"{metric}_lower": estimate - critical * mcse,
                    f"{metric}_upper": estimate + critical * mcse,
                }
            )
        row["min_sample_risk"] = int(values.min_sample_risk.min())
        summary.append(row)
    summary = pd.DataFrame(summary)
    paired, ratios = [], []
    rng = np.random.default_rng(622113)
    for scene in SCENES:
        values = blocks[blocks.scene.eq(scene)]
        mse = values.pivot(index="cohort", columns="method", values="mse").sort_index()
        width = values.pivot(
            index="cohort", columns="method", values="width"
        ).sort_index()
        # Use the same cohort indices for all methods so the ratios stay paired.
        resamples = rng.integers(0, len(mse), size=(8000, len(mse)))
        for comparator in ("SRS", "PC1-X"):
            difference = mse[BUD] - mse[comparator]
            mcse, critical = (
                float(difference.std(ddof=1) / np.sqrt(len(difference))),
                student_t.ppf(0.975, len(difference) - 1),
            )
            ratio = float(np.sqrt(mse[BUD].mean() / mse[comparator].mean()))
            samples = np.sqrt(
                mse[BUD].to_numpy()[resamples].mean(axis=1)
                / mse[comparator].to_numpy()[resamples].mean(axis=1)
            )
            lower, upper = np.quantile(samples, [0.025, 0.975])
            paired.append(
                {
                    "scene": scene,
                    "benchmark": comparator,
                    "rmse_ratio": ratio,
                    "ratio_lower": lower,
                    "ratio_upper": upper,
                    "relative_rmse_reduction": 1 - ratio,
                    "mse_difference": float(difference.mean()),
                    "mse_difference_mcse": mcse,
                    "mse_difference_lower": float(difference.mean() - critical * mcse),
                    "mse_difference_upper": float(difference.mean() + critical * mcse),
                    "p_two_sided": float(
                        2
                        * student_t.sf(
                            abs(difference.mean() / mcse), len(difference) - 1
                        )
                    ),
                    "ci_width_ratio": float(
                        width[BUD].mean() / width[comparator].mean()
                    ),
                }
            )
        for method in ("PC1-X", BUD):
            for metric, frame, transform in (
                ("rmse", mse, np.sqrt),
                ("width", width, lambda x: x),
            ):
                ratio = float(transform(frame[method].mean() / frame.SRS.mean()))
                samples = transform(
                    frame[method].to_numpy()[resamples].mean(axis=1)
                    / frame.SRS.to_numpy()[resamples].mean(axis=1)
                )
                lower, upper = np.quantile(samples, [0.025, 0.975])
                ratios.append(
                    {
                        "scene": scene,
                        "method": method,
                        "metric": metric,
                        "ratio": ratio,
                        "lower": lower,
                        "upper": upper,
                    }
                )
    paired, ratios = pd.DataFrame(paired), pd.DataFrame(ratios)
    for comparator in ("SRS", "PC1-X"):
        mask = paired.benchmark.eq(comparator)
        paired.loc[mask, "p_holm_12"] = _holm(paired.loc[mask, "p_two_sided"])
    paired.to_csv(output / "paired_evidence.csv", index=False)
    ratios.to_csv(output / "plot_ratios.csv", index=False)
    summary.to_csv(output / "simulation_summary.csv", index=False)
    pivot = summary.pivot(index="scene", columns="method", values="rmse")
    rmse_vs_pc1 = pivot[BUD] / pivot["PC1-X"]
    bud = summary[summary.method.eq(BUD)].set_index("scene")
    headline = {
        "number_scenarios": len(SCENES),
        "independent_cohorts_per_scenario": cohorts,
        "conditional_draws_per_cohort": draws,
        "estimates_three_methods": len(raw),
        "bud_point_wins": int(pivot.idxmin(axis=1).eq(BUD).sum()),
        "bud_within_5pct_of_best": int((pivot[BUD] / pivot.min(axis=1) <= 1.05).sum()),
        "rmse_geometric_ratio_to_srs": float(
            np.exp(np.log(pivot[BUD] / pivot.SRS).mean())
        ),
        "rmse_geometric_ratio_to_pc1": float(np.exp(np.log(rmse_vs_pc1).mean())),
        "width_geometric_ratio_to_srs": float(
            np.exp(np.log(paired[paired.benchmark.eq("SRS")].ci_width_ratio).mean())
        ),
        "coverage_range": [float(bud.coverage.min()), float(bud.coverage.max())],
        "coverage_mean": float(bud.coverage.mean()),
        "significant_improvements_srs": int(
            (
                (paired.benchmark.eq("SRS"))
                & (paired.mse_difference < 0)
                & (paired.p_holm_12 < 0.05)
            ).sum()
        ),
        "significant_improvements_pc1": int(
            (
                (paired.benchmark.eq("PC1-X"))
                & (paired.mse_difference < 0)
                & (paired.p_holm_12 < 0.05)
            ).sum()
        ),
        "negative_pc1_scene": rmse_vs_pc1.idxmax() if rmse_vs_pc1.max() > 1 else None,
        "pc1_friendly_rmse_ratio": float(rmse_vs_pc1.loc["val_pc1"]),
        "all_draws_retained": True,
        "cluster_unit": "complete cohort",
        "ratio_mc_replicates": 8000,
        "three_method_selection": "Current comparison set requested after the broader experiment; historical archive retained.",
    }
    (output / "simulation_headline.json").write_text(json.dumps(headline, indent=2))
    diagnostics = _input(source, "diagnostics")
    geometry = diagnostics[diagnostics.variant.eq(BUD)].copy()
    assert (
        len(geometry) == len(SCENES) * cohorts * 2
        and (geometry.full_after <= geometry.full_before + 1e-10).all()
    )
    geometry = geometry.groupby("scene", as_index=False).agg(
        q_initial=("q85", "mean"),
        scatter_before=("full_before", "mean"),
        scatter_after=("full_after", "mean"),
        event_fraction=("event_fraction", "mean"),
        risk_at_tau=("risk_at_tau", "mean"),
        seconds_design_per_arm=("seconds", "mean"),
    )
    geometry["scatter_ratio"] = geometry.scatter_after / geometry.scatter_before
    geometry.to_csv(output / "geometry_diagnostics.csv", index=False)
    _main_figure(summary, ratios, figures)
    _paired_figure(paired, figures)
    _main_table(summary, output / "table_main.tex")
    if planned_table:
        write_grouped_table(summary, output / "table_planned.tex")
    print(json.dumps(headline, indent=2))


def _main_figure(summary, ratios, figures):
    """Write the full-width and compact reference-budget figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.5, 5.4),
        sharey=True,
        gridspec_kw={"width_ratios": [1.15, 1, 1]},
    )
    y = np.arange(len(SCENES))
    for j, (metric, title) in enumerate(
        [
            ("rmse", "A  RMSE relative to SRS"),
            ("coverage", "B  95% interval coverage"),
            ("width", "C  Mean length relative to SRS"),
        ]
    ):
        ax = axes[j]
        if metric == "coverage":
            ax.axvline(0.95, color="#404040", lw=1, ls="--", zorder=0)
            for method, off in zip(METHODS, [-0.20, 0, 0.20]):
                v = (
                    summary[summary.method == method]
                    .set_index("scene")
                    .loc[list(SCENES)]
                )
                ax.errorbar(
                    v.coverage,
                    y + off,
                    xerr=np.vstack(
                        [v.coverage - v.coverage_lower, v.coverage_upper - v.coverage]
                    ),
                    fmt="o",
                    ms=3.6,
                    color=COLORS[method],
                    elinewidth=0.8,
                    capsize=1.5,
                    label=LABELS[method],
                )
            ax.set_xlim(0.905, 0.982)
            ax.set_xticks([0.91, 0.93, 0.95, 0.97])
            ax.set_xlabel("Coverage probability")
        else:
            ax.axvline(1, color="#404040", lw=1, ls="--", zorder=0)
            for method, off in [("PC1-X", -0.11), (BUD, 0.11)]:
                v = (
                    ratios[(ratios.method == method) & (ratios.metric == metric)]
                    .set_index("scene")
                    .loc[list(SCENES)]
                )
                ax.errorbar(
                    v.ratio,
                    y + off,
                    xerr=np.vstack([v.ratio - v.lower, v.upper - v.ratio]),
                    fmt="o",
                    ms=4,
                    color=COLORS[method],
                    elinewidth=1,
                    capsize=2,
                    label=LABELS[method],
                )
            ax.set_xlim(0.57, 1.08)
            ax.set_xticks([0.6, 0.8, 1.0])
            ax.set_xlabel(
                "Smaller indicates lower " + ("error" if metric == "rmse" else "length")
            )
        ax.axhline(7.5, color="#BCC1C7", lw=0.8)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=10)
        ax.grid(axis="x", alpha=0.18)
        ax.set_yticks(y)
        ax.set_ylim(11.6, -0.6)
    axes[0].set_yticklabels(SCENE_LABELS)
    axes[1].legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False
    )
    fig.subplots_adjust(left=0.21, right=0.99, top=0.93, bottom=0.17, wspace=0.14)
    fig.savefig(figures / "simulation_main.pdf", bbox_inches="tight")
    fig.savefig(figures / "simulation_main.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    # A native journal-width variant keeps all twelve settings and readable type.
    fig, axes = plt.subplots(
        1, 3, figsize=(7.15, 4.9), sharey=True, gridspec_kw={"width_ratios": [1, 1, 1]}
    )
    for j, (metric, title) in enumerate(
        [
            ("rmse", "A  RMSE / SRS"),
            ("coverage", "B  Coverage"),
            ("width", "C  Length / SRS"),
        ]
    ):
        ax = axes[j]
        if metric == "coverage":
            ax.axvline(0.95, color="#404040", lw=0.8, ls="--", zorder=0)
            for method, off in zip(METHODS, [-0.21, 0, 0.21]):
                v = (
                    summary[summary.method == method]
                    .set_index("scene")
                    .loc[list(SCENES)]
                )
                ax.errorbar(
                    v.coverage,
                    y + off,
                    xerr=np.vstack(
                        [v.coverage - v.coverage_lower, v.coverage_upper - v.coverage]
                    ),
                    fmt="o",
                    ms=2.8,
                    color=COLORS[method],
                    elinewidth=0.7,
                    capsize=1,
                    label=LABELS[method],
                )
            ax.set_xlim(0.905, 0.982)
            ax.set_xticks([0.91, 0.95, 0.98])
            ax.set_xlabel("Nominal: 0.95", fontsize=9.4)
        else:
            ax.axvline(1, color="#404040", lw=0.8, ls="--", zorder=0)
            for method, off in [("PC1-X", -0.13), (BUD, 0.13)]:
                v = (
                    ratios[(ratios.method == method) & (ratios.metric == metric)]
                    .set_index("scene")
                    .loc[list(SCENES)]
                )
                ax.errorbar(
                    v.ratio,
                    y + off,
                    xerr=np.vstack([v.ratio - v.lower, v.upper - v.ratio]),
                    fmt="o",
                    ms=3.1,
                    color=COLORS[method],
                    elinewidth=0.8,
                    capsize=1.3,
                    label=LABELS[method],
                )
            ax.set_xlim(0.57, 1.08)
            ax.set_xticks([0.6, 0.8, 1.0])
            ax.set_xlabel("Lower is better", fontsize=9.4)
        ax.axhline(7.5, color="#BCC1C7", lw=0.7)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=8)
        ax.grid(axis="x", alpha=0.15)
        ax.set_yticks(y)
        ax.set_ylim(11.6, -0.6)
        ax.tick_params(labelsize=9.4)
    axes[0].set_yticklabels(COMPACT_LABELS, fontsize=9.4)
    axes[1].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.27),
        ncol=3,
        frameon=False,
        fontsize=9.4,
        handletextpad=0.3,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.26, right=0.995, top=0.91, bottom=0.21, wspace=0.18)
    fig.savefig(figures / "simulation_main_compact.pdf", bbox_inches="tight")
    fig.savefig(figures / "simulation_main_compact.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _paired_figure(paired, figures):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 5.2), sharey=True)
    y = np.arange(len(SCENES))
    for axis, comparator in zip(axes, ("SRS", "PC1-X")):
        values = (
            paired[paired.benchmark.eq(comparator)].set_index("scene").loc[list(SCENES)]
        )
        center, lower, upper = (
            100 * (1 - values.rmse_ratio),
            100 * (1 - values.ratio_upper),
            100 * (1 - values.ratio_lower),
        )
        axis.axvline(0, color="#444444", lw=1, ls="--")
        axis.errorbar(
            center,
            y,
            xerr=np.vstack((center - lower, upper - center)),
            fmt="o",
            color=COLORS[BUD],
            ms=4,
            capsize=2,
        )
        axis.axhline(7.5, color="#BCC1C7", lw=0.8)
        axis.grid(axis="x", alpha=0.18)
        axis.set_title("BUD versus " + LABELS[comparator])
        axis.set_xlabel("RMSE reduction (%)")
        axis.set_ylim(11.6, -0.6)
    axes[0].set_yticks(y, SCENE_LABELS)
    fig.tight_layout()
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 220})):
        fig.savefig(figures / f"paired_effects.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def _main_table(summary, path):
    rows = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Setting & SRS & First PC & BUD & BUD coverage & Length/SRS \\",
        r"\midrule",
    ]
    for scene, label in zip(SCENES, SCENE_LABELS):
        values = summary[summary.scene.eq(scene)].set_index("method")
        rmse = values["rmse"]
        coverage = values.loc[BUD, "coverage"]
        length_ratio = values.loc[BUD, "width"] / values.loc["SRS", "width"]
        rows.append(
            f"{label} & {rmse['SRS']:.4f} & {rmse['PC1-X']:.4f} & {rmse[BUD]:.4f}"
            f" & {coverage:.4f} & {length_ratio:.3f}" + r" \\"
        )
    path.write_text("\n".join(rows + [r"\bottomrule", r"\end{tabular}"]))


def plot_budgets(
    output,
    planned_summary,
    planned_budget,
    reference_summary,
    *,
    planned_table=None,
    reference_table=None,
):
    """Draw joint contrasts from the fixed cohort-level analysis outputs."""
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = {
        BUD: ("BUD", "#2166ac", "o"),
        "PC1-X": ("PC1-X", "#a5520d", "^"),
        "SRS": ("SRS", "#535d68", "s"),
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 4.65), sharey=True)
    fig.subplots_adjust(left=0.264, right=0.986, top=0.924, bottom=0.173, wspace=0.20)
    rows = np.arange(len(SCENES))
    bounds = [[], [], []]

    def points(panel, values, offset, method, filled):
        center, lower, upper = values
        _, color, marker = styles[method]
        axes[panel].errorbar(
            center,
            rows + offset,
            xerr=[center - lower, upper - center],
            fmt=marker,
            markerfacecolor=color if filled else "white",
            markeredgecolor=color,
            markeredgewidth=0.85,
            markersize=3.7,
            ecolor=color,
            elinewidth=0.73,
            capsize=1.35,
            zorder=3,
        )
        bounds[panel].extend([float(np.min(lower)), float(np.max(upper))])

    for budget, base, offset in (
        (120, reference_summary, -0.23),
        (planned_budget, planned_summary, 0.23),
    ):
        summary = pd.read_csv(base / "simulation_summary.csv").set_index(
            ["method", "scene"]
        )
        ratios = (
            pd.read_csv(base / "plot_ratios.csv")
            .set_index(["method", "metric", "scene"])
            .sort_index()
        )
        assert summary.index.is_unique and ratios.index.is_unique
        filled = budget == 120
        for method, shift in ((BUD, -0.08), ("PC1-X", 0.08)):
            rmse = ratios.loc[(method, "rmse"), :].loc[list(SCENES)]
            points(
                0,
                rmse[["ratio", "lower", "upper"]].to_numpy().T,
                offset + shift,
                method,
                filled,
            )
        bud = summary.loc[BUD].loc[list(SCENES)]
        points(
            1,
            bud[["coverage", "coverage_lower", "coverage_upper"]].to_numpy().T,
            offset,
            BUD,
            filled,
        )
        for method, shift in zip(styles, (-0.14, 0, 0.14)):
            width = summary.loc[method].loc[
                list(SCENES), ["width", "width_lower", "width_upper"]
            ]
            points(2, width.to_numpy().T / 2, offset + shift, method, filled)

    axes[0].set_yticks(rows, COMPACT_LABELS)
    for i, (ax, reference, xlabel, ticks) in enumerate(
        zip(
            axes,
            (1, 0.95, 0.15),
            ("RMSE / SRS", "BUD coverage", "Mean half-width"),
            ((0.6, 0.8, 1), (0.91, 0.93, 0.95, 0.97), (0.05, 0.10, 0.15, 0.20, 0.25)),
        )
    ):
        ax.axvline(reference, color="#606060", linestyle=(0, (3, 3)), linewidth=0.85)
        ax.axhline(7.5, color="#b8c0c8", linewidth=0.7)
        ax.grid(axis="x", color="#e7e9ec", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.set_ylim(11.59, -0.59)
        ax.set_xlabel(xlabel)
        ax.set_xticks(ticks)
        lo, hi = min(*bounds[i], *ticks), max(*bounds[i], *ticks)
        pad = 0.025 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        ax.tick_params(axis="both", length=2.6)
        ax.text(
            0, 1.017, "ABC"[i], transform=ax.transAxes, fontweight="bold", fontsize=10
        )
    handles = [
        Line2D(
            [],
            [],
            marker=marker,
            color=color,
            linestyle="none",
            label=label,
            markersize=4.5,
        )
        for label, color, marker in styles.values()
    ]
    for budget, face in ((120, "#30343a"), (planned_budget, "white")):
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                markerfacecolor=face,
                color="#30343a",
                linestyle="none",
                label=f"{budget} per group",
                markersize=4.5,
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.535, 0.01),
        ncol=5,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.15,
        handlelength=1,
    )
    fig.savefig(output)
    plt.close(fig)
    if planned_table:
        write_grouped_table(
            pd.read_csv(planned_summary / "simulation_summary.csv"), planned_table
        )
    if reference_table:
        write_grouped_table(
            pd.read_csv(reference_summary / "simulation_summary.csv"), reference_table
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    analysis = commands.add_parser("analyze")
    analysis.add_argument("--confirmation", type=Path, required=True)
    analysis.add_argument("--out", type=Path, required=True)
    analysis.add_argument("--figures", type=Path, required=True)
    analysis.add_argument("--cohorts", type=int, default=64)
    analysis.add_argument("--draws", type=int, default=20)
    analysis.add_argument("--planned-table", action="store_true")
    chart = commands.add_parser("budget-plot")
    chart.add_argument("--out", type=Path, required=True)
    chart.add_argument("--planned-summary", type=Path, required=True)
    chart.add_argument("--reference-summary", type=Path, required=True)
    chart.add_argument("--planned-budget", type=int, required=True)
    chart.add_argument("--planned-table", type=Path)
    chart.add_argument("--reference-table", type=Path)
    args = parser.parse_args()
    if args.command == "analyze":
        analyze(
            args.confirmation,
            args.out,
            args.figures,
            cohorts=args.cohorts,
            draws=args.draws,
            planned_table=args.planned_table,
        )
    else:
        plot_budgets(
            args.out,
            args.planned_summary,
            args.planned_budget,
            args.reference_summary,
            planned_table=args.planned_table,
            reference_table=args.reference_table,
        )


if __name__ == "__main__":
    main()
