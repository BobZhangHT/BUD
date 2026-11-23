"""Run the S11 budget audit, evidence checks, and retained figures."""

import argparse
from collections import Counter
import csv
import gzip
from itertools import combinations, product
import hashlib
import json
from math import comb, exp, floor, sqrt
from pathlib import Path
from statistics import NormalDist
import sys

import bud_experiment

sys.path.insert(0, str(bud_experiment.LEGACY))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import expit
from scipy.stats import norm
from refinement import refined_design
from survival import km_if, km_rmst, _paired_rescaled_weights

ROOT = bud_experiment.PROJECT
N, COHORTS, DRAWS, BOOTSTRAP, PILOT_N = 120, 50, 20, 1000, 2400
GRID = np.array([1., 2., 3.])
TAU = 4.
BUDGETS = (12, 20, 30, 40, 48, 60, 80)
TARGET, BETA = .32, .05
Z = NormalDist().inv_cdf(.975)
SEED = 840917


def cohort(rng, n, late_events=False, support=.9):
    x = rng.standard_normal((n, 4))
    alive = rng.random(n) < expit(-.15 + .6*x[:, 0] - .4*x[:, 1])
    early = expit(.18 + .45*x[:, 2])
    middle = expit(.18 - .35*x[:, 3])
    u, v = rng.random((2, n))
    failure = np.where(u < early, 1., np.where(v < middle, 2., 3.))
    follow_up = np.where(rng.random(n) < support, 4.5, 3.5)
    return x, np.where(alive, follow_up, 3. if late_events else failure), ~alive


def empty_pair_probability(counts, k):
    missed_pairs = (k-counts) * (k-counts-1)
    return float(np.prod(missed_pairs / (k*(k-1))))


def estimates(event_risk):
    hazard = np.divide(event_risk[..., :3], event_risk[..., 3:],
                       out=np.zeros_like(event_risk[..., :3]), where=event_risk[..., 3:] > 0)
    return 1 + np.cumprod(1 - hazard, axis=-1).sum(axis=-1)


def coefficient(k):
    return k*(k-2) / (2*N*(k-1))


def planned_candidates(moment, risk):
    rows = []
    for m in BUDGETS:
        k = N//(m//2)
        srs_variance = (1/m - 1/N) * (N/(N-1)) * moment
        bound = coefficient(k)*moment
        risk_paired = exp(-2*risk/k)
        risk_srs = comb(N-risk, m)/comb(N, m)
        width = Z*sqrt(max(bound, srs_variance))
        risk_bound = max(risk_paired, risk_srs)
        rows.append(dict(m=m, max_projected_half_width=width, max_projected_empty=risk_bound,
                         passes=width <= TARGET and risk_bound <= BETA))
    return rows


def enumerate_small():
    cells = ((3, 1), (4, 1), (5, 0))
    flags = [np.arange(k) < risk for k, risk in cells]
    pair_counts = []
    for frame in flags:
        pair_counts.append([int(frame[list(ids)].sum()) for ids in combinations(range(len(frame)), 2)])
    outcomes = np.array(list(product(*pair_counts)))
    p_pair = float(np.mean(outcomes.sum(axis=1) == 0))
    exact_pair = np.prod([comb(k-r, 2)/comb(k, 2) for k, r in cells])
    upper_pair = exp(-2*sum(r for k, r in cells)/max(k for k, r in cells))
    all_flags = np.concatenate(flags)
    srs_empty = [not all_flags[list(ids)].any() for ids in combinations(range(len(all_flags)), 6)]
    p_srs = float(np.mean(srs_empty))
    exact_srs = comb(len(all_flags)-int(all_flags.sum()), 6)/comb(len(all_flags), 6)
    p_either = float(np.logical_or.outer(outcomes.sum(axis=1) == 0, srs_empty).mean())
    exact_either = 1-(1-exact_pair)*(1-exact_srs)
    errors = np.abs(np.array([p_pair-exact_pair, p_srs-exact_srs, p_either-exact_either]))
    assert errors.max() < 1e-14 and p_pair <= upper_pair
    return dict(paired=p_pair, paired_upper=upper_pair, srs=p_srs, either=p_either,
                max_formula_error=float(errors.max()))


def phase_ii(cohort_id, m, time, e, grouped, phi, truth, check_weights):
    h, k = grouped.shape
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 2, cohort_id]))
    priorities = rng.random((DRAWS, N))
    order = np.argsort(priorities[:, grouped], axis=2)[..., :2]
    selected = grouped[np.arange(h)[None, :, None], order]
    v = np.c_[e[:, None] & (time[:, None] == GRID), time[:, None] >= GRID].astype(float)
    pair_v = v[selected]
    observed = pair_v.mean(axis=(1, 2))
    estimate = estimates(observed)
    signs = 2*rng.integers(0, 2, size=(DRAWS, BOOTSTRAP, h), dtype=np.int8).astype(float)-1
    scale = sqrt(1 - 2/k)/(2*h)
    increments = signs @ (scale*(pair_v[:, :, 0, :]-pair_v[:, :, 1, :]))
    boot = estimates(observed[:, None, :] + increments)
    if check_weights:
        weights = np.full(m, k / 2)
        meta = [dict(start=2*j, stop=2*j+2, N=k, s=2, f=2/k) for j in range(h)]
        ids = selected[0].ravel()
        reweighted = _paired_rescaled_weights(weights, meta, signs[0, :4])
        np.testing.assert_allclose(km_rmst(time[ids], e[ids], reweighted, TAU), boot[0, :4], atol=1e-12, rtol=0)
    qlo, qhi = np.quantile(boot-estimate[:, None], [.025, .975], axis=1)
    error = estimate-truth
    linear = phi[selected].mean(axis=(1,2))
    return dict(sample_index=np.arange(DRAWS), error=error, half_width=(qhi-qlo)/2,
                covered=(qlo <= error) & (error <= qhi),
                empty=(time[selected] < TAU).all(axis=(1,2)),
                linear_exceed=np.abs(linear) >= TARGET)


def run_budget(output):
    output.mkdir(parents=True, exist_ok=False)
    scenario_specs = {"matched": (False, .9), "low_pilot_variation": (True, .9),
                      "optimistic_pilot_support": (False, .025)}
    diagnostics, draws, projections = [], [], {}
    for name, (pilot_late, target_support) in scenario_specs.items():
        _, p_time, p_event = cohort(np.random.default_rng(np.random.SeedSequence([SEED, 0])),
                                    PILOT_N, late_events=pilot_late)
        score = km_if(p_time, p_event, TAU)
        moment = float(np.mean(score**2))
        risk = floor(N*np.mean(p_time >= TAU))
        plans = planned_candidates(moment, risk)
        chosen = next(row["m"] for row in plans if row["passes"])
        preceding = BUDGETS[BUDGETS.index(chosen)-1] if chosen > BUDGETS[0] else None
        projections[name] = dict(pilot_score_moment=moment, pilot_projected_risk=risk,
                                 chosen=chosen, preceding=preceding, candidates=plans)
        for j in range(COHORTS):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, 1, j]))
            x, time, event = cohort(rng, N, support=target_support)
            truth = km_rmst(time, event, np.ones(N), TAU)
            phi = km_if(time, event, TAU)
            actual_moment = float(np.mean(phi**2))
            r_true = int(np.sum(time >= TAU))
            exact = {}
            for m in BUDGETS:
                labels, _, _ = refined_design(x, m)
                grouped = np.argsort(labels, kind="stable").reshape(m//2, -1)
                k = grouped.shape[1]
                W = phi[grouped].var(axis=1, ddof=1).mean()
                V = (1/m-1/N)*W
                count = np.sum(time[grouped] >= TAU, axis=1)
                exact[m] = (grouped, V, empty_pair_probability(count,k), W,
                            coefficient(k)*actual_moment, exp(-2*r_true/k))
            W0 = max(row[3] for row in exact.values())
            cutoff = Z**2*W0/(TARGET**2 + Z**2*W0/N)
            for m, (_, v, p0, _, upper, risk_bound) in exact.items():
                assert v <= upper + 1e-12 and p0 <= risk_bound + 1e-12
                if m >= cutoff:
                    assert Z*sqrt(v) <= TARGET + 1e-12
            for role, m in (("planned", chosen), ("preceding", preceding)):
                if m is None:
                    continue
                grouped, v, p0, _, upper, risk_bound = exact[m]
                diagnostic = dict(regime=name, role=role, m=m, cohort=j, exact_half_width=Z*sqrt(v),
                                  exact_empty=p0, oracle_empty_bound=risk_bound, target_risk=r_true,
                                  oracle_width_bound=Z*sqrt(upper), chebyshev_bound=min(1., upper/TARGET**2),
                                  meets_exact=(Z*sqrt(v) <= TARGET and p0 <= BETA))
                diagnostics.append(diagnostic)
                draws.append(pd.DataFrame(phase_ii(j,m,time,event,grouped,phi,truth,j == 0)).assign(
                    regime=name, role=role, m=m, cohort=j))
        print(f"{name}: selected {chosen}, preceding {preceding}", flush=True)
    data, diagnostic = pd.concat(draws, ignore_index=True), pd.DataFrame(diagnostics)
    keys = ["regime", "role", "m", "cohort"]
    assert not data.duplicated(keys+["sample_index"]).any()
    assert data.groupby(keys).size().eq(DRAWS).all()
    blocks = data.groupby(keys, as_index=False).agg(half_width=("half_width","mean"),
                   coverage=("covered","mean"), empty=("empty","mean"), linear_exceed=("linear_exceed","mean"))
    blocks = blocks.merge(diagnostic, on=keys, validate="one_to_one")
    summary = []
    metrics = ("half_width","coverage","empty","exact_empty","exact_half_width",
               "oracle_width_bound","oracle_empty_bound","chebyshev_bound","linear_exceed","meets_exact")
    for (name, role, m), group in blocks.groupby(keys[:3]):
        entry = dict(regime=name, role=role, m=int(m), cohorts=len(group), draws=len(group)*DRAWS)
        for col in metrics:
            entry[col] = float(group[col].mean())
            entry[col+"_mcse"] = float(group[col].std(ddof=1)/sqrt(len(group)))
        discrepancy = group["empty"] - group["exact_empty"]
        entry["empty_discrepancy_mcse"] = float(discrepancy.std(ddof=1) / sqrt(len(group)))
        summary.append(entry)
    report = dict(target=TARGET, beta=BETA, n=N, pilot_n=PILOT_N, B=BOOTSTRAP, seed=SEED,
                  budget_candidates=BUDGETS, exact_checks=enumerate_small(), projections=projections,
                  summary=summary, source_sha256={Path(__file__).resolve().relative_to(ROOT).as_posix():
                                 hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    data.to_csv(output/"budget_draws.csv.gz", index=False)
    blocks.to_csv(output/"budget_cohorts.csv", index=False)
    (output/"budget.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


def same_law_cdf_distances(archive, rng, repetitions):
    distances = {}
    with np.load(archive) as reference:
        for group in ("arm0", "arm1", "contrast"):
            law = reference[f"{group}_H128"].round(12)
            support, indices = np.unique(law, return_inverse=True)
            errors = []
            for _ in range(repetitions):
                first = np.bincount(rng.choice(indices, 1000), minlength=len(support)) / 1000
                second = np.bincount(rng.choice(indices, 8192), minlength=len(support)) / 8192
                errors.append(np.max(np.abs(np.cumsum(first - second))))
            distances[group] = float(np.mean(errors))
    return distances


def pilot_variation(rng, repetitions):
    sizes = []
    for _ in range(repetitions):
        _, time, event = cohort(rng, PILOT_N)
        scores = km_if(time, event, TAU)
        horizon_count = floor(N * np.mean(time >= TAU))
        candidates = planned_candidates(float(np.mean(scores**2)), horizon_count)
        sizes.append(next((row["m"] for row in candidates if row["passes"]), "none"))
    return dict(Counter(map(str, sizes)))


def numerical_checks(root):
    geometry = json.loads((root / "geometry.json").read_bytes())
    margins = [
        grid["triangle_continuous_lower_rhs"] - example["sqrt_transport"]
        for example in geometry["transport_cases"] if example["q"] == 2
        for grid in example["grids"]
    ]
    assert len(margins) == 18 and min(margins) >= -1e-9
    with (root / "bootstrap/draws.csv").open(newline="") as inputs:
        tail = [row for row in csv.DictReader(inputs)
                if row["target"] == "arm0" and row["H_per_arm"] == "128"]
    assert len(tail) == 1000
    design_exceed = sum(row["design_tail_20"] == "True" for row in tail)
    bootstrap_exceed = round(1000 * sum(float(row["bootstrap_tail_20"]) for row in tail))
    with gzip.open(root / "budget_corrected" / "budget_draws.csv.gz", "rt", newline="") as inputs:
        selected = [row for row in csv.DictReader(inputs)
                    if row["regime"] == "matched" and row["role"] == "planned"]
    assert len(selected) == 1000
    realized_exceed = sum(float(row["half_width"]) > TARGET for row in selected)
    return dict(transport_minimum_triangle_margin=min(margins),
                arm0_design_tail_exceed=design_exceed,
                arm0_bootstrap_tail_exceed=bootstrap_exceed,
                matched_realized_half_width_exceed=realized_exceed)


def write_evidence(study, destination):
    repetitions = 1000
    seed = 5439022
    cdf_rng, pilot_rng = [np.random.default_rng(child)
                          for child in np.random.SeedSequence(seed).spawn(2)]
    result = dict(seed=seed, repetitions=repetitions, **numerical_checks(study),
                  mean_same_law_cdf_distance=same_law_cdf_distances(
                      study / "bootstrap/reference.npz", cdf_rng, repetitions),
                  matched_pilot_budget_counts=pilot_variation(pilot_rng, repetitions))
    with destination.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result, indent=2))


FIGURE_NAMES = ("theory_geometry.pdf", "theory_bootstrap.pdf", "theory_budget.pdf")
BLUE, GREEN, ORANGE = "#2166ac", "#11766c", "#a5520d"
GRAY, GRID_COLOR = "#58636d", "#e7ebef"
SERIES = (
    ("arm0", "Group 0", BLUE, "o", "-"),
    ("arm1", "Group 1", ORANGE, "^", "--"),
    ("contrast", "Group difference", GREEN, "D", "-."),
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def panel(ax, letter, title, grid="y"):
    ax.set_title(f"{letter}   {title}", loc="left", fontweight="bold", pad=8)
    ax.grid(axis=grid, color=GRID_COLOR, lw=.7)
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
        ratios = [draws["final_variance_projection_plus_noise"] / draws[denominator]
                  for draws in cohorts]
        assert max(r.max() for r in ratios) <= 1 + 1e-12
        variance.boxplot(ratios, positions=positions + offset, orientation="horizontal", widths=.24,
                         whis=(0, 100), manage_ticks=False, patch_artist=True,
                         boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 1.2},
                         medianprops={"color": color, "linewidth": 1.6},
                         whiskerprops={"color": color}, capprops={"color": color})
    reference_line(variance, "x", 1)
    variance.set(xlim=(0, 1.06), xticks=np.arange(0, 1.01, .2),
                 xlabel="Exact BUD variance divided by reference")
    variance.legend(handles=[Line2D([], [], color=BLUE, label="Variance bound"),
                             Line2D([], [], color=ORANGE, label="SRS variance")],
                    loc="lower center", bbox_to_anchor=(.5, -.52), ncol=2)
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
    trend, qq, coverage = (fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]),
                           fig.add_subplot(grid[1, 1]))
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
        errors = draws.loc[(draws["target"] == target) &
                           (draws["H_per_arm"] == 128), "error"]
        standardized = errors * np.sqrt(largest["m"] / largest["mV"])
        qq.plot(theoretical, np.quantile(standardized, probabilities), color=color,
                linestyle=linestyle, lw=1.35)
        level = 100 * largest["basic_coverage"]
        half_length = 100 * 1.96 * largest["basic_coverage_mcse"]
        coverage.errorbar(i, level, yerr=half_length, marker=marker, color=color,
                          markersize=5.5, capsize=4, linewidth=1.25)
        coverage.annotate(f"{level:.1f}", (i, level + half_length), xytext=(0, 6),
                          textcoords="offset points", ha="center", color=color)
    benchmark = np.mean([same_law[target] for target, *_ in SERIES])
    trend.plot(128, benchmark, linestyle="none", marker="x", color=GRAY,
               markersize=6, label="Same-law benchmark")
    trend.set(xscale="log", xlim=(6.7, 151), ylim=(.021, .111),
              xlabel="Cells per group", ylabel="Mean CDF distance")
    trend.set_xticks(sizes, sizes)
    trend.minorticks_off()
    trend.legend(loc="upper right", ncol=1)
    qq.set(xlim=(-2.55, 2.55), ylim=(-2.55, 2.55), xticks=[-2, -1, 0, 1, 2],
           yticks=[-2, -1, 0, 1, 2], xlabel="Normal quantile",
           ylabel="Design-error quantile")
    coverage.set(xticks=range(3), xticklabels=["Group 0", "Group 1", "Group\n difference"],
                 xlim=(-.5, 2.5), ylim=(92.2, 97.3), yticks=[93, 94, 95, 96, 97],
                 ylabel="Coverage (%)")
    fig.savefig(destination)
    plt.close(fig)


def plot_budget(study, destination):
    report = read_json(study / "budget_corrected" / "budget.json")
    pilot = report["projections"]["matched"]
    target = report["target"]
    selected = pilot["chosen"]
    candidates = pd.DataFrame(pilot["candidates"]).set_index("m")
    cohorts = pd.read_csv(study / "budget_corrected" / "budget_cohorts.csv")
    cohorts = cohorts.loc[(cohorts["regime"] == "matched") & (cohorts["role"] == "planned")]
    frequencies = read_json(study / "evidence_checks.json")["matched_pilot_budget_counts"]
    assert selected == candidates.index[candidates["passes"]][0]
    assert candidates["max_projected_empty"].le(report["beta"]).all()
    assert len(cohorts) == 50 and cohorts["m"].eq(selected).all()
    fig = plt.figure(figsize=(6.4, 5.35))
    grid = fig.add_gridspec(2, 2, left=.14, right=.965, top=.93, bottom=.13,
                           hspace=.73, wspace=.49, height_ratios=(1, 1))
    planning, precision, repeat = (fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]),
                                   fig.add_subplot(grid[1, 1]))
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


def draw_figures(study, destination):
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "pdf.fonttype": 42,
    })
    for draw, name in zip((plot_geometry, plot_bootstrap, plot_budget), FIGURE_NAMES):
        draw(study, destination / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "results" / "theory_validation")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures")
    parser.add_argument("--stage", choices=("all", "budget", "evidence", "figures"), default="all",
                        help="Regenerate all outputs or run a single stage on existing study files.")
    args = parser.parse_args()
    study = args.study.resolve()
    if args.stage in {"all", "budget"}:
        run_budget(study / "budget_corrected")
    if args.stage in {"all", "evidence"}:
        write_evidence(study, study / "evidence_checks.json")
    if args.stage in {"all", "figures"}:
        draw_figures(study, args.figures.resolve())


if __name__ == "__main__":
    main()
