"""Illustrate phase-II planning for GBSG using external Rotterdam outcomes."""

import argparse
import json
from math import ceil, comb, exp, floor, sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 1825
TARGET_SIZE = 686
EVALUATED_BUDGETS = (40, 80, 120, 240)


def rmst_scores(time, event, horizon):
    """KM integral and raw-weight derivatives multiplied by cohort size."""
    jumps, deaths = np.unique(time[event & (time < horizon)], return_counts=True)
    risk = len(time) - np.searchsorted(np.sort(time), jumps, side="left")
    if np.any(deaths >= risk):
        raise ValueError("RMST score requires positive survival at each jump")
    hazard = deaths / risk
    survival = np.cumprod(1 - hazard)
    widths = np.diff(np.r_[jumps, horizon])
    tail = np.cumsum((widths * survival)[::-1])[::-1]
    multiplier = len(time) * tail / (risk - deaths)
    increments = np.r_[0, np.cumsum(multiplier * hazard)]
    scores = increments[np.searchsorted(jumps, time, side="right")]
    recorded = event & (time < horizon)
    scores[recorded] -= multiplier[np.searchsorted(jumps, time[recorded])]
    if not np.isclose(scores.mean(), 0, atol=1e-9, rtol=0):
        raise ArithmeticError("The centered KM score failed its zero-mean identity")
    value = jumps[0] + tail[0] if len(jumps) else float(horizon)
    return float(value), scores


def planning_row(m, second_moment, risk_count, z):
    n = TARGET_SIZE
    largest_cell = ceil(n / (m // 2))
    variance_bound = largest_cell * (largest_cell - 2) / (2 * n * (largest_cell - 1)) * second_moment
    srs_variance = (1 / m - 1 / n) * (n / (n - 1)) * second_moment
    return {
        "m": m,
        "paired_half_width": z * sqrt(variance_bound),
        "srs_half_width": z * sqrt(srs_variance),
        "paired_empty_tail_bound": exp(-2 * risk_count / largest_cell),
        "srs_empty_tail_probability": comb(n - risk_count, m) / comb(n, m),
    }


def main(output):
    data = pd.read_csv(ROOT / "data/rotterdam.csv")
    data = data[data.nodes > 0]
    assert len(data) == 1546 and data.pid.is_unique
    recur = data.recur.eq(1)
    late_death = ~recur & data.death.eq(1) & data.rtime.lt(data.dtime)
    event = recur | (data.death.eq(1) & ~late_death)
    time = np.where(recur | late_death, data.rtime, data.dtime)
    value, scores = rmst_scores(time, event.to_numpy(), HORIZON)
    second_moment = float(np.mean(scores**2))
    observed_risk = int(np.count_nonzero(time >= HORIZON))
    risk_count = floor(TARGET_SIZE * observed_risk / len(time))
    z = NormalDist().inv_cdf(0.975)
    candidates = [planning_row(m, second_moment, risk_count, z) for m in range(2, 2 * (TARGET_SIZE // 3) + 1, 2)]
    evaluated = [row for row in candidates if row["m"] in EVALUATED_BUDGETS]
    recommendations = []
    for half_width in (60, 90, 120):
        passing = [
            row["m"] for row in candidates
            if max(row["paired_half_width"], row["srs_half_width"]) <= half_width
            and max(row["paired_empty_tail_bound"], row["srs_empty_tail_probability"]) <= 0.01
        ]
        recommendations.append({
            "half_width_days": half_width,
            "smallest_even_budget": next(iter(passing), None),
            "first_evaluated_budget": next((m for m in passing if m in EVALUATED_BUDGETS), None),
        })
    record = dict(
        external_cohort_size=len(time), external_events=int(event.sum()), external_late_deaths_censored=int(late_death.sum()),
        external_risk_count_at_horizon=observed_risk, external_rmst_days=value, external_score_second_moment=second_moment,
        horizon_days=HORIZON, target_size=TARGET_SIZE, working_target_risk_count=risk_count,
        evaluated_budgets=evaluated, recommendations=recommendations,
    )
    serialized = json.dumps(record, indent=2) + "\n"
    with output.open("x", encoding="utf-8") as file:
        file.write(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
