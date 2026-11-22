"""Audit finite-simulation resolution, pilot variability, and transport certificates."""

import argparse
from collections import Counter
import csv
import gzip
import json
from math import floor
from pathlib import Path

import numpy as np

from theory_check_budget import N, PILOT_N, TARGET, TAU, cohort, km_if, planned_candidates


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
    corrected = root / "budget_corrected"
    with gzip.open(corrected / "budget_draws.csv.gz", "rt", newline="") as inputs:
        selected = [row for row in csv.DictReader(inputs)
                    if row["regime"] == "matched" and row["role"] == "planned"]
    assert len(selected) == 1000
    realized_exceed = sum(float(row["half_width"]) > TARGET for row in selected)
    return dict(transport_minimum_triangle_margin=min(margins),
                arm0_design_tail_exceed=design_exceed, arm0_bootstrap_tail_exceed=bootstrap_exceed,
                matched_realized_half_width_exceed=realized_exceed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repetitions = 1000
    seed = 5439022
    cdf_rng, pilot_rng = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(2)]
    result = dict(seed=seed, repetitions=repetitions, **numerical_checks(args.study),
                  mean_same_law_cdf_distance=same_law_cdf_distances(
                      args.study / "bootstrap/reference.npz", cdf_rng, repetitions
                  ),
                  matched_pilot_budget_counts=pilot_variation(pilot_rng, repetitions))
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
