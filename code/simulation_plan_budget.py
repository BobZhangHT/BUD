"""Calibrate one common budget for the twelve frozen simulation settings."""

import argparse
import hashlib
import json
from math import comb, exp, sqrt
from pathlib import Path
from statistics import NormalDist
import sys

import bud_experiment

sys.path.insert(0, str(bud_experiment.LEGACY))
from run_confirmation import SCENES, VALIDATION, generate_cohort, km_if

import numpy as np

N = 960
COHORTS = 20
SEED = 709153


def main(output):
    inputs = {}
    for family, scenes in (("legacy", SCENES), ("new-confirmation", VALIDATION)):
        for scene in scenes:
            moments, risk_counts = [], []
            for cohort in range(COHORTS):
                data = generate_cohort(family, scene, cohort, {"seed": SEED, "n_group": N})
                assert data["tau"] == 3
                arms = [data["group"] == group for group in (0, 1)]
                assert all(mask.sum() == N for mask in arms)
                scores = [km_if(data["time"][mask], data["event"][mask], 3) for mask in arms]
                moments.append([float(np.mean(score**2)) for score in scores])
                risk_counts.append([int(np.count_nonzero(data["time"][mask] >= 3)) for mask in arms])
            inputs[scene] = dict(score_second_moment=np.mean(moments, axis=0).tolist(),
                                 risk_count=np.floor(np.mean(risk_counts, axis=0)).astype(int).tolist())
    candidates = []
    z = NormalDist().inv_cdf(.975)
    for h in range(1, N // 3 + 1):
        if N % h:
            continue
        m, k = 2 * h, N // h
        paired_coeff = k * (k - 2) / (2 * N * (k - 1))
        srs_coeff = (1 / m - 1 / N) * N / (N - 1)
        projections = {}
        for scene, inputs_by_arm in inputs.items():
            moment = sum(inputs_by_arm["score_second_moment"])
            risk = inputs_by_arm["risk_count"]
            paired = [exp(-2 * r / k) for r in risk]
            srs = [comb(N - r, m) / comb(N, m) for r in risk]
            prob = [a + b - a*b for a, b in (paired, srs)]
            projections[scene] = dict(max_half_width=z*sqrt(max(paired_coeff, srs_coeff)*moment),
                                      max_empty_probability=max(prob))
        max_width = max(x["max_half_width"] for x in projections.values())
        max_empty = max(x["max_empty_probability"] for x in projections.values())
        candidates.append(dict(m=m, k=k, max_half_width=max_width, max_empty_probability=max_empty,
                               passes=max_width <= .15 and max_empty <= .01, projections=projections))
    selected = next((row["m"] for row in candidates if row["passes"]), None)
    sources = [Path(__file__), bud_experiment.PROJECT/"handoff/BUD_REFRESH_PROTOCOL.md",
               *bud_experiment.LEGACY.rglob("*.py")]
    result = dict(selected_m=selected, half_width_target=.15, empty_probability_target=.01,
                  pilot_cohorts=COHORTS, pilot_seed=SEED, n_group=N, pilot_inputs=inputs,
                  candidates=candidates,
                  sha256={p.relative_to(bud_experiment.PROJECT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources})
    with output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print("Selected common even budget:", selected, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
