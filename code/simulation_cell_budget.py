"""Plan and evaluate the common near-equal simulation budget."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from math import ceil, comb, exp, sqrt
from pathlib import Path
from statistics import NormalDist
import sys
from time import perf_counter

import bud_experiment as engine
from simulation_plan_budget import COHORTS, N, SEED

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


BUD = "BUD-fullspace-onepass"
METHODS = (BUD, "PC1-X")
HALF_WIDTH = .15
EMPTY = .01
Z = NormalDist().inv_cdf(.975)
ROOT = engine.PROJECT
BASE_PLAN = ROOT / "results/budget_refresh_20261122/simulation_budget_plan.json"


def cell_sizes(n, h):
    sizes = np.full(h, n // h, dtype=int)
    sizes[:n % h] += 1
    return sizes


def quantile_labels(scores, h):
    order = np.argsort(scores, kind="stable")
    labels = np.empty(len(order), dtype=int)
    labels[order] = np.repeat(np.arange(h), cell_sizes(len(order), h))
    return labels


def python_assignment(coords, anchors, capacities):
    distances = cdist(coords, anchors, metric="sqeuclidean")
    slots = np.repeat(np.arange(len(anchors)), capacities)
    row, col = linear_sum_assignment(distances[:, slots])
    labels = np.empty(len(coords), dtype=int)
    labels[row] = slots[col]
    return labels, float(distances[row, labels[row]].mean())


def near_equal_design(x, m, method):
    if len(x) % (m // 2) == 0 or method == "SRS":
        return _locked_dispatch(x, m, method)
    if method not in METHODS:
        raise ValueError(f"Unsupported near-equal design: {method}")
    start = perf_counter()
    n, p = x.shape
    h = m // 2
    capacities = cell_sizes(n, h)
    allocation = np.full(h, 2, dtype=int)
    scaling = sys.modules["designs"]
    z, pc, _, q = scaling.standardized_pca(x)
    info = dict(variant=method, N=n, m=m, p=p, q85=q, H=h,
                min_population_cell=int(capacities.min()), max_population_cell=int(capacities.max()))
    if method == "PC1-X":
        labels = quantile_labels(pc[:, 0], h)
    else:
        anchors = scaling.inverse_quantile_anchors(pc[:, :q], scaling.glp_uniform_design(h, q))
        initial, _ = _assign(pc[:, :q], anchors, capacities)
        centroids = np.vstack([z[initial == j].mean(axis=0) for j in range(h)])
        labels, _ = _assign(z, centroids, capacities)
        before = scaling._within_scatter(z, initial)
        after = scaling._within_scatter(z, labels)
        assert after <= before + 1e-10 * max(1., before)
        info.update(full_before=before, full_after=after)
    assert np.array_equal(np.bincount(labels), capacities) and np.all(capacities >= 3)
    info["seconds"] = perf_counter() - start
    return labels, allocation, info


def initialize(_module, backend, bootstrap):
    global _locked_dispatch, _assign
    runner = engine._import_runner("run_confirmation")
    scaling = sys.modules["designs"]
    _assign = python_assignment
    if backend == "c":
        from design_backend import DesignBackend
        _assign = DesignBackend(
            scaling.glp_uniform_design, scaling.inverse_quantile_anchors
        ).capacitated_assignment
    engine._patch_backend(runner, engine._new_backend(backend), bootstrap=bootstrap)
    _locked_dispatch = runner.design_dispatch
    runner.design_dispatch = near_equal_design
    sys.modules["historical_check"].equal_quantile_labels = quantile_labels
    engine._WORKER_MODULE = runner


def pilot_cohort(task):
    family, scene, cohort, m = task
    runner = engine._WORKER_MODULE
    data = runner.generate_cohort(family, scene, cohort, {"seed": SEED, "n_group": N})
    results = {method: 0. for method in METHODS}
    for group in (0, 1):
        mask = data["group"] == group
        scores = runner.km_if(data["time"][mask], data["event"][mask], data["tau"])
        for method in METHODS:
            labels, allocation, _ = near_equal_design(data["x"][mask], m, method)
            results[method] += runner.design_variance(scores, labels, allocation)
    return scene, cohort, results


def risk_either(probabilities):
    first, second = probabilities
    return first + (1 - first) * second


def srs_inputs(pilots, m):
    projections = {}
    for scene, pilot in pilots.items():
        risk = pilot["risk_count"]
        variance = (1 / m - 1 / N) * (N / (N - 1)) * sum(pilot["score_second_moment"])
        probability = risk_either([comb(N - r, m) / comb(N, m) for r in risk])
        projections[scene] = dict(half_width=Z * sqrt(variance), empty_probability=probability)
    return projections


def feasible(projections):
    return all(row["half_width"] <= HALF_WIDTH and row["empty_probability"] <= EMPTY
               for row in projections)


def wide_projection(projections):
    scene = max(projections, key=lambda name: projections[name]["half_width"])
    return {"scene": scene, **projections[scene],
            "max_empty_probability": max(p["empty_probability"] for p in projections.values())}


def plan(path, backend, workers):
    runner = engine._import_runner("run_confirmation")
    families = [(family, scene) for family, names in
                (("legacy", runner.SCENES), ("new-confirmation", runner.VALIDATION)) for scene in names]
    old = json.loads(BASE_PLAN.read_bytes())
    assert old["pilot_cohorts"] == COHORTS and old["pilot_seed"] == SEED and old["n_group"] == N
    pilots = old["pilot_inputs"]
    candidates = range(4, 2 * (N // 3) + 1, 2)
    floor = next(m for m in candidates if feasible(srs_inputs(pilots, m).values()))
    print(f"SRS feasibility floor: {floor}", flush=True)
    examined = []
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize,
                             initargs=("run_confirmation", backend, False)) as pool:
        for m in candidates:
            if m < floor:
                continue
            srs = srs_inputs(pilots, m)
            k = ceil(N / (m // 2))
            tail = {name: risk_either([exp(-2 * r / k) for r in val["risk_count"]])
                    for name, val in pilots.items()}
            if max(tail.values()) > EMPTY:
                examined.append(dict(m=m, passes=False, max_paired_empty=max(tail.values())))
                continue
            jobs = [(family, scene, i, m) for family, scene in families for i in range(COHORTS)]
            variances = {scene: {method: [] for method in METHODS} for _, scene in families}
            for scene, cohort, values in pool.map(pilot_cohort, jobs, chunksize=4):
                for method, variance in values.items():
                    variances[scene][method].append(variance)
            projection = {"SRS": srs}
            for method in METHODS:
                projection[method] = {
                    scene: dict(half_width=Z * sqrt(np.mean(variances[scene][method])),
                                empty_probability=tail[scene]) for scene in pilots
                }
            passes = all(feasible(rows.values()) for rows in projection.values())
            examined.append(dict(m=m, projections=projection, passes=passes,
                                 limiting={method: wide_projection(rows) for method, rows in projection.items()},
                                 pilot_variance_draws=variances))
            print(f"Candidate m={m}: passes={passes}", flush=True)
            if passes:
                break
    chosen = next(row["m"] for row in examined if row["passes"])
    report = dict(selected_m=chosen, previous_m=old["selected_m"], backend=backend,
                  srs_floor=floor, srs_predecessor=wide_projection(srs_inputs(pilots, floor - 2)),
                  half_width_target=HALF_WIDTH, empty_probability_target=EMPTY,
                  n_group=N, pilot_seed=SEED, pilot_cohorts=COHORTS, candidates=examined,
                  sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), BASE_PLAN, *engine.LEGACY.glob("*.py"))})
    with path.open("x", encoding="utf8") as file:
        json.dump(report, file, indent=2)
    print("Selected simulation budget:", chosen, examined[-1]["limiting"], flush=True)


def evaluate(plan_path, out, backend_kind, workers):
    selection = json.loads(plan_path.read_bytes())
    m = selection["selected_m"]
    args = argparse.Namespace(out=out, command="simulation", cohorts=50, draws=20,
                              bootstrap_replicates=1000, n_group=N, m_group=m,
                              workers=workers, seed=33891027, backend=backend_kind)
    backend = engine._new_backend(backend_kind)
    engine._worker_initializer = initialize
    out = engine._prepare_out(out, "simulation-cell-budget", args, backend,
                              [Path(__file__), plan_path.resolve(),
                               engine.LEGACY / "run_confirmation.py"])
    started = perf_counter()
    engine.run_simulation(args, backend, out)
    engine._finish_out(out, started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--backend", choices=("c", "python"), default="c")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if not args.plan.exists():
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        plan(args.plan, args.backend, args.workers)
    if args.evaluate:
        evaluate(args.plan, args.evaluate, args.backend, args.workers)
