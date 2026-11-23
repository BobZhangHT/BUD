"""Calibrate and evaluate the frozen simulation sampling budgets.

The named stages preserve the two historical budgets and the primary BUD-only
budget.  They use covariates and fixed pilot cohorts only.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
from itertools import batched
import json
from math import ceil, comb, exp, sqrt
from pathlib import Path
from statistics import NormalDist
import sys
from time import perf_counter

import bud_experiment as engine

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


N = 960
PILOT_COHORTS = 20
PILOT_SEED = 709153
BUD = "BUD-fullspace-onepass"
METHODS = (BUD, "PC1-X")
HALF_WIDTH = 0.15
EMPTY_PROBABILITY = 0.01
Z_975 = NormalDist().inv_cdf(0.975)
ROOT = engine.PROJECT
MARGINAL_PLAN = ROOT / "results/budget_refresh/simulation_budget_plan.json"
COMMON_PLAN = ROOT / "results/budget_optimization/simulation_plan.json"


def _sha256(paths):
    return {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _legacy_runner():
    return engine._import_runner("run_confirmation")


def _scenes(runner):
    return [
        (family, scene)
        for family, names in (
            ("legacy", runner.SCENES),
            ("new-confirmation", runner.VALIDATION),
        )
        for scene in names
    ]


def _cell_sizes(n, h):
    sizes = np.full(h, n // h, dtype=int)
    sizes[: n % h] += 1
    return sizes


def _quantile_labels(scores, h):
    order = np.argsort(scores, kind="stable")
    labels = np.empty(len(order), dtype=int)
    labels[order] = np.repeat(np.arange(h), _cell_sizes(len(order), h))
    return labels


def _python_assignment(coordinates, anchors, capacities):
    distances = cdist(coordinates, anchors, metric="sqeuclidean")
    slots = np.repeat(np.arange(len(anchors)), capacities)
    row, col = linear_sum_assignment(distances[:, slots])
    labels = np.empty(len(coordinates), dtype=int)
    labels[row] = slots[col]
    return labels, float(distances[row, labels[row]].mean())


def _near_equal_design(x, m, method):
    if len(x) % (m // 2) == 0 or method == "SRS":
        return _locked_dispatch(x, m, method)
    if method not in METHODS:
        raise ValueError(f"Unsupported near-equal design: {method}")
    started = perf_counter()
    n, p = x.shape
    h = m // 2
    capacities = _cell_sizes(n, h)
    allocation = np.full(h, 2, dtype=int)
    designs = sys.modules["designs"]
    standardized, pc, _, q = designs.standardized_pca(x)
    info = dict(
        variant=method,
        N=n,
        m=m,
        p=p,
        q85=q,
        H=h,
        min_population_cell=int(capacities.min()),
        max_population_cell=int(capacities.max()),
    )
    if method == "PC1-X":
        labels = _quantile_labels(pc[:, 0], h)
    else:
        anchors = designs.inverse_quantile_anchors(
            pc[:, :q], designs.glp_uniform_design(h, q)
        )
        initial, _ = _assign(pc[:, :q], anchors, capacities)
        centroids = np.vstack(
            [standardized[initial == j].mean(axis=0) for j in range(h)]
        )
        labels, _ = _assign(standardized, centroids, capacities)
        before = designs._within_scatter(standardized, initial)
        after = designs._within_scatter(standardized, labels)
        assert after <= before + 1e-10 * max(1.0, before)
        info.update(full_before=before, full_after=after)
    assert np.array_equal(np.bincount(labels), capacities) and np.all(capacities >= 3)
    info["seconds"] = perf_counter() - started
    return labels, allocation, info


def _initialize(_module, backend, bootstrap):
    """Install backend functions inside each spawned Windows worker."""
    global _locked_dispatch, _assign
    runner = _legacy_runner()
    designs = sys.modules["designs"]
    _assign = _python_assignment
    if backend == "c":
        from backend import DesignBackend

        _assign = DesignBackend(
            designs.glp_uniform_design, designs.inverse_quantile_anchors
        ).capacitated_assignment
    engine._patch_backend(runner, engine._new_backend(backend), bootstrap=bootstrap)
    _locked_dispatch = runner.design_dispatch
    runner.design_dispatch = _near_equal_design
    sys.modules["historical_check"].equal_quantile_labels = _quantile_labels
    engine._WORKER_MODULE = runner


def marginal_calibration(output):
    """Write the archived common-budget marginal screening record."""
    runner = _legacy_runner()
    inputs = {}
    for family, scenes in (
        ("legacy", runner.SCENES),
        ("new-confirmation", runner.VALIDATION),
    ):
        for scene in scenes:
            moments, risks = [], []
            for cohort in range(PILOT_COHORTS):
                data = runner.generate_cohort(
                    family, scene, cohort, {"seed": PILOT_SEED, "n_group": N}
                )
                assert data["tau"] == 3
                arms = [data["group"] == group for group in (0, 1)]
                assert all(mask.sum() == N for mask in arms)
                scores = [
                    runner.km_if(data["time"][mask], data["event"][mask], 3)
                    for mask in arms
                ]
                moments.append([float(np.mean(score**2)) for score in scores])
                risks.append(
                    [int(np.count_nonzero(data["time"][mask] >= 3)) for mask in arms]
                )
            inputs[scene] = {
                "score_second_moment": np.mean(moments, axis=0).tolist(),
                "risk_count": np.floor(np.mean(risks, axis=0)).astype(int).tolist(),
            }
    candidates = []
    for h in range(1, N // 3 + 1):
        if N % h:
            continue
        m, k = 2 * h, N // h
        paired = k * (k - 2) / (2 * N * (k - 1))
        srs = (1 / m - 1 / N) * N / (N - 1)
        projections = {}
        for scene, values in inputs.items():
            moment, risk = sum(values["score_second_moment"]), values["risk_count"]
            pair_empty = [exp(-2 * r / k) for r in risk]
            srs_empty = [comb(N - r, m) / comb(N, m) for r in risk]
            probabilities = [a + b - a * b for a, b in zip(pair_empty, srs_empty)]
            projections[scene] = {
                "max_half_width": Z_975 * sqrt(max(paired, srs) * moment),
                "max_empty_probability": max(probabilities),
            }
        width, empty = (
            max(x["max_half_width"] for x in projections.values()),
            max(x["max_empty_probability"] for x in projections.values()),
        )
        candidates.append(
            {
                "m": m,
                "k": k,
                "max_half_width": width,
                "max_empty_probability": empty,
                "passes": width <= HALF_WIDTH and empty <= EMPTY_PROBABILITY,
                "projections": projections,
            }
        )
    result = {
        "selected_m": next((row["m"] for row in candidates if row["passes"]), None),
        "half_width_target": HALF_WIDTH,
        "empty_probability_target": EMPTY_PROBABILITY,
        "pilot_cohorts": PILOT_COHORTS,
        "pilot_seed": PILOT_SEED,
        "n_group": N,
        "pilot_inputs": inputs,
        "candidates": candidates,
        "sha256": _sha256([Path(__file__), *engine.LEGACY.rglob("*.py")]),
    }
    with output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print("Selected common even budget:", result["selected_m"], flush=True)


def _risk_either(probabilities):
    return probabilities[0] + (1 - probabilities[0]) * probabilities[1]


def _srs_projections(pilots, m):
    return {
        scene: {
            "half_width": Z_975
            * sqrt((1 / m - 1 / N) * N / (N - 1) * sum(row["score_second_moment"])),
            "empty_probability": _risk_either(
                [comb(N - r, m) / comb(N, m) for r in row["risk_count"]]
            ),
        }
        for scene, row in pilots.items()
    }


def _feasible(rows):
    return all(
        row["half_width"] <= HALF_WIDTH
        and row["empty_probability"] <= EMPTY_PROBABILITY
        for row in rows
    )


def _widest(rows):
    scene = max(rows, key=lambda name: rows[name]["half_width"])
    return {
        "scene": scene,
        **rows[scene],
        "max_empty_probability": max(row["empty_probability"] for row in rows.values()),
    }


def _pilot_cell_variance(task):
    family, scene, cohort, m = task
    runner = engine._WORKER_MODULE
    data = runner.generate_cohort(
        family, scene, cohort, {"seed": PILOT_SEED, "n_group": N}
    )
    results = {method: 0.0 for method in METHODS}
    for group in (0, 1):
        arm = data["group"] == group
        scores = runner.km_if(data["time"][arm], data["event"][arm], data["tau"])
        for method in METHODS:
            labels, allocation, _ = _near_equal_design(data["x"][arm], m, method)
            results[method] += runner.design_variance(scores, labels, allocation)
    return scene, cohort, results


def common_cell_calibration(output, backend, workers):
    """Calibrate the historical common near-equal-cell budget."""
    runner = _legacy_runner()
    historical = json.loads(MARGINAL_PLAN.read_bytes())
    assert (
        historical["pilot_cohorts"] == PILOT_COHORTS
        and historical["pilot_seed"] == PILOT_SEED
        and historical["n_group"] == N
    )
    pilots = historical["pilot_inputs"]
    candidates = range(4, 2 * (N // 3) + 1, 2)
    floor = next(
        m for m in candidates if _feasible(_srs_projections(pilots, m).values())
    )
    print(f"SRS feasibility floor: {floor}", flush=True)
    examined = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize,
        initargs=("run_confirmation", backend, False),
    ) as pool:
        for m in candidates:
            if m < floor:
                continue
            srs = _srs_projections(pilots, m)
            k = ceil(N / (m // 2))
            tail = {
                name: _risk_either([exp(-2 * r / k) for r in row["risk_count"]])
                for name, row in pilots.items()
            }
            if max(tail.values()) > EMPTY_PROBABILITY:
                examined.append(
                    {"m": m, "passes": False, "max_paired_empty": max(tail.values())}
                )
                continue
            values = {
                scene: {method: [] for method in METHODS}
                for _, scene in _scenes(runner)
            }
            jobs = [
                (family, scene, cohort, m)
                for family, scene in _scenes(runner)
                for cohort in range(PILOT_COHORTS)
            ]
            for scene, _, estimates in pool.map(
                _pilot_cell_variance, jobs, chunksize=4
            ):
                for method, estimate in estimates.items():
                    values[scene][method].append(estimate)
            projection = {"SRS": srs}
            for method in METHODS:
                projection[method] = {
                    scene: {
                        "half_width": Z_975 * sqrt(np.mean(values[scene][method])),
                        "empty_probability": tail[scene],
                    }
                    for scene in pilots
                }
            passes = all(_feasible(rows.values()) for rows in projection.values())
            examined.append(
                {
                    "m": m,
                    "projections": projection,
                    "passes": passes,
                    "limiting": {
                        method: _widest(rows) for method, rows in projection.items()
                    },
                    "pilot_variance_draws": values,
                }
            )
            print(f"Candidate m={m}: passes={passes}", flush=True)
            if passes:
                break
    selected = next(row["m"] for row in examined if row["passes"])
    report = {
        "selected_m": selected,
        "previous_m": historical["selected_m"],
        "backend": backend,
        "srs_floor": floor,
        "srs_predecessor": _widest(_srs_projections(pilots, floor - 2)),
        "half_width_target": HALF_WIDTH,
        "empty_probability_target": EMPTY_PROBABILITY,
        "n_group": N,
        "pilot_seed": PILOT_SEED,
        "pilot_cohorts": PILOT_COHORTS,
        "candidates": examined,
        "sha256": _sha256([Path(__file__), MARGINAL_PLAN, *engine.LEGACY.glob("*.py")]),
    }
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print("Selected simulation budget:", selected, examined[-1]["limiting"], flush=True)


def _bud_variance(task):
    family, scene, cohort, sizes = task
    runner = engine._WORKER_MODULE
    data = runner.generate_cohort(
        family, scene, cohort, {"seed": PILOT_SEED, "n_group": N}
    )
    result = np.zeros(len(sizes))
    for group in (0, 1):
        arm = data["group"] == group
        scores, x = (
            runner.km_if(data["time"][arm], data["event"][arm], data["tau"]),
            data["x"][arm],
        )
        for index, m in enumerate(sizes):
            labels, allocation, _ = _near_equal_design(x, m, BUD)
            result[index] += runner.design_variance(scores, labels, allocation)
    return scene, cohort, sizes, result.tolist()


def _projection(values):
    values = np.asarray(values)
    mean = float(values.mean())
    return {
        "half_width": float(Z_975 * np.sqrt(mean)),
        "mcse": float(Z_975 * values.std(ddof=1) / (2 * np.sqrt(len(values) * mean))),
        "pilot_variances": values.tolist(),
    }


def bud_calibration(output, backend, workers):
    """Select the primary BUD-only budget after the historical screen."""
    runner = _legacy_runner()
    scenes = {scene: family for family, scene in _scenes(runner)}
    previous, marginals = (
        json.loads(COMMON_PLAN.read_bytes()),
        json.loads(MARGINAL_PLAN.read_bytes())["pilot_inputs"],
    )
    largest = previous["selected_m"]
    screen = previous["candidates"][-1]["limiting"][BUD]["scene"]
    assert (
        previous["pilot_seed"] == PILOT_SEED
        and previous["pilot_cohorts"] == PILOT_COHORTS
    )
    candidates = {}
    for m in range(4, largest + 1, 2):
        k = ceil(N / (m // 2))
        worst = max(
            _risk_either([exp(-2 * r / k) for r in row["risk_count"]])
            for row in marginals.values()
        )
        candidates[m] = {"max_empty_probability": worst}
        if worst > EMPTY_PROBABILITY:
            candidates[m].update(passes=False, reason="horizon")
    eligible = [m for m, row in candidates.items() if "reason" not in row]
    print(
        f"Screening {len(eligible)} sizes with the {screen} pilot cohorts", flush=True
    )

    def scan(pool, names, sizes):
        tasks = [
            (scenes[name], name, cohort, batch)
            for name in names
            for batch in batched(sizes, 5)
            for cohort in range(PILOT_COHORTS)
        ]
        estimates = defaultdict(dict)
        for scene, cohort, sizes, result in pool.map(_bud_variance, tasks, chunksize=2):
            for m, value in zip(sizes, result, strict=True):
                estimates[scene, m][cohort] = value
        for (name, m), draws in estimates.items():
            assert len(draws) == PILOT_COHORTS
            candidates[m].setdefault("projections", {})[name] = _projection(
                [draws[index] for index in range(PILOT_COHORTS)]
            )

    selected = None
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize,
        initargs=("run_confirmation", backend, False),
    ) as pool:
        for block in batched(eligible, 10):
            scan(pool, [screen], block)
            print(f"Screened through {block[-1]} per group", flush=True)
            for m in block:
                row = candidates[m]
                if row["projections"][screen]["half_width"] > HALF_WIDTH:
                    row.update(passes=False, reason="screening scenario")
                    continue
                scan(pool, [name for name in scenes if name != screen], [m])
                worst = max(
                    row["projections"],
                    key=lambda name: row["projections"][name]["half_width"],
                )
                passes = row["projections"][worst]["half_width"] <= HALF_WIDTH
                row.update(
                    passes=passes,
                    reason="passes" if passes else "other scenarios",
                    limiting_scene=worst,
                )
                if passes:
                    selected = m
                    break
            if selected is not None:
                break
    if selected is None:
        raise RuntimeError(
            "No candidate passed despite the previous feasible BUD projection"
        )
    assert all(
        row.get("passes") is False for m, row in candidates.items() if m < selected
    )
    report = {
        "selected_m": selected,
        "previous_common_m": largest,
        "screening_scene": screen,
        "pilot_n": N,
        "pilot_seed": PILOT_SEED,
        "pilot_cohorts": PILOT_COHORTS,
        "backend": backend,
        "half_width_target": HALF_WIDTH,
        "empty_probability_target": EMPTY_PROBABILITY,
        "evaluated": {m: row for m, row in candidates.items() if m <= selected},
        "sha256": _sha256([Path(__file__), COMMON_PLAN, MARGINAL_PLAN]),
    }
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    winner = candidates[selected]
    print(
        f"Selected BUD-only budget: {selected}; limiting scene {winner['limiting_scene']}; half-width {winner['projections'][winner['limiting_scene']]['half_width']:.6f}",
        flush=True,
    )


def evaluate(
    plan_path,
    output,
    backend,
    workers,
    *,
    cohorts=50,
    draws=20,
    bootstrap_replicates=1000,
):
    """Evaluate a selected near-equal-cell budget with locked process isolation."""
    m = json.loads(plan_path.read_bytes())["selected_m"]
    args = argparse.Namespace(
        out=output,
        command="simulation",
        cohorts=cohorts,
        draws=draws,
        bootstrap_replicates=bootstrap_replicates,
        n_group=N,
        m_group=m,
        workers=workers,
        seed=33891027,
        backend=backend,
    )
    native = engine._new_backend(backend)
    engine._worker_initializer = _initialize
    output = engine._prepare_out(
        output,
        "simulation-cell-budget",
        args,
        native,
        [Path(__file__), plan_path.resolve(), engine.LEGACY / "run_confirmation.py"],
    )
    started = perf_counter()
    engine.run_simulation(args, native, output)
    engine._finish_out(output, started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "marginal-calibration",
            "common-cell-calibration",
            "bud-calibration",
            "evaluate",
        ),
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="New JSON plan or evaluation directory."
    )
    parser.add_argument(
        "--plan", type=Path, help="Selected plan required for the evaluation stage."
    )
    parser.add_argument("--backend", choices=("c", "python"), default="c")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cohorts", type=int, default=50)
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    if (
        args.workers < 1
        or args.cohorts < 1
        or args.draws < 1
        or args.bootstrap_replicates < 2
    ):
        parser.error(
            "workers, cohorts, and draws must be positive; bootstrap replicates must be at least two."
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.stage == "marginal-calibration":
        marginal_calibration(args.out)
    elif args.stage == "common-cell-calibration":
        common_cell_calibration(args.out, args.backend, args.workers)
    elif args.stage == "bud-calibration":
        bud_calibration(args.out, args.backend, args.workers)
    else:
        if args.plan is None:
            parser.error("--plan is required for the evaluate stage.")
        evaluate(
            args.plan,
            args.out,
            args.backend,
            args.workers,
            cohorts=args.cohorts,
            draws=args.draws,
            bootstrap_replicates=args.bootstrap_replicates,
        )


if __name__ == "__main__":
    main()
