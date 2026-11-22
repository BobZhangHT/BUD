"""Search for the smallest BUD budget across the twelve pilot scenarios."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import batched
import hashlib
import json
from math import ceil, exp
from pathlib import Path

import simulation_cell_budget as base

import numpy as np


ROOT = base.ROOT
PREVIOUS_PLAN = ROOT / "results/budget_optimization/simulation_plan.json"


def pilot_variances(task):
    family, scene, cohort, sizes = task
    runner = base.engine._WORKER_MODULE
    data = runner.generate_cohort(family, scene, cohort, {"seed": base.SEED, "n_group": base.N})
    result = np.zeros(len(sizes))
    for group in (0, 1):
        mask = data["group"] == group
        scores = runner.km_if(data["time"][mask], data["event"][mask], data["tau"])
        x = data["x"][mask]
        for i, m in enumerate(sizes):
            labels, allocation, _ = base.near_equal_design(x, m, base.BUD)
            result[i] += runner.design_variance(scores, labels, allocation)
    return scene, cohort, sizes, result.tolist()


def projection(variances):
    values = np.asarray(variances)
    mean = float(values.mean())
    half_width = base.Z * np.sqrt(mean)
    mcse = base.Z * values.std(ddof=1) / (2 * np.sqrt(len(values) * mean))
    return dict(half_width=float(half_width), mcse=float(mcse),
                pilot_variances=values.tolist())


def plan(path, backend, workers):
    runner = base.engine._import_runner("run_confirmation")
    scenes = {scene: family for family, names in
              (("legacy", runner.SCENES), ("new-confirmation", runner.VALIDATION))
              for scene in names}
    previous = json.loads(PREVIOUS_PLAN.read_bytes())
    marginals = json.loads(base.BASE_PLAN.read_bytes())["pilot_inputs"]
    largest = previous["selected_m"]
    screen = previous["candidates"][-1]["limiting"][base.BUD]["scene"]
    assert largest <= 2 * (base.N // 3)
    assert previous["pilot_seed"] == base.SEED and previous["pilot_cohorts"] == base.COHORTS
    assert previous["half_width_target"] == base.HALF_WIDTH and previous["empty_probability_target"] == base.EMPTY
    assert previous["candidates"][-1]["limiting"][base.BUD]["half_width"] <= base.HALF_WIDTH
    candidates = {}
    for m in range(4, largest + 1, 2):
        kmax = ceil(base.N / (m // 2))
        risk = {scene: base.risk_either([exp(-2 * r / kmax) for r in row["risk_count"]])
                for scene, row in marginals.items()}
        worst = max(risk.values())
        candidates[m] = dict(max_empty_probability=worst)
        if worst > base.EMPTY:
            candidates[m].update(passes=False, reason="horizon")
    eligible = [m for m, row in candidates.items() if "reason" not in row]
    print(f"Screening {len(eligible)} sizes with the {screen} pilot cohorts", flush=True)

    def scan(pool, chosen_scenes, sizes):
        tasks = [(scenes[name], name, cohort, batch)
                 for name in chosen_scenes for batch in batched(sizes, 5)
                 for cohort in range(base.COHORTS)]
        estimates = defaultdict(dict)
        for scene, cohort, sizes, result in pool.map(pilot_variances, tasks, chunksize=2):
            for m, v in zip(sizes, result, strict=True):
                estimates[scene, m][cohort] = v
        for (name, m), draws in estimates.items():
            assert len(draws) == base.COHORTS
            candidates[m].setdefault("projections", {})[name] = projection(
                [draws[i] for i in range(base.COHORTS)]
            )

    chosen = None
    with ProcessPoolExecutor(max_workers=workers, initializer=base.initialize,
                             initargs=("run_confirmation", backend, False)) as pool:
        # Process a small group of sizes at a time to retain parallel work
        # and limit calculations beyond the first passing candidate.
        for block in batched(eligible, 10):
            scan(pool, [screen], block)
            print(f"Screened through {block[-1]} per group", flush=True)
            for m in block:
                row = candidates[m]
                if row["projections"][screen]["half_width"] > base.HALF_WIDTH:
                    row.update(passes=False, reason="screening scenario")
                    continue
                scan(pool, [name for name in scenes if name != screen], [m])
                values = row["projections"]
                worst = max(values, key=lambda name: values[name]["half_width"])
                passes = values[worst]["half_width"] <= base.HALF_WIDTH
                row.update(passes=passes, reason="passes" if passes else "other scenarios", limiting_scene=worst)
                if passes:
                    chosen = m
                    break
            if chosen is not None:
                break
    if chosen is None:
        raise RuntimeError("No candidate passed despite the previous feasible BUD projection")
    assert all(row.get("passes") is False for m, row in candidates.items() if m < chosen)
    report = dict(selected_m=chosen, previous_common_m=largest, screening_scene=screen,
                  pilot_n=base.N, pilot_seed=base.SEED, pilot_cohorts=base.COHORTS,
                  backend=backend, half_width_target=base.HALF_WIDTH, empty_probability_target=base.EMPTY,
                  evaluated={m: row for m, row in candidates.items() if m <= chosen},
                  sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), Path(base.__file__), PREVIOUS_PLAN, base.BASE_PLAN)})
    with path.open("x", encoding="utf8") as file:
        json.dump(report, file, indent=2)
    win = candidates[chosen]
    print(f"Selected BUD-only budget: {chosen}; limiting scene {win['limiting_scene']}; "
          f"half-width {win['projections'][win['limiting_scene']]['half_width']:.6f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--backend", choices=("c", "python"), default="c")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--cohorts", type=int, default=50, help="Independent evaluation cohorts per scenario.")
    parser.add_argument("--draws", type=int, default=20, help="Phase-II draws per evaluation cohort.")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.cohorts < 1 or args.draws < 1 or args.bootstrap_replicates < 2:
        parser.error("Evaluation requires positive cohort and draw counts and at least two bootstrap replicates.")
    if not args.plan.exists():
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        plan(args.plan, args.backend, args.workers)
    if args.evaluate:
        base.evaluate(args.plan, args.evaluate, args.backend, args.workers,
                      cohorts=args.cohorts, draws=args.draws,
                      bootstrap_replicates=args.bootstrap_replicates)
