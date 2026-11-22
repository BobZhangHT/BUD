"""Freeze a BUD-only GBSG budget, then evaluate all three methods at that size."""

import argparse
import json
from math import sqrt
import traceback
from unittest.mock import patch

import gbsg_cell_budget_planning as archived
from gbsg_cell_budget_planning import (
    DRAWS, EMPTY_LIMIT, GBSG, HALF_WIDTH, HORIZON, MC_DRAWS, METHODS, ROOT,
    RESAMPLES, TARGET_SIZE, Z, driver, software_versions, utc_now, write_json,
)
from km_backend import KMBackend

import numpy as np
import pandas as pd


OUT = ROOT / "results/bud_budget/gbsg"
PLAN_PROTOCOL = "gbsg_plan_protocol.json"
PLAN_LOCK = "gbsg_plan_lock.json"
SNAPSHOT = "gbsg_plan_covariates.csv"
PARTITIONS = "gbsg_plan_frozen_partitions.csv"
SCAN = "gbsg_plan_candidates.csv"
WORKING_RISK = 277


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verified_sources():
    return {**archived.source_hashes(),
            "code/gbsg_bud_only_budget.py": driver.sha(ROOT / "code/gbsg_bud_only_budget.py")}


def scan_bud(x, forecast, calibration):
    rows = []
    sigma2 = calibration["residual_second_moment"]
    risk = calibration["projected_risk_count"]
    for m in range(4, 2 * (TARGET_SIZE // 3) + 1, 2):
        h = m // 2
        base, larger = divmod(TARGET_SIZE, h)
        capacities = np.full(h, base, dtype=int)
        capacities[:larger] += 1
        weights = (capacities / TARGET_SIZE)**2 * (1 / 2 - 1 / capacities)
        variance_floor = sigma2 * float(weights.sum())
        half_width_floor = Z * sqrt(variance_floor)
        # Screening dimension is BUD's own archived paired horizon bound.
        baseline = archived.planning_row(m, calibration["marginal_score_second_moment"], risk, Z)
        empty_bound = baseline["paired_empty_tail_bound"]
        row = dict(m=m, min_capacity=base, max_capacity=int(capacities.max()),
                   residual_variance_floor=variance_floor, half_width_floor=half_width_floor,
                   paired_empty_bound=empty_bound, bud_projected_variance=None,
                   bud_half_width=None, bud_pass=False)
        if empty_bound > EMPTY_LIMIT:
            row["status"] = "horizon_screen"
        elif half_width_floor > HALF_WIDTH:
            row["status"] = "variance_screen"
        else:
            designs, _ = archived.build(x, m)
            labels, allocation = designs["BUD"]
            np.testing.assert_array_equal(np.bincount(labels), capacities)
            np.testing.assert_array_equal(allocation, np.full(h, 2))
            within = np.array([np.var(forecast[labels == cell], ddof=1) for cell in range(h)])
            variance = float(np.sum(weights * (within + sigma2)))
            half_width = Z * sqrt(variance)
            row.update(bud_projected_variance=variance, bud_half_width=half_width,
                       bud_pass=half_width <= HALF_WIDTH, status="exact_projection")
        rows.append(row)
    return rows


def plan():
    # Regenerate the external-pilot calibration and this BUD-only frozen plan
    # in the same standalone run.  No historical result lock is required.
    protocol = dict(
        created_utc=utc_now(), sources=verified_sources(), versions=software_versions(),
        horizon=HORIZON, target_size=TARGET_SIZE, half_width_days=HALF_WIDTH,
        max_empty_risk=EMPTY_LIMIT, working_risk_count=WORKING_RISK,
        pilot_endpoint="Node-positive Rotterdam; conservative recurrence/death construction",
        forecast="Fixed out-of-fold score regression on age, menopause, log nodes, progesterone, and estrogen receptors",
        target_design=GBSG, seed=driver.BASE_SEED, phase2_draws=DRAWS,
        bootstrap_resamples=RESAMPLES, paired_mc_resamples=MC_DRAWS,
        scope="Retrospective GBSG evaluation with a separately recorded external planning pilot.",
        scan="Screen every even m from 4 through 2*floor(N/3). The frozen builder "
             "requires m>=4 and cell capacities >=3. For every budget surviving "
             "the screens, calculate the BUD working projection using its exact "
             "partition. Select the smallest budget passing both BUD thresholds. "
             "Compute comparator projections only after selection, at the selected "
             "budget and its immediate admissible predecessor.",
        variance_screen="For fixed capacities and two draws per cell, subtracting "
                        "the residual-only variance from the working projected variance "
                        "leaves a sum of nonnegative forecast means' within-cell sample "
                        "variances with nonnegative finite-population coefficients. "
                        "Thus sigma^2 * sum((k_h/N)^2*(1/2 - 1/k_h)) is a lower bound "
                        "on the working projection. Screen if its normal-reference "
                        "half-width exceeds the target.",
        tail="Use exp(-2*R/k_max) for BUD and PC1-X and the exact hypergeometric "
             "probability for SRS in projections, with Rotterdam R=277. "
             "Evaluate the exact product of cell hypergeometric probabilities "
             "using actual GBSG horizon membership after the freeze.",
        evaluation="Archived small_budget_application.run with frozen designs "
                   "at the BUD-only budget, common priorities, method-specific "
                   "bootstrap streams, and the selected KM backend. Compare "
                   "all first-draw bootstrap values against NumPy for each method. "
                   "Count intervals with half-width strictly above 90 days and "
                   "report paired percentile Monte Carlo intervals.",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / PLAN_PROTOCOL, protocol)
    raw, frame, x = archived.target_covariates(ROOT / "data/gbsg.csv")
    raw.to_csv(OUT / SNAPSHOT, index=False, mode="x")
    forecast, calibration = archived.calibrate(frame)
    if calibration["projected_risk_count"] != WORKING_RISK:
        raise RuntimeError("Unexpected Rotterdam working risk count")
    rows = scan_bud(x, forecast, calibration)
    pd.DataFrame(rows).to_csv(OUT / SCAN, index=False, mode="x")
    first = next(i for i, row in enumerate(rows) if row["bud_pass"])
    sizes = [rows[first]["m"]]
    if first:
        sizes.insert(0, rows[first - 1]["m"])
    projections, cells = [], []
    for m in sizes:
        row, cell_rows, designs, geometry = archived.candidate(x, forecast, calibration, m)
        # The archived common-pass flag played no role in choosing the size.
        del row["common_pass"]
        projections.append(row)
        cells.extend(cell_rows)
    selected = projections[-1]
    predecessor = projections[0] if first else None
    np.testing.assert_allclose(selected["bud_half_width"], rows[first]["bud_half_width"], rtol=1e-14)
    if not selected["bud_pass"] or (predecessor and predecessor["bud_pass"]):
        raise RuntimeError("Exact projections disagree with scan")
    pd.DataFrame(cells).to_csv(OUT / "gbsg_plan_cells.csv", index=False, mode="x")
    pd.DataFrame({"source_row": frame.source_row, **{name: pair[0] for name, pair in designs.items()}}
                 ).to_csv(OUT / PARTITIONS, index=False, mode="x")
    plan_files = [PLAN_PROTOCOL, SNAPSHOT, SCAN, "gbsg_plan_cells.csv", PARTITIONS]
    record = dict(frozen_utc=utc_now(), calibration=calibration, selected_m=selected["m"],
                  selected=selected, predecessor=predecessor, geometry=geometry,
                  allocation={name: pair[1].tolist() for name, pair in designs.items()},
                  scan_counts=pd.Series([row["status"] for row in rows]).value_counts().to_dict(),
                  total_admissible_budgets=len(rows), target_planning_columns=raw.columns.tolist(),
                  artifacts={name: driver.sha(OUT / name) for name in plan_files})
    write_json(OUT / PLAN_LOCK, record)
    print(json.dumps({key: record[key] for key in ("selected", "predecessor", "scan_counts")}, indent=2),
          flush=True)


def evaluate(backend_name="c"):
    lock_path = OUT / PLAN_LOCK
    lock, protocol = read_json(lock_path), read_json(OUT / PLAN_PROTOCOL)
    if protocol["sources"] != verified_sources() or protocol["versions"] != software_versions():
        raise RuntimeError("Source or numerical environment differs from the frozen protocol")
    for name, digest in lock["artifacts"].items():
        if driver.sha(OUT / name) != digest:
            raise RuntimeError(f"Changed planning artifact: {name}")
    m = lock["selected_m"]
    frozen = pd.read_csv(OUT / PARTITIONS)
    _, frame, x = archived.target_covariates(OUT / SNAPSHOT)
    replay, _ = archived.build(x, m)
    np.testing.assert_array_equal(frozen.source_row, frame.source_row)
    for name, (labels, allocation) in replay.items():
        np.testing.assert_array_equal(frozen[name], labels)
        np.testing.assert_array_equal(lock["allocation"][name], allocation)
    design = {name: (frozen[name].to_numpy(), np.asarray(lock["allocation"][name])) for name in replay}
    exact_risk = {}
    archived_load, archived_km = driver.load_cohort, driver.km_rmst
    target_path = OUT / "evaluation"
    target_path.mkdir(exist_ok=False)

    try:
        backend = KMBackend(backend_name)
        parity = []
        audit = dict(verified_utc=utc_now(), lock_sha256=driver.sha(lock_path),
                     planning_artifact_hashes_verified=True, archived_source_hashes_verified=True,
                     covariate_only_replay_matches_frozen_designs=True, backend=backend.metadata(),
                     planning_columns=lock["target_planning_columns"], selected_m=m)
        write_json(target_path / "verified_before_target_outcome_load.json", audit)

        def audited_load(name, spec):
            audit["target_load_started_utc"] = utc_now()
            actual, observed_x, provenance = archived_load(name, spec)
            pd.testing.assert_frame_equal(actual[frame.columns], frame, check_exact=True)
            pd.testing.assert_frame_equal(observed_x, x, check_exact=True)
            exact_risk.update(archived.exact_empty_risk(actual[spec["time"]].to_numpy(float), design))
            audit.update(target_load_utc=utc_now(), target_file_sha256=provenance["sha256"],
                         outcome_evaluation_covariates_match_planning=True)
            return actual, observed_x, provenance

        def frozen_build(observed_x, requested_m, seed):
            np.testing.assert_array_equal(observed_x, x.to_numpy())
            if requested_m != m or seed != driver.seed(driver.BASE_SEED, "gbsg", m, "design"):
                raise RuntimeError("Driver requested a different budget or seed")
            return design, lock["geometry"].copy()

        def checked_km(time, event, weights, tau):
            value = backend.km_rmst(time, event, weights, tau)
            if weights.ndim == 2 and len(parity) < len(METHODS):
                np.testing.assert_equal(len(weights), RESAMPLES)
                reference = archived_km(time, event, weights, tau)
                np.testing.assert_allclose(value, reference, atol=1e-9, rtol=0)
                parity.append(dict(method=tuple(design)[len(parity)], replicates=len(weights),
                                   max_absolute_difference=float(np.max(np.abs(value - reference)))))
            return value

        with patch.multiple(driver, BUDGETS=(m,), COHORTS={"gbsg": GBSG}, DRAWS=DRAWS, RESAMPLES=RESAMPLES,
                            MC_DRAWS=MC_DRAWS, load_cohort=audited_load, build_partitions=frozen_build,
                            km_rmst=checked_km):
            driver.run(target_path)
        raw = pd.read_csv(target_path / "raw.csv.gz", float_precision="round_trip")
        summary = pd.read_csv(target_path / "summary.csv", float_precision="round_trip").set_index("method")
        assert len(raw) == 3 * DRAWS and set(raw.method) == set(METHODS) and raw.m.eq(m).all()
        assert len(parity) == len(METHODS)
        for _, subset in raw.groupby("method"):
            np.testing.assert_array_equal(np.sort(subset.draw), np.arange(DRAWS))
        executed = pd.read_csv(target_path / f"gbsg_m{m}_partitions.csv")
        pd.testing.assert_frame_equal(executed, frozen, check_exact=True)
        grouped = raw.groupby("method")
        np.testing.assert_allclose(summary.rmse, grouped.error.agg(lambda e: sqrt(np.mean(e**2))), rtol=1e-13)
        np.testing.assert_allclose(summary.coverage, grouped.covered.mean(), atol=0, rtol=0)
        summary["mean_half_width"] = grouped.width.mean() / 2
        summary["over_90_days_count"] = raw.width.div(2).gt(HALF_WIDTH).groupby(raw.method).sum()
        summary["observed_empty_risk"] = summary.empty_tail_count / summary.draws
        summary["exact_empty_risk"] = pd.Series(exact_risk)
        summary.to_csv(target_path / "planning_evaluation_summary.csv")
        archived.paired_intervals(raw).to_csv(target_path / "paired_metric_intervals.csv", index=False)
        audit.update(completed_utc=utc_now(), rows=len(raw), first_draw_c_parity=parity,
                     executed_partitions_match_frozen=True, direct_full_and_first_draw_km_checks_passed=True,
                     no_failed_or_dropped_draws=True, exact_conditional_empty_risk=exact_risk,
                     output_sha256={path.name: driver.sha(path) for path in sorted(target_path.iterdir())
                                    if path.is_file()})
        write_json(target_path / "audit.json", audit)
        print(summary.to_string(), flush=True)
    except Exception:
        (target_path / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "evaluate"))
    parser.add_argument("--backend", choices=("c", "python"), default="c",
                        help="KM evaluator; first-draw batches are always checked against the archived reference.")
    args = parser.parse_args()
    if args.stage == "plan":
        plan()
    else:
        evaluate(args.backend)
