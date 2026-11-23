"""Plan and evaluate the Rotterdam-calibrated GBSG sampling budgets."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from math import ceil, comb, exp, floor, prod, sqrt
from pathlib import Path
from statistics import NormalDist
import traceback
from unittest.mock import patch

import gbsg_eval as driver
from backend import KMBackend

import numpy as np
import pandas as pd


ROOT = driver.ROOT
DEFAULT_OUT = ROOT / "results/budget_optimization"
DEFAULT_BUD_OUT = ROOT / "results/bud_budget/gbsg"
HORIZON = 1825
TARGET_SIZE = 686
GBSG = driver.COHORTS["gbsg"]
STRATIFIED = ("BUD", "PC1-X")
METHODS = ("SRS", *STRATIFIED)
FEATURES = ("age", "meno", "nodes", "pgr", "er")
LOG_FEATURES = ("nodes", "pgr", "er")
FOLDS, FOLD_SEED = 5, 0x78c404e3
DRAWS, RESAMPLES, MC_DRAWS = 1000, 1000, 4000
HALF_WIDTH, EMPTY_LIMIT = 90, 0.01
Z = NormalDist().inv_cdf(0.975)
SOURCE_FILES = (
    "code/gbsg_plan.py", "code/gbsg_eval.py", "code/real_application.py",
    "code/real_core/designs.py", "code/real_core/survival.py",
    "code/backend.py", "code/native/weighted_km.c",
)


def rmst_scores(time, event, horizon):
    """Return the complete-cohort KM integral and centered raw-weight scores."""
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
    return float(jumps[0] + tail[0]) if len(jumps) else float(horizon), scores


def planning_row(m, second_moment, risk_count, z):
    """Calculate the fixed-capacity planning bounds for one even budget."""
    largest_cell = ceil(TARGET_SIZE / (m // 2))
    variance_bound = largest_cell * (largest_cell - 2) / (2 * TARGET_SIZE * (largest_cell - 1)) * second_moment
    srs_variance = (1 / m - 1 / TARGET_SIZE) * (TARGET_SIZE / (TARGET_SIZE - 1)) * second_moment
    return {"m": m, "paired_half_width": z * sqrt(variance_bound), "srs_half_width": z * sqrt(srs_variance),
            "paired_empty_tail_bound": exp(-2 * risk_count / largest_cell),
            "srs_empty_tail_probability": comb(TARGET_SIZE - risk_count, m) / comb(TARGET_SIZE, m)}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, record):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_hashes():
    return {name: driver.sha(ROOT / name) for name in SOURCE_FILES}


def software_versions():
    return {name: importlib.metadata.version(name) for name in
            ("numpy", "pandas", "scipy", "scikit-learn")}


def target_covariates(path):
    """Mirror the driver's covariate construction with an explicit CSV projection."""
    columns = [GBSG["id"], *GBSG["numeric"], *GBSG["log"],
               *GBSG["indicators"], *GBSG["categorical"]]
    if {GBSG["time"], GBSG["event"]}.intersection(columns):
        raise ValueError("Target outcome column in the planning projection")
    raw = pd.read_csv(path, usecols=columns)[columns]
    ids = raw[GBSG["id"]]
    if len(raw) != TARGET_SIZE or ids.isna().any() or not ids.is_unique:
        raise ValueError("Unexpected GBSG frame or identifiers")
    raw["source_row"] = np.arange(1, len(raw) + 1)
    order = np.random.default_rng(driver.seed(driver.BASE_SEED, "gbsg", "row_order")).permutation(len(raw))
    frame = raw.iloc[order].reset_index(drop=True)
    x = pd.DataFrame(index=frame.index)
    for name in GBSG["numeric"] + GBSG["log"]:
        value = frame[name].astype(float)
        if value.isna().any():
            x[f"{name}_missing"] = value.isna().astype(float)
        value = value.fillna(value.median())
        x[name] = np.log1p(value) if name in GBSG["log"] else value
    for name, level in GBSG["indicators"].items():
        x[name] = frame[name].eq(level).astype(float)
        if frame[name].isna().any():
            x[f"{name}_missing"] = frame[name].isna().astype(float)
    for name in GBSG["categorical"]:
        x = x.join(pd.get_dummies(frame[name].astype("string").fillna("missing"), prefix=name, dtype=float))
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Nonfinite GBSG design covariates")
    return raw, frame, x


def mean_features(frame):
    features = frame[list(FEATURES)].to_numpy(float)
    if (not np.isfinite(features).all() or not np.isin(features[:, 1], [0, 1]).all()
            or np.any(features[:, 2:] < 0)):
        raise ValueError("Invalid harmonized score predictors")
    features[:, 2:] = np.log1p(features[:, 2:])
    return features


def fit_linear(train_x, scores):
    center, scale = train_x.mean(axis=0), train_x.std(axis=0)
    # Keep age in years and menopause as a 0/1 indicator.
    center[:2], scale[:2] = 0, 1
    if np.any(scale <= 0):
        raise ValueError("A pilot log predictor has zero variation")
    x = np.column_stack([np.ones(len(train_x)), (train_x - center) / scale])
    coefficients, _, rank, _ = np.linalg.lstsq(x, scores, rcond=None)
    if rank != x.shape[1]:
        raise ValueError("Rank-deficient Rotterdam score regression")
    return center, scale, coefficients


def predict(features, fit):
    center, scale, coefficients = fit
    return coefficients[0] + ((features - center) / scale) @ coefficients[1:]


def calibrate(target):
    pilot_path = ROOT / "data/rotterdam.csv"
    pilot = pd.read_csv(pilot_path)
    pilot = pilot[pilot.nodes > 0]
    if len(pilot) != 1546 or not pilot.pid.is_unique:
        raise ValueError("Unexpected Rotterdam node-positive cohort")
    recur = pilot.recur.eq(1)
    late_death = ~recur & pilot.death.eq(1) & pilot.rtime.lt(pilot.dtime)
    event = (recur | (pilot.death.eq(1) & ~late_death)).to_numpy()
    time = np.where(recur | late_death, pilot.rtime, pilot.dtime)
    value, scores = rmst_scores(time, event, HORIZON)
    x = mean_features(pilot)
    full_fit = fit_linear(x, scores)
    residual = np.empty(len(pilot))
    folds = []
    for held_out in np.array_split(np.random.default_rng(FOLD_SEED).permutation(len(pilot)), FOLDS):
        train = np.ones(len(pilot), dtype=bool)
        train[held_out] = False
        # Each block computes its own KM scores. The regression and its score
        # nuisance use no held-out outcomes, including through the KM risk sets.
        _, train_scores = rmst_scores(time[train], event[train], HORIZON)
        _, held_scores = rmst_scores(time[held_out], event[held_out], HORIZON)
        fit = fit_linear(x[train], train_scores)
        residual[held_out] = held_scores - predict(x[held_out], fit)
        folds.append(dict(n=len(held_out), residual_second_moment=float(np.mean(residual[held_out]**2))))
    risk = int(np.count_nonzero(time >= HORIZON))
    return predict(mean_features(target), full_fit), dict(
        pilot_source_sha256=driver.sha(pilot_path), n=len(pilot), events=int(event.sum()),
        late_deaths_censored=int(late_death.sum()), pilot_rmst=value, pilot_risk_at_horizon=risk,
        projected_risk_count=floor(TARGET_SIZE * risk / len(pilot)),
        marginal_score_second_moment=float(np.mean(scores**2)),
        residual_second_moment=float(np.mean(residual**2)), folds=folds,
        full_fit_center=full_fit[0].tolist(), full_fit_scale=full_fit[1].tolist(),
        full_fit_coefficients=full_fit[2].tolist(),
    )


def build(x, m):
    return driver.build_partitions(x.to_numpy(), m, driver.seed(driver.BASE_SEED, "gbsg", m, "design"))


def candidate(x, forecast, calibration, m):
    designs, geometry = build(x, m)
    baseline = planning_row(m, calibration["marginal_score_second_moment"],
                            calibration["projected_risk_count"], Z)
    row = dict(m=m, srs_half_width=baseline["srs_half_width"],
               srs_empty_risk=baseline["srs_empty_tail_probability"],
               paired_empty_bound=baseline["paired_empty_tail_bound"],
               old_paired_half_width_bound=baseline["paired_half_width"])
    row["srs_pass"] = row["srs_half_width"] <= HALF_WIDTH and row["srs_empty_risk"] <= EMPTY_LIMIT
    cells = []
    for method, prefix in [("BUD", "bud"), ("PC1-X", "pc1x")]:
        labels, allocation = designs[method]
        variance = 0.0
        for cell, n_h in enumerate(allocation):
            projected = forecast[labels == cell]
            k = len(projected)
            mean_variance = float(np.var(projected, ddof=1))
            score_variance = mean_variance + calibration["residual_second_moment"]
            term = (k / TARGET_SIZE)**2 * (1 / n_h - 1 / k) * score_variance
            variance += term
            cells.append(dict(m=m, method=method, cell=cell, population=k, allocation=int(n_h),
                              forecast_mean_variance=mean_variance, projected_score_variance=score_variance,
                              variance_contribution=term))
        row[f"{prefix}_half_width"] = Z * sqrt(variance)
        row[f"{prefix}_pass"] = row[f"{prefix}_half_width"] <= HALF_WIDTH and row["paired_empty_bound"] <= EMPTY_LIMIT
    row["common_pass"] = row["srs_pass"] and row["bud_pass"] and row["pc1x_pass"]
    return row, cells, designs, geometry


def plan_common(out):
    # Lock the executable protocol before reading even the pilot outcomes.
    protocol_path = out / "gbsg_plan_protocol.json"
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        created_utc=utc_now(), sources=source_hashes(), versions=software_versions(),
        horizon=HORIZON, target_size=TARGET_SIZE, half_width_days=HALF_WIDTH, max_empty_risk=EMPTY_LIMIT,
        pilot_endpoint="Node-positive Rotterdam; archived conservative recurrence/death construction",
        forecast="Fixed OLS with intercept; age in years, postmenopause indicator, log1p positive nodes, progesterone and estrogen receptors; logs standardized using the fitting block's population SD; omit grade and size",
        feature_order=list(FEATURES), pilot_folds=FOLDS, pilot_fold_seed=FOLD_SEED,
        residual="Row-weighted squared out-of-fit residuals; separate KM score calculation in each training and held-out block",
        projection="Unbiased finite-cell variance of forecast means plus common residual second moment under independent homoskedastic errors",
        planning_variance="Sum (k_h/N)^2 (1/2 - 1/k_h) times projected within-cell score variance; fixed external raw marginal second moment with finite population correction for SRS",
        tail="Archived upper bound exp(-2 R/k_max) for paired designs, exact hypergeometric probability for SRS, with R projected from Rotterdam",
        scan="Increasing even sizes with cell capacity at least three; start at SRS-only floor; BUD-only uses same floor; report lower adjacent size even if below floor",
        target_design=GBSG, seed=driver.BASE_SEED, phase2_draws=DRAWS, bootstrap_resamples=RESAMPLES,
        paired_mc_resamples=MC_DRAWS, evaluation_backend="c",
        evaluation="gbsg_eval.run with in-process budget, cohort, frozen partitions and C KM kernel; common priorities within draw; paired percentile Monte Carlo CIs",
        scope="Retrospective evaluation in a previously studied released cohort. Working projection for the planning calculation.",
    )
    write_json(protocol_path, protocol)
    raw, target, x = target_covariates(ROOT / "data/gbsg.csv")
    snapshot = out / "gbsg_plan_covariates.csv"
    raw.to_csv(snapshot, index=False, mode="x")
    forecast, calibration = calibrate(target)
    baseline = [planning_row(m, calibration["marginal_score_second_moment"],
                             calibration["projected_risk_count"], Z)
                for m in range(4, 2 * (TARGET_SIZE // 3) + 1, 2)]
    srs_floor = next(row["m"] for row in baseline if row["srs_half_width"] <= HALF_WIDTH
                     and row["srs_empty_tail_probability"] <= EMPTY_LIMIT)
    rows, cell_rows, bud_only, common = [], [], None, None
    preceding = max(4, srs_floor - 2)
    for m in range(preceding, baseline[-1]["m"] + 1, 2):
        row, cells, designs, geometry = candidate(x, forecast, calibration, m)
        row["eligible_at_srs_floor"] = m >= srs_floor
        rows.append(row)
        cell_rows.extend(cells)
        if m < srs_floor:
            continue
        if bud_only is None and row["bud_pass"]:
            bud_only = m
        if row["common_pass"]:
            common = m
            break
    scan = out / "gbsg_plan_candidates.csv"
    cell_table = out / "gbsg_plan_cells.csv"
    pd.DataFrame(rows).to_csv(scan, index=False, mode="x")
    pd.DataFrame(cell_rows).to_csv(cell_table, index=False, mode="x")
    artifacts = [protocol_path, snapshot, scan, cell_table]
    selected = None
    predecessor = None
    if common is not None:
        frozen = out / "gbsg_plan_frozen_partitions.csv"
        pd.DataFrame({"source_row": target.source_row, **{name: value[0] for name, value in designs.items()}}
                     ).to_csv(frozen, index=False, mode="x")
        artifacts.append(frozen)
        selected = rows[-1]
        predecessor = next((row for row in rows if row["m"] == common - 2), None)
    record = dict(frozen_utc=utc_now(), calibration=calibration, srs_only_floor=srs_floor,
                  bud_only_budget_at_or_above_srs_floor=bud_only, common_budget=common,
                  selected=selected, predecessor=predecessor, geometry=geometry if common else None,
                  target_planning_columns=raw.columns.tolist(),
                  artifacts={path.name: driver.sha(path) for path in artifacts})
    write_json(out / "gbsg_plan_lock.json", record)
    print(json.dumps({key: record[key] for key in ("srs_only_floor", "bud_only_budget_at_or_above_srs_floor",
                                                   "common_budget", "selected", "predecessor")}, indent=2), flush=True)


def exact_empty_risk(time, partitions):
    at_horizon = time >= HORIZON
    return {method: prod(comb(int(k - r), int(n)) / comb(int(k), int(n))
                         for k, r, n in zip(np.bincount(labels), np.bincount(labels[at_horizon],
                                                                           minlength=len(allocation)), allocation))
            for method, (labels, allocation) in partitions.items()}


def paired_intervals(raw):
    metrics = raw.assign(squared_error=raw.error**2, half_width=raw.width / 2)
    rng = np.random.default_rng(driver.seed(driver.BASE_SEED, "gbsg", int(raw.m.iloc[0]), "paired_mc"))
    indices = rng.integers(0, DRAWS, (MC_DRAWS, DRAWS))
    records = []
    for source, metric in [("squared_error", "rmse_days"), ("half_width", "mean_half_width_days"),
                           ("covered", "coverage_probability")]:
        wide = metrics.pivot(index="draw", columns="method", values=source)
        averages = {method: wide[method].to_numpy()[indices].mean(axis=1) for method in METHODS}
        points = {method: float(wide[method].mean()) for method in METHODS}
        if source == "squared_error":
            averages = {method: np.sqrt(value) for method, value in averages.items()}
            points = {method: sqrt(value) for method, value in points.items()}
        for other in ("SRS", "PC1-X"):
            low, high = np.quantile(averages["BUD"] - averages[other], [.025, .975])
            records.append(dict(metric=metric, comparator=other, bud_minus_comparator=points["BUD"] - points[other],
                                mc_low=low, mc_high=high))
    return pd.DataFrame(records)


def evaluate_common(out):
    lock_path = out / "gbsg_plan_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    protocol = json.loads((out / "gbsg_plan_protocol.json").read_text(encoding="utf-8"))
    if protocol["sources"] != source_hashes() or protocol["versions"] != software_versions():
        raise RuntimeError("Source or numerical environment differs from the frozen protocol")
    for name, digest in lock["artifacts"].items():
        if driver.sha(out / name) != digest:
            raise RuntimeError(f"Changed planning artifact: {name}")
    m = lock["common_budget"]
    if m is None:
        raise RuntimeError("No feasible common budget in the frozen plan")
    archived_load = driver.load_cohort
    archived_km = driver.km_rmst
    frozen = pd.read_csv(out / "gbsg_plan_frozen_partitions.csv")
    # Reconstruct the locked design from a physically outcome-free file before
    # opening the original GBSG CSV for outcome evaluation.
    snapshot = out / "gbsg_plan_covariates.csv"
    _, planning_frame, planning_x = target_covariates(snapshot)
    replay, _ = build(planning_x, m)
    np.testing.assert_array_equal(frozen.source_row, planning_frame.source_row)
    for name in METHODS:
        np.testing.assert_array_equal(frozen[name], replay[name][0])
    design = {name: (frozen[name].to_numpy(), alloc) for name, (_, alloc) in replay.items()}

    target_path = out / "gbsg_new"
    target_path.mkdir(exist_ok=False)
    try:
        backend = KMBackend("c")
    except Exception:
        (target_path / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


    audit = dict(verified_utc=utc_now(), lock_sha256=driver.sha(lock_path),
                 planning_artifact_hashes_verified=True, source_hashes_verified=True,
                 covariate_only_replay_matches_frozen_labels=True, backend=backend.metadata(),
                 planning_columns=lock["target_planning_columns"], selected_m=m,
                 reference_engine="Archived driver NumPy core", first_draw_c_parity=[])
    write_json(target_path / "verified_before_target_outcome_load.json", audit)
    observed_empty = None
    methods_in_driver_order = tuple(design)

    def audited_load(name, spec):
        nonlocal observed_empty
        frame, x, provenance = archived_load(name, spec)
        pd.testing.assert_frame_equal(frame[planning_frame.columns], planning_frame, check_exact=True)
        pd.testing.assert_frame_equal(x, planning_x, check_exact=True)
        observed_empty = exact_empty_risk(frame[spec["time"]].to_numpy(float), design)
        audit.update(target_load_utc=utc_now(), target_file_sha256=provenance["sha256"],
                     outcome_evaluation_covariates_match_planning=True)
        return frame, x, provenance

    def frozen_build(x, requested_m, seed):
        np.testing.assert_array_equal(x, planning_x.to_numpy())
        if requested_m != m or seed != driver.seed(driver.BASE_SEED, "gbsg", m, "design"):
            raise RuntimeError("Driver requested a different budget or seed")
        return design, lock["geometry"].copy()

    def checked_km(time, event, weights, tau):
        estimate = backend.km_rmst(time, event, weights, tau)
        parity = audit["first_draw_c_parity"]
        if weights.ndim == 2 and len(parity) < len(METHODS):
            reference = archived_km(time, event, weights, tau)
            np.testing.assert_allclose(estimate, reference, atol=1e-9, rtol=0)
            parity.append(dict(method=methods_in_driver_order[len(parity)], replicates=len(weights),
                               max_absolute_difference=float(np.max(np.abs(estimate - reference)))))
        return estimate

    try:
        with patch.multiple(driver, BUDGETS=(m,), COHORTS={"gbsg": GBSG}, DRAWS=DRAWS, RESAMPLES=RESAMPLES,
                            MC_DRAWS=MC_DRAWS, load_cohort=audited_load, build_partitions=frozen_build,
                            km_rmst=checked_km):
            driver.run(target_path)
        raw = pd.read_csv(target_path / "raw.csv.gz", float_precision="round_trip")
        summary = pd.read_csv(target_path / "summary.csv", float_precision="round_trip")
        assert len(raw) == 3 * DRAWS and raw.groupby("method").draw.nunique().eq(DRAWS).all()
        assert set(raw.method) == set(METHODS) and raw.m.eq(m).all()
        assert len(audit["first_draw_c_parity"]) == len(METHODS)
        executed = pd.read_csv(target_path / f"gbsg_m{m}_partitions.csv")
        pd.testing.assert_frame_equal(executed, frozen, check_exact=True)
        np.testing.assert_allclose(summary.set_index("method").rmse.sort_index(),
                                   raw.groupby("method").error.agg(lambda e: sqrt(np.mean(e**2))), rtol=1e-13)
        summary["mean_half_width"] = summary.mean_width / 2
        summary["observed_empty_risk"] = summary.empty_tail_count / summary.draws
        summary["exact_empty_risk"] = summary.method.map(observed_empty)
        summary.to_csv(target_path / "planning_evaluation_summary.csv", index=False)
        paired_intervals(raw).to_csv(target_path / "paired_metric_intervals.csv", index=False)
        audit.update(completed_utc=utc_now(), rows=len(raw), executed_partitions_match_frozen=True,
                     direct_full_and_first_draw_km_checks_passed=True,
                     no_failed_or_dropped_draws=True,
                     output_sha256={path.name: driver.sha(path) for path in sorted(target_path.iterdir())
                                    if path.is_file()})
        write_json(target_path / "audit.json", audit)
        print(summary.to_string(index=False), flush=True)
    except Exception:
        (target_path / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def _bud_sources():
    return source_hashes()


def _scan_bud(x, forecast, calibration):
    rows = []
    for m in range(4, 2 * (TARGET_SIZE // 3) + 1, 2):
        cells, remainder = divmod(TARGET_SIZE, m // 2)
        capacities = np.full(m // 2, cells, dtype=int)
        capacities[:remainder] += 1
        coefficients = (capacities / TARGET_SIZE) ** 2 * (1 / 2 - 1 / capacities)
        floor_variance = calibration["residual_second_moment"] * float(coefficients.sum())
        baseline = planning_row(m, calibration["marginal_score_second_moment"],
                                calibration["projected_risk_count"], Z)
        row = dict(m=m, min_capacity=cells, max_capacity=int(capacities.max()),
                   residual_variance_floor=floor_variance, half_width_floor=Z * sqrt(floor_variance),
                   paired_empty_bound=baseline["paired_empty_tail_bound"], bud_projected_variance=None,
                   bud_half_width=None, bud_pass=False)
        if row["paired_empty_bound"] > EMPTY_LIMIT:
            row["status"] = "horizon_screen"
        elif row["half_width_floor"] > HALF_WIDTH:
            row["status"] = "variance_screen"
        else:
            designs, _ = build(x, m)
            labels, allocation = designs["BUD"]
            np.testing.assert_array_equal(np.bincount(labels), capacities)
            np.testing.assert_array_equal(allocation, np.full(m // 2, 2))
            within = np.array([np.var(forecast[labels == cell], ddof=1) for cell in range(m // 2)])
            variance = float(np.sum(coefficients * (within + calibration["residual_second_moment"])))
            row.update(bud_projected_variance=variance, bud_half_width=Z * sqrt(variance),
                       bud_pass=Z * sqrt(variance) <= HALF_WIDTH, status="exact_projection")
        rows.append(row)
    return rows


def plan_bud(out=DEFAULT_BUD_OUT):
    """Freeze the BUD-only budget before loading any GBSG outcome columns."""
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(sources=_bud_sources(), versions=software_versions(), horizon=HORIZON,
                    target_size=TARGET_SIZE, half_width_days=HALF_WIDTH, max_empty_risk=EMPTY_LIMIT,
                    working_risk_count=277, target_design=GBSG, seed=driver.BASE_SEED,
                    phase2_draws=DRAWS, bootstrap_resamples=RESAMPLES, paired_mc_resamples=MC_DRAWS,
                    scope="Retrospective GBSG evaluation with a separately recorded external planning pilot.",
                    scan="Screen every even admissible budget; select the smallest BUD budget passing both thresholds.",
                    evaluation="GBSG evaluation with frozen designs, common priorities, and the selected KM backend.")
    write_json(out / "gbsg_plan_protocol.json", protocol)
    raw, frame, x = target_covariates(ROOT / "data/gbsg.csv")
    raw.to_csv(out / "gbsg_plan_covariates.csv", index=False, mode="x")
    forecast, calibration = calibrate(frame)
    if calibration["projected_risk_count"] != protocol["working_risk_count"]:
        raise RuntimeError("Unexpected Rotterdam working risk count")
    rows = _scan_bud(x, forecast, calibration)
    pd.DataFrame(rows).to_csv(out / "gbsg_plan_candidates.csv", index=False, mode="x")
    selected_index = next(index for index, row in enumerate(rows) if row["bud_pass"])
    budgets = [rows[selected_index]["m"]]
    if selected_index:
        budgets.insert(0, rows[selected_index - 1]["m"])
    projections, cell_rows = [], []
    for m in budgets:
        row, cells, designs, geometry = candidate(x, forecast, calibration, m)
        del row["common_pass"]
        projections.append(row)
        cell_rows.extend(cells)
    selected, predecessor = projections[-1], projections[0] if selected_index else None
    np.testing.assert_allclose(selected["bud_half_width"], rows[selected_index]["bud_half_width"], rtol=1e-14)
    if not selected["bud_pass"] or predecessor and predecessor["bud_pass"]:
        raise RuntimeError("Exact projections disagree with the BUD scan")
    pd.DataFrame(cell_rows).to_csv(out / "gbsg_plan_cells.csv", index=False, mode="x")
    pd.DataFrame({"source_row": frame.source_row, **{name: pair[0] for name, pair in designs.items()}}).to_csv(
        out / "gbsg_plan_frozen_partitions.csv", index=False, mode="x")
    names = ("gbsg_plan_protocol.json", "gbsg_plan_covariates.csv", "gbsg_plan_candidates.csv",
             "gbsg_plan_cells.csv", "gbsg_plan_frozen_partitions.csv")
    record = dict(calibration=calibration, selected_m=selected["m"], selected=selected, predecessor=predecessor,
                  geometry=geometry, allocation={name: pair[1].tolist() for name, pair in designs.items()},
                  scan_counts=pd.Series([row["status"] for row in rows]).value_counts().to_dict(),
                  total_admissible_budgets=len(rows), target_planning_columns=raw.columns.tolist(),
                  artifacts={name: driver.sha(out / name) for name in names})
    write_json(out / "gbsg_plan_lock.json", record)
    print(json.dumps({key: record[key] for key in ("selected", "predecessor", "scan_counts")}, indent=2), flush=True)


def evaluate_bud(backend_name="c", out=DEFAULT_BUD_OUT):
    lock_path = out / "gbsg_plan_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    protocol = json.loads((out / "gbsg_plan_protocol.json").read_text(encoding="utf-8"))
    if protocol["sources"] != _bud_sources() or protocol["versions"] != software_versions():
        raise RuntimeError("Source or numerical environment differs from the frozen protocol")
    for name, digest in lock["artifacts"].items():
        if driver.sha(out / name) != digest:
            raise RuntimeError(f"Changed planning artifact: {name}")
    m = lock["selected_m"]
    frozen = pd.read_csv(out / "gbsg_plan_frozen_partitions.csv")
    _, frame, x = target_covariates(out / "gbsg_plan_covariates.csv")
    replay, _ = build(x, m)
    np.testing.assert_array_equal(frozen.source_row, frame.source_row)
    for name, (labels, allocation) in replay.items():
        np.testing.assert_array_equal(frozen[name], labels)
        np.testing.assert_array_equal(lock["allocation"][name], allocation)
    design = {name: (frozen[name].to_numpy(), np.asarray(lock["allocation"][name])) for name in replay}
    target = out / "evaluation"
    target.mkdir(exist_ok=False)
    backend, parity, exact_risk = KMBackend(backend_name), [], {}
    audit = dict(lock_sha256=driver.sha(lock_path), planning_artifact_hashes_verified=True,
                 archived_source_hashes_verified=True, covariate_only_replay_matches_frozen_designs=True,
                 backend=backend.metadata(), planning_columns=lock["target_planning_columns"], selected_m=m)
    write_json(target / "verified_before_target_outcome_load.json", audit)
    archived_load, archived_km = driver.load_cohort, driver.km_rmst

    def audited_load(name, spec):
        actual, observed_x, provenance = archived_load(name, spec)
        pd.testing.assert_frame_equal(actual[frame.columns], frame, check_exact=True)
        pd.testing.assert_frame_equal(observed_x, x, check_exact=True)
        exact_risk.update(exact_empty_risk(actual[spec["time"]].to_numpy(float), design))
        audit.update(target_file_sha256=provenance["sha256"], outcome_evaluation_covariates_match_planning=True)
        return actual, observed_x, provenance

    def frozen_build(observed_x, requested_m, design_seed):
        np.testing.assert_array_equal(observed_x, x.to_numpy())
        if requested_m != m or design_seed != driver.seed(driver.BASE_SEED, "gbsg", m, "design"):
            raise RuntimeError("Driver requested a different budget or seed")
        return design, lock["geometry"].copy()

    def checked_km(time, event, weights, tau):
        value = backend.km_rmst(time, event, weights, tau)
        if weights.ndim == 2 and len(parity) < len(METHODS):
            reference = archived_km(time, event, weights, tau)
            np.testing.assert_allclose(value, reference, atol=1e-9, rtol=0)
            parity.append(dict(method=tuple(design)[len(parity)], replicates=len(weights),
                               max_absolute_difference=float(np.max(np.abs(value - reference)))))
        return value

    try:
        with patch.multiple(driver, BUDGETS=(m,), COHORTS={"gbsg": GBSG}, DRAWS=DRAWS, RESAMPLES=RESAMPLES,
                            MC_DRAWS=MC_DRAWS, load_cohort=audited_load, build_partitions=frozen_build,
                            km_rmst=checked_km):
            driver.run(target)
        raw = pd.read_csv(target / "raw.csv.gz", float_precision="round_trip")
        summary = pd.read_csv(target / "summary.csv", float_precision="round_trip").set_index("method")
        assert len(raw) == 3 * DRAWS and set(raw.method) == set(METHODS) and raw.m.eq(m).all()
        assert len(parity) == len(METHODS)
        executed = pd.read_csv(target / f"gbsg_m{m}_partitions.csv")
        pd.testing.assert_frame_equal(executed, frozen, check_exact=True)
        grouped = raw.groupby("method")
        np.testing.assert_allclose(summary.rmse, grouped.error.agg(lambda error: sqrt(np.mean(error**2))), rtol=1e-13)
        np.testing.assert_allclose(summary.coverage, grouped.covered.mean(), atol=0, rtol=0)
        summary["mean_half_width"] = grouped.width.mean() / 2
        summary["over_90_days_count"] = raw.width.div(2).gt(HALF_WIDTH).groupby(raw.method).sum()
        summary["observed_empty_risk"] = summary.empty_tail_count / summary.draws
        summary["exact_empty_risk"] = pd.Series(exact_risk)
        summary.to_csv(target / "planning_evaluation_summary.csv")
        paired_intervals(raw).to_csv(target / "paired_metric_intervals.csv", index=False)
        audit.update(rows=len(raw), first_draw_c_parity=parity, executed_partitions_match_frozen=True,
                     direct_full_and_first_draw_km_checks_passed=True, no_failed_or_dropped_draws=True,
                     exact_conditional_empty_risk=exact_risk,
                     output_sha256={path.name: driver.sha(path) for path in sorted(target.iterdir()) if path.is_file()})
        write_json(target / "audit.json", audit)
        print(summary.to_string(), flush=True)
    except Exception:
        (target / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def external_plan(output):
    """Write the archived broad Rotterdam planning illustration."""
    pilot = pd.read_csv(ROOT / "data/rotterdam.csv")
    pilot = pilot[pilot.nodes > 0]
    recur = pilot.recur.eq(1)
    late_death = ~recur & pilot.death.eq(1) & pilot.rtime.lt(pilot.dtime)
    event = recur | (pilot.death.eq(1) & ~late_death)
    time = np.where(recur | late_death, pilot.rtime, pilot.dtime)
    value, scores = rmst_scores(time, event.to_numpy(), HORIZON)
    risk = floor(TARGET_SIZE * np.count_nonzero(time >= HORIZON) / len(time))
    candidates = [planning_row(m, float(np.mean(scores**2)), risk, Z)
                  for m in range(2, 2 * (TARGET_SIZE // 3) + 1, 2)]
    record = dict(external_cohort_size=len(time), external_events=int(event.sum()),
                  external_late_deaths_censored=int(late_death.sum()), external_risk_count_at_horizon=int(np.count_nonzero(time >= HORIZON)),
                  external_rmst_days=value, external_score_second_moment=float(np.mean(scores**2)), horizon_days=HORIZON,
                  target_size=TARGET_SIZE, working_target_risk_count=risk,
                  evaluated_budgets=[row for row in candidates if row["m"] in (40, 80, 120, 240)],
                  recommendations=[dict(
                      half_width_days=width,
                      smallest_even_budget=next((row["m"] for row in candidates if max(row["paired_half_width"], row["srs_half_width"]) <= width and max(row["paired_empty_tail_bound"], row["srs_empty_tail_probability"]) <= .01), None),
                      first_evaluated_budget=next((row["m"] for row in candidates if row["m"] in (40, 80, 120, 240) and max(row["paired_half_width"], row["srs_half_width"]) <= width and max(row["paired_empty_tail_bound"], row["srs_empty_tail_probability"]) <= .01), None),
                  ) for width in (60, 90, 120)])
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("bud-plan", "bud-evaluate", "common-plan", "common-evaluate", "external"))
    parser.add_argument("--backend", choices=("c", "python"), default="c")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.stage == "bud-plan":
        plan_bud((args.out or DEFAULT_BUD_OUT).resolve())
    elif args.stage == "bud-evaluate":
        evaluate_bud(args.backend, (args.out or DEFAULT_BUD_OUT).resolve())
    elif args.stage == "common-plan":
        plan_common((args.out or DEFAULT_OUT).resolve())
    elif args.stage == "common-evaluate":
        evaluate_common((args.out or DEFAULT_OUT).resolve())
    else:
        if args.out is None:
            parser.error("external requires --out")
        external_plan(args.out.resolve())
