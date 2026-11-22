#!/usr/bin/env python3
"""Numerical checks of the paired finite-grid KM–RMST theorem (S4–S6)."""

import argparse
import csv
import hashlib
from itertools import combinations, product
import json
import os
from pathlib import Path
import sys
from time import perf_counter

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import scipy
from scipy.special import ndtr, ndtri

PROJECT = Path(__file__).resolve().parents[1]
LOCKED = PROJECT / "code" / "legacy_locked"
sys.path.insert(0, str(LOCKED))
from refinement import refined_design
from survival import km_if, km_rmst

SEED = 2026112206
GRID = np.array([1., 2., 3.])
TAU = 4.
WIDTH = np.diff(np.r_[GRID, TAU])
J = len(GRID)
TAIL = np.array([.10, .20])
K = 8
TEMPLATES = {
    "arm0": [(1, 1), (1, 1), (1, 0), (1.5, 0), (2, 1), (2.5, 0), (3, 1), (4, 0)],
    "arm1": [(1, 1), (1, 0), (1.5, 0), (2, 1), (2, 1), (2.5, 0), (3, 1), (4, 1)],
}
PROFILES = {"full": ([8, 16, 32, 64, 128], 1000, 1000, 8192),
            "smoke": ([8, 16], 32, 64, 256), "exact": ([], 0, 0, 0)}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def protocol(profile, chunk):
    sizes, draws, bootstrap, reference = PROFILES[profile]
    sources = [Path(__file__), *LOCKED.glob("*.py"), LOCKED / "FINAL_METHOD_LOCK.json"]
    return dict(
        profile=profile, software=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
        source_sha256={str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(sources)},
        sections="S4 derivative and exact integration, S5 paired identities, S6 fixed-grid limits",
        estimand="Exact complete-cohort KM RMST on the fixed grid, conditional on cohort and partition",
        seed=SEED, rng="PCG64 SeedSequence([seed,H,arm_id,stream_id]); streams 0 design, 1 reference, 2 signs",
        arm_ids={"arm0": 0, "arm1": 1, "sparse": 2}, H=sizes, capacity=K, sample_per_cell=2,
        design_draws=draws, signs_per_draw=bootstrap, independent_reference_draws=reference,
        chunk_draws=chunk, grid=GRID.tolist(), horizon=TAU, templates=TEMPLATES,
        covariate="One coordinate h=0,...,H-1 shared by the eight template rows. "
                  "Run locked BUD-fullspace-onepass, verify recovery of the template cells, "
                  "and order the verified cells by covariate.",
        repeated_cohorts="Each arm repeats its own fixed record template in every cell.",
        contrast="Arm 1 minus arm 0, independent design/reference/sign streams; m=4H.",
        stress="Arm 0 template, changing rows 6 and 7 to censoring at 2.5 except row 6 "
               "in the first cell and row 7 in the last cell. Last-grid risk is 2/N, "
               "horizon risk is 1/N. This sequence violates uniform positive risk.",
        basic_interval="95%, NumPy linear sample quantiles of centered bootstrap errors",
        covariance="Exact conditional covariance of HT vector from the observed pairs; "
                   "exact design covariance from cell pair enumeration. Contrast uses block diagonal "
                   "covariance for concatenated HT vectors and sums arm variances for the scalar target.",
        norm_moments="Euclidean vector norm, analytic second/fourth moments and Monte Carlo estimates; "
                     "contrast norm is on the concatenated HT vectors",
        largest_cell="Maximum possible absolute design score contribution over cells and pairs, "
                     "and maximum conditional bootstrap contribution from each observed sample, "
                     "both scaled by sqrt(m); contrast uses the larger arm contribution.",
        fixed_tail_thresholds=TAIL.tolist(),
        cdf="Exact two-sided empirical Kolmogorov distances with left/right limits; empirical errors "
            "rounded to 12 RMST decimals for numerical ties. Reference draws are independent of "
            "the study draws. Normal distances use exact linearization variance; CDF distances include MC noise.",
        taylor="Design remainder uses the exact complete-cohort gradient. Bootstrap remainders use "
               "both the complete-cohort gradient and the exact sample gradient on the plug-in domain; "
               "outside that domain the sample gradient is zero, so the remainder retains the entire increment.",
        plug_in="Use the analytic gradient at the sample vector when each risk and survival factor is "
                "at least half the minimum complete-cohort risk/factor for that arm. Report variance "
                "consistency, normal KS among available statistics, interval issuance, and coverage "
                "with all draws as denominator (unavailable does not cover).",
        failure_rule="Use zero hazard for empty grid risk. Positive bootstrap weights preserve the same "
                     "risk support as the phase-II draw. Keep every draw including empty horizons and "
                     "exhausted risk sets. Report fit and optional studentization failures separately.",
        stored="All design draws and their conditional diagnostics in draws.csv, independent reference "
               "errors in reference.npz. Individual sign replicates regenerate from the recorded RNG and chunk.",
        enumeration="Two cells with sizes 4 and 5, all unordered pairs and all Rademacher vectors, "
                    "2e-12 tolerance for algebraic identities and 1e-9 for finite differences. "
                    "Include exact-weight KM parity and full gradient checks.",
    )


def vectors(records):
    time, event = records[..., 0, None], records[..., 1, None]
    return np.concatenate(((time == GRID) & (event == 1), time >= GRID), axis=-1).astype(float)


def hazard(a):
    d, y = a[..., :J], a[..., J:]
    return np.clip(np.divide(d, y, out=np.zeros_like(d), where=y > 0), 0., 1.)


def rmst(a):
    return GRID[0] + np.cumprod(1 - hazard(a), axis=-1) @ WIDTH


def gradient(a, floor):
    alpha = hazard(a)
    y, factor = a[..., J:], 1. - alpha
    tail = np.cumsum((WIDTH * np.cumprod(factor, axis=-1))[..., ::-1], axis=-1)[..., ::-1]
    valid = ((y >= floor) & (factor >= floor)).all(axis=-1)
    slope = np.divide(-tail, y * factor, out=np.zeros_like(tail), where=valid[..., None])
    return np.concatenate((slope, -alpha * slope), axis=-1), valid


def norm2(x):
    return (x * x).sum(axis=-1)


def outer(x):
    return x[..., :, None] * x[..., None, :]


def covariance(x):
    return outer(x - x.mean(axis=0)).mean(axis=0)


def cell_terms(pop):
    k = len(pop)
    pairs = np.array(list(combinations(range(k), 2)))
    observed = pop[pairs]
    centered = observed.mean(axis=1) - pop.mean(axis=0)
    signed = np.sqrt(1 - 2 / k) * (observed[:, 0] - observed[:, 1]) / 2
    return pairs, centered, signed


def enumerate_pairs():
    records = [np.array(TEMPLATES["arm0"])[[0, 2, 4, 7]],
               np.array(TEMPLATES["arm1"])[[0, 2, 3, 6, 7]]]
    cells = [vectors(x) for x in records]
    capacity = np.array([len(x) for x in cells])
    total = capacity.sum()
    mass = capacity / total
    correction = 1 - 2 / capacity
    signs = np.array(list(product([-1, 1], repeat=len(cells))))
    members = [cell_terms(x)[0] for x in cells]
    full = np.concatenate(cells)
    full_records = np.concatenate(records)
    a = full.mean(axis=0)
    c = min(a[J:].min(), (1 - hazard(a)).min())
    g, valid = gradient(a, c / 2)
    assert valid
    identity = np.eye(len(a))
    step = 1e-5
    finite_difference = (
        -rmst(a + 2 * step * identity) + 8 * rmst(a + step * identity)
        - 8 * rmst(a - step * identity) + rmst(a - 2 * step * identity)
    ) / (12 * step)
    expected = sum(w**2 * f * np.cov(x, rowvar=False, ddof=1) / 2
                   for w, f, x in zip(mass, correction, cells))
    estimates, estimates_cov = [], []
    errors = {name: 0. for name in ("cell_mass", "weight_mean", "conditional_increment_mean",
                                    "conditional_covariance", "sign_increment", "weighted_km")}
    min_weight = np.inf
    for indices in product(*members):
        observed = np.array([x[pair] for x, pair in zip(cells, indices)])
        pair_records = np.concatenate([x[pair] for x, pair in zip(records, indices)])
        estimate = np.einsum("h,hj->j", mass, observed.mean(axis=1))
        adjust = signs * np.sqrt(correction)
        weights = capacity[:, None] / 2 * np.stack((1 + adjust, 1 - adjust), axis=-1)
        star = weights.reshape(len(signs), -1) @ observed.reshape(-1, len(a)) / total
        gap = observed[:, 0] - observed[:, 1]
        sign_factors = mass[:, None] * np.sqrt(correction[:, None]) * gap / 2
        theoretical = np.sum((mass**2 * correction / 4)[:, None, None] * outer(gap), axis=0)
        errors["cell_mass"] = max(errors["cell_mass"], np.max(np.abs(weights.sum(-1) - capacity)))
        errors["weight_mean"] = max(errors["weight_mean"],
                                    np.max(np.abs(weights.mean(0) - capacity[:, None] / 2)))
        errors["conditional_increment_mean"] = max(errors["conditional_increment_mean"],
                                                   np.max(np.abs((star - estimate).mean(0))))
        errors["conditional_covariance"] = max(errors["conditional_covariance"],
                                              np.max(np.abs(covariance(star) - theoretical)))
        errors["sign_increment"] = max(errors["sign_increment"],
                                       np.max(np.abs(star - estimate - signs @ sign_factors)))
        weighted_km = km_rmst(*pair_records.T, weights.reshape(len(signs), -1), TAU)
        errors["weighted_km"] = max(errors["weighted_km"], np.max(np.abs(rmst(star) - weighted_km)))
        min_weight = min(min_weight, weights.min())
        estimates.append(estimate)
        estimates_cov.append(theoretical)
    perturb = step * np.eye(len(full))
    weight_plus, weight_minus = 1 + perturb, 1 - perturb
    weight_derivative = (
        rmst((weight_plus @ full) / weight_plus.sum(axis=1, keepdims=True))
        - rmst((weight_minus @ full) / weight_minus.sum(axis=1, keepdims=True))
    ) / (2 * step)
    errors.update(
        design_ht_mean=np.max(np.abs(np.mean(estimates, axis=0) - a)),
        design_covariance=np.max(np.abs(covariance(np.array(estimates)) - expected)),
        expected_conditional_covariance=np.max(np.abs(np.mean(estimates_cov, axis=0) - expected)),
        cohort_km=abs(rmst(a) - km_rmst(*full_records.T, np.ones(total), TAU)),
        gradient_finite_difference=np.max(np.abs(finite_difference - g)),
        score_zero_mean=abs((full @ g).mean()),
        score_legacy_parity=np.max(np.abs(full @ g - km_if(*full_records.T, TAU))),
        score_weight_finite_difference=np.max(np.abs(weight_derivative - full @ g / total)),
    )
    tolerance = 2e-12
    assert min_weight > 0
    for name, value in errors.items():
        limit = 1e-9 if "finite_difference" in name else tolerance
        assert value < limit, (name, value, limit)
    return dict(design_outcomes=len(estimates), signs_per_outcome=len(signs), minimum_weight=min_weight,
                tolerance=tolerance, finite_difference_tolerance=1e-9, finite_difference_step=step,
                max_abs_errors={k: float(v) for k, v in errors.items()})


def cohort_records(h, arm):
    records = np.tile(TEMPLATES["arm0" if arm == "sparse" else arm], (h, 1, 1))
    if arm == "sparse":
        records[1:, 6] = (2.5, 0)
        records[:-1, 7] = (2.5, 0)
    return records


def verify_bud(h):
    cell_id = np.repeat(np.arange(h), K)
    labels, allocation, info = refined_design(cell_id[:, None], 2 * h, space="full")
    assert len(allocation) == h and np.all(allocation == 2)
    labels = labels.reshape(h, K)
    assert len(np.unique(labels[:, 0])) == h and np.all(labels == labels[:, :1])
    return dict(variant=info["variant"], scatter=info["refinement_after"],
                cell_labels_in_covariate_order=labels[:, 0].tolist())


def stream(h, arm, index):
    arm_id = ("arm0", "arm1", "sparse").index(arm)
    return np.random.default_rng(np.random.SeedSequence([SEED, h, arm_id, index]))


def exact_design(cells):
    """Use independent cell contributions and cumulant additivity for the fourth moment."""
    h = len(cells)
    terms = [cell_terms(x) for x in cells]
    pairs = terms[0][0]
    centered = np.array([x[1] for x in terms]) / h
    signed = np.array([x[2] for x in terms]) / h
    cell_cov = outer(centered).mean(axis=1)
    c = cell_cov.sum(axis=0)
    trace = np.trace(c)
    cell_trace = np.trace(cell_cov, axis1=-2, axis2=-1)
    correction = (norm2(centered)**2).mean(axis=1) - cell_trace**2 - 2 * norm2(cell_cov).sum(axis=1)
    fourth = trace**2 + 2 * np.sum(c**2) + correction.sum()
    cov_mse = np.sum((outer(signed) - cell_cov[:, None])**2, axis=(-1, -2)).mean(axis=1).sum()
    return pairs, signed, c, trace, fourth, (2 * h)**2 * cov_mse


def sample(cells, pairs, rng, count):
    h = len(cells)
    choice = rng.integers(len(pairs), size=(count, h))
    means = cells[:, pairs].mean(axis=2)
    return choice, means[np.arange(h), choice].mean(axis=1)


def arm_draws(h, arm, draws, b, ref, chunk):
    records = cohort_records(h, arm)
    cells = vectors(records)
    a = cells.mean(axis=(0, 1))
    theta = float(rmst(a))
    y, factors = a[J:], 1 - hazard(a)
    c = min(y.min(), factors.min())
    g, _ = gradient(a, c / 2)
    flat = records.reshape(-1, 2)
    assert abs(theta - km_rmst(*flat.T, np.ones(len(flat)), TAU)) < 2e-12
    assert np.max(np.abs(cells @ g - km_if(*flat.T, TAU).reshape(h, K))) < 2e-12
    pairs, signed, cov, design_second, design_fourth, cov_mse = exact_design(cells)
    choice, estimate = sample(cells, pairs, stream(h, arm, 0), draws)
    _, reference = sample(cells, pairs, stream(h, arm, 1), ref)
    horizon_records = records[..., 0] >= TAU
    horizon_observed = horizon_records[np.arange(h)[None, :, None], pairs[choice]]
    delta = estimate - a
    error = rmst(estimate) - theta
    plug_g, plug_domain = gradient(estimate, c / 2)
    sign_factors = signed[np.arange(h), choice]
    cov_hat = sign_factors.transpose(0, 2, 1) @ sign_factors
    trace = np.trace(cov_hat, axis1=-2, axis2=-1)
    exact_fourth = trace**2 + 2 * np.sum(cov_hat**2, axis=(-1, -2)) - 2 * np.sum(norm2(sign_factors)**2, axis=1)
    boot_rng = stream(h, arm, 2)
    result = dict(
        error=error, linear=delta @ g, remainder=error - delta @ g, norm2=norm2(delta),
        plugin_variance=np.sum((sign_factors @ plug_g[..., None])**2, axis=(1, 2)),
        oracle_conditional_variance=np.sum((sign_factors @ g)**2, axis=1),
        max_boot_cell_score=np.sqrt(2 * h) * np.abs(
            (sign_factors @ plug_g[..., None])[..., 0]).max(axis=1),
        scaled_cov_mse=(2 * h)**2 * np.sum((cov_hat - cov)**2, axis=(1, 2)),
        empty_horizon=~horizon_observed.any(axis=(1, 2)),
        empty_last_grid=estimate[:, -1] == 0,
        empty_grid=(estimate[:, J:] == 0).any(axis=1),
        zero_survival_factor=((estimate[:, :J] == estimate[:, J:]) &
                              (estimate[:, J:] > 0)).any(axis=1),
        plugin_domain_failure=~plug_domain,
        boot_exact_second=trace, boot_exact_fourth=exact_fourth,
        reference=rmst(reference) - theta,
    )
    for name in ("boot_error", "boot_norm2", "boot_remainder_full", "boot_remainder_sample"):
        result[name] = np.empty((draws, b))
    max_parity = 0.
    for first in range(0, draws, chunk):
        rows = slice(first, min(first + chunk, draws))
        signs = 2. * boot_rng.integers(0, 2, (len(estimate[rows]), b, h), dtype=np.int8) - 1.
        increment = signs @ sign_factors[rows]
        star_error = rmst(estimate[rows, None] + increment) - rmst(estimate[rows, None])
        result["boot_error"][rows] = star_error
        result["boot_norm2"][rows] = norm2(increment)
        result["boot_remainder_full"][rows] = star_error - increment @ g
        result["boot_remainder_sample"][rows] = star_error - (increment @ plug_g[rows, :, None])[..., 0]
        if first == 0:
            sampled_records = records[np.arange(h)[:, None], pairs[choice[0]]].reshape(-1, 2)
            signed_weights = np.sqrt(1 - 2 / K) * signs[0]
            weights = K / 2 * np.stack((1 + signed_weights, 1 - signed_weights), axis=-1)
            legacy = km_rmst(*sampled_records.T, weights.reshape(b, -1), TAU)
            max_parity = float(np.max(np.abs(star_error[0] + rmst(estimate[0]) - legacy)))
            assert max_parity < 2e-12
    result["meta"] = dict(theta=theta, N=h * K, min_y=float(y.min()), min_factor=float(factors.min()),
                          horizon_count=int(horizon_records.sum()), variance=float(g @ cov @ g),
                          complete_a=a.tolist(), complete_gradient=g.tolist(),
                          exact_design_second=float(design_second), exact_design_fourth=float(design_fourth),
                          exact_cov_mse_scaled=float(cov_mse), legacy_bootstrap_max_abs_diff=max_parity,
                          max_scaled_design_cell_score=float(
                              np.sqrt(2 * h) / h *
                              np.abs((cells[:, pairs].mean(axis=2) -
                                      cells.mean(axis=1)[:, None]) @ g).max()),
                          exact_empty_horizon_probability=float(np.prod([
                              np.mean(~support[pairs].any(axis=1)) for support in horizon_records])),
                          exact_empty_last_grid_probability=float(np.prod([
                              np.mean(~support[pairs].any(axis=1))
                              for support in cells[:, :, -1].astype(bool)])))
    return result


def contrast(arm0, arm1, h):
    result = {name: arm1[name] - arm0[name] for name in (
        "error", "linear", "remainder", "reference", "boot_error",
        "boot_remainder_full", "boot_remainder_sample")}
    for name in ("norm2", "boot_norm2", "boot_exact_second", "oracle_conditional_variance",
                 "plugin_variance"):
        result[name] = arm0[name] + arm1[name]
    for name in ("empty_horizon", "empty_last_grid", "empty_grid",
                 "zero_survival_factor", "plugin_domain_failure"):
        result[name] = arm0[name] | arm1[name]
    result["boot_exact_fourth"] = (
        arm0["boot_exact_fourth"] + arm1["boot_exact_fourth"]
        + 2 * arm0["boot_exact_second"] * arm1["boot_exact_second"])
    result["scaled_cov_mse"] = 4 * (arm0["scaled_cov_mse"] + arm1["scaled_cov_mse"])
    result["max_boot_cell_score"] = np.sqrt(2) * np.maximum(
        arm0["max_boot_cell_score"], arm1["max_boot_cell_score"])
    a, b = arm0["meta"], arm1["meta"]
    result["meta"] = dict(
        theta=b["theta"] - a["theta"], N=a["N"] + b["N"], min_y=min(a["min_y"], b["min_y"]),
        min_factor=min(a["min_factor"], b["min_factor"]),
        horizon_count=a["horizon_count"] + b["horizon_count"],
        variance=a["variance"] + b["variance"],
        exact_design_second=a["exact_design_second"] + b["exact_design_second"],
        exact_design_fourth=(a["exact_design_fourth"] + b["exact_design_fourth"]
                             + 2 * a["exact_design_second"] * b["exact_design_second"]),
        exact_cov_mse_scaled=4 * (a["exact_cov_mse_scaled"] + b["exact_cov_mse_scaled"]),
        max_scaled_design_cell_score=np.sqrt(2) * max(a["max_scaled_design_cell_score"],
                                                      b["max_scaled_design_cell_score"]),
        exact_empty_horizon_probability=1 - ((1 - a["exact_empty_horizon_probability"]) *
                                            (1 - b["exact_empty_horizon_probability"])),
        exact_empty_last_grid_probability=1 - ((1 - a["exact_empty_last_grid_probability"]) *
                                              (1 - b["exact_empty_last_grid_probability"])),
        legacy_bootstrap_max_abs_diff=max(a["legacy_bootstrap_max_abs_diff"],
                                         b["legacy_bootstrap_max_abs_diff"]))
    return result


def normal_distance(values):
    if not values.size:
        return None
    sorted_values = np.sort(values, axis=-1)
    n = sorted_values.shape[-1]
    cdf = ndtr(sorted_values)
    rank = np.arange(n)
    return np.maximum(np.max((rank + 1) / n - cdf, axis=-1),
                      np.max(cdf - rank / n, axis=-1))


def reference_distances(bootstrap, reference):
    reference = np.sort(np.round(reference, 12))
    distances = []
    for values in bootstrap:
        # On open intervals between bootstrap jumps, the reference CDF is monotone.
        unique, counts = np.unique(np.round(values, 12), return_counts=True)
        right = counts.cumsum() / len(values)
        left = right - counts / len(values)
        fn_left = np.searchsorted(reference, unique, side="left") / len(reference)
        fn_right = np.searchsorted(reference, unique, side="right") / len(reference)
        distances.append(max(np.max(np.abs(left - fn_left)), np.max(np.abs(right - fn_right))))
    return np.array(distances)


def summarize(name, h, data):
    m = (4 if name == "contrast" else 2) * h
    meta = data["meta"]
    v = meta["variance"]
    err, boot = data["error"], data["boot_error"]
    draws, b = boot.shape
    fit_failures = ~np.isfinite(err)
    bootstrap_failures = ~np.isfinite(boot)
    assert not fit_failures.any() and not bootstrap_failures.any(), (name, h)
    lower, upper = np.quantile(boot, [.025, .975], axis=1)
    cover = (lower <= err) & (err <= upper)
    issued = (data["plugin_variance"] > 0) & ~data["plugin_domain_failure"]
    studentized = err[issued] / np.sqrt(data["plugin_variance"][issued])
    student_cover = np.zeros(draws, dtype=bool)
    student_cover[issued] = np.abs(studentized) <= ndtri(.975)
    ks = reference_distances(boot, data["reference"])
    boot_normal = normal_distance(boot / np.sqrt(v))
    s = dict(target=name, H_per_arm=h, m=m, N=meta["N"], draws=draws, B=b,
             reference_draws=len(data["reference"]), theta=meta["theta"],
             min_full_risk=meta["min_y"], min_full_survival_factor=meta["min_factor"],
             full_horizon_count=meta["horizon_count"], mV=m * v,
             bias=err.mean(), bias_mcse=err.std(ddof=1) / np.sqrt(draws),
             sampling_variance=err.var(ddof=1), sampling_mse=np.mean(err**2),
             sampling_rmse=np.sqrt(np.mean(err**2)),
             m_sampling_variance=m * err.var(ddof=1), m_sampling_mse=m * np.mean(err**2),
             m_bias=m * err.mean(), linear_variance_ratio=data["linear"].var(ddof=1) / v,
             sampling_variance_ratio=err.var(ddof=1) / v,
             normal_ks=float(normal_distance(err / np.sqrt(v))),
             reference_normal_ks=float(normal_distance(data["reference"] / np.sqrt(v))),
             linear_normal_ks=float(normal_distance(data["linear"] / np.sqrt(v))),
             plugin_normal_ks=normal_distance(studentized),
             plugin_issued_n=int(issued.sum()), plugin_unavailable_n=int((~issued).sum()),
             plugin_coverage_all=student_cover.mean(),
             plugin_coverage_issued=student_cover[issued].mean() if issued.any() else None,
             basic_coverage=cover.mean(), basic_coverage_mcse=np.sqrt(cover.mean() * (1 - cover.mean()) / draws),
             basic_mean_width=np.mean(upper - lower), basic_width_mcse=(upper - lower).std(ddof=1) / np.sqrt(draws),
             empty_horizon_n=int(data["empty_horizon"].sum()),
             empty_last_grid_n=int(data["empty_last_grid"].sum()),
             empty_any_grid_n=int(data["empty_grid"].sum()),
             zero_survival_factor_n=int(data["zero_survival_factor"].sum()),
             plugin_domain_failure_n=int(data["plugin_domain_failure"].sum()),
             fit_failure_n=int(fit_failures.sum()), bootstrap_failure_n=int(bootstrap_failures.sum()),
             bootstrap_empty_horizon_n=b * int(data["empty_horizon"].sum()),
             bootstrap_empty_last_grid_n=b * int(data["empty_last_grid"].sum()),
             bootstrap_zero_survival_factor_n=b * int(data["zero_survival_factor"].sum()),
             exact_empty_horizon_probability=meta["exact_empty_horizon_probability"],
             exact_empty_last_grid_probability=meta["exact_empty_last_grid_probability"],
             max_scaled_design_cell_score=meta["max_scaled_design_cell_score"],
             exact_design_norm_second=meta["exact_design_second"],
             exact_design_norm_fourth=meta["exact_design_fourth"],
             exact_scaled_cov_mse=meta["exact_cov_mse_scaled"])
    per_draw = dict(
        error=err, linear=data["linear"], design_remainder=data["remainder"],
        design_norm_second=data["norm2"], design_norm_fourth=data["norm2"]**2,
        scaled_cov_mse=data["scaled_cov_mse"],
        scaled_plugin_variance=m * data["plugin_variance"],
        scaled_oracle_conditional_variance=m * data["oracle_conditional_variance"],
        scaled_boot_variance=m * boot.var(axis=1, ddof=1),
        max_boot_cell_score=data["max_boot_cell_score"],
        conditional_ks=ks, conditional_normal_ks=boot_normal,
        bootstrap_norm_second=data["boot_norm2"].mean(axis=1),
        bootstrap_norm_fourth=(data["boot_norm2"]**2).mean(axis=1),
        exact_bootstrap_norm_second=data["boot_exact_second"], exact_bootstrap_norm_fourth=data["boot_exact_fourth"],
        design_remainder_sq=data["remainder"]**2,
        boot_remainder_full_mse=(data["boot_remainder_full"]**2).mean(axis=1),
        boot_remainder_sample_mse=(data["boot_remainder_sample"]**2).mean(axis=1),
        bootstrap_q025=lower, bootstrap_q975=upper, basic_cover=cover,
        empty_horizon=data["empty_horizon"], empty_last_grid=data["empty_last_grid"],
        empty_grid=data["empty_grid"], zero_survival_factor=data["zero_survival_factor"],
        plugin_domain_failure=data["plugin_domain_failure"], plugin_issued=issued,
        plugin_cover_all=student_cover, fit_failure=fit_failures,
        bootstrap_failure_count=bootstrap_failures.sum(axis=1))
    for threshold in TAIL:
        tag = f"{round(100 * threshold):02d}"
        per_draw[f"design_tail_{tag}"] = data["norm2"] > threshold**2
        per_draw[f"bootstrap_tail_{tag}"] = (data["boot_norm2"] > threshold**2).mean(axis=1)
        s[f"design_tail_{tag}_n"] = int(per_draw[f"design_tail_{tag}"].sum())
    means = ("design_norm_second", "design_norm_fourth", "scaled_cov_mse",
             "bootstrap_norm_second", "bootstrap_norm_fourth", "exact_bootstrap_norm_second",
             "exact_bootstrap_norm_fourth", "design_remainder_sq", "boot_remainder_full_mse",
             "boot_remainder_sample_mse", "scaled_plugin_variance", "scaled_oracle_conditional_variance",
             "scaled_boot_variance", "conditional_ks", "conditional_normal_ks",
             "design_tail_10", "design_tail_20", "bootstrap_tail_10", "bootstrap_tail_20",
             "max_boot_cell_score")
    for key in means:
        s[key + "_mean"] = per_draw[key].mean()
    for key in ("conditional_ks", "conditional_normal_ks", "scaled_boot_variance",
                "scaled_plugin_variance", "boot_remainder_sample_mse", "bootstrap_tail_10",
                "bootstrap_tail_20"):
        s[key + "_p10"], s[key + "_p90"] = np.quantile(per_draw[key], [.10, .90])
    s["scaled_plugin_variance_mse_to_mV"] = np.mean((per_draw["scaled_plugin_variance"] - m * v)**2)
    s["scaled_boot_variance_mse_to_mV"] = np.mean((per_draw["scaled_boot_variance"] - m * v)**2)
    s["m2_design_remainder_mse"] = m**2 * s["design_remainder_sq_mean"]
    s["m2_boot_full_remainder_mse"] = m**2 * s["boot_remainder_full_mse_mean"]
    s["m2_boot_sample_remainder_mse"] = m**2 * s["boot_remainder_sample_mse_mean"]
    s["m2_boot_sample_remainder_mse_p90"] = m**2 * s["boot_remainder_sample_mse_p90"]
    rows = [dict(target=name, H_per_arm=h, draw=i + 1, **dict(zip(per_draw, row)))
            for i, row in enumerate(zip(*per_draw.values()))]
    return s, rows


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rate_slopes(rows):
    metrics = ("exact_design_norm_second", "exact_design_norm_fourth", "exact_scaled_cov_mse",
               "design_norm_second_mean", "design_norm_fourth_mean", "scaled_cov_mse_mean",
               "bootstrap_norm_second_mean", "bootstrap_norm_fourth_mean",
               "design_remainder_sq_mean", "boot_remainder_full_mse_mean", "boot_remainder_sample_mse_mean")
    slopes = []
    for target in ("arm0", "arm1", "contrast", "sparse"):
        group = [r for r in rows if r["target"] == target]
        x = np.log([r["H_per_arm"] for r in group])
        for metric in metrics:
            values = np.array([r[metric] for r in group])
            if np.all(values > 0):
                slopes.append(dict(target=target, metric=metric, slope=np.polyfit(x, np.log(values), 1)[0]))
    return slopes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="full")
    parser.add_argument("--out", type=Path, required=True, help="Fresh directory, relative to project root or absolute")
    parser.add_argument("--chunk", type=int, default=16, help="Number of phase-II draws per sign batch")
    args = parser.parse_args()
    if args.chunk < 1:
        parser.error("--chunk must be positive")
    out = PROJECT / args.out
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", protocol(args.profile, args.chunk))
    start = perf_counter()
    write_json(out / "exact_pairs.json", enumerate_pairs())
    sizes, draws, b, ref = PROFILES[args.profile]
    if not sizes:
        print(f"Exact paired enumeration complete: {out}", flush=True)
        return
    rows, summaries, metadata, references = [], [], {}, {}
    for h in sizes:
        metadata[str(h)] = {"design": verify_bud(h)}
        arm0 = arm_draws(h, "arm0", draws, b, ref, args.chunk)
        arm1 = arm_draws(h, "arm1", draws, b, ref, args.chunk)
        targets = {"arm0": arm0, "arm1": arm1, "contrast": contrast(arm0, arm1, h),
                   "sparse": arm_draws(h, "sparse", draws, b, ref, args.chunk)}
        for name, data in targets.items():
            summary, details = summarize(name, h, data)
            summaries.append(summary)
            rows.extend(details)
            metadata[str(h)][name] = data["meta"]
            references[f"{name}_H{h}"] = data["reference"]
        del targets, arm0, arm1, data
        print(f"H={h} completed in {perf_counter() - start:.1f}s", flush=True)
    write_rows(out / "summary.csv", summaries)
    write_rows(out / "draws.csv", rows)
    write_rows(out / "rate_slopes.csv", rate_slopes(summaries))
    write_json(out / "cohorts.json", metadata)
    np.savez_compressed(out / "reference.npz", **references)
    write_json(out / "completed.json", dict(runtime_seconds=perf_counter() - start,
                                          summary_rows=len(summaries), draw_rows=len(rows),
                                          profile=args.profile,
                                          sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                  for p in sorted(out.iterdir())}))


if __name__ == "__main__":
    main()
