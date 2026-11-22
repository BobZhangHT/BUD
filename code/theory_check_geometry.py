#!/usr/bin/env python3
"""Reproducible BUD geometry, fixed-score variance, and transport checks."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
from itertools import product
import json
from math import ceil, gcd, lcm
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback

sys.dont_write_bytecode = True
from real_application import capacitated_assignment
from designs import glp_uniform_design, inverse_quantile_anchors, standardized_pca
import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from scipy.stats import qmc


SEED = 202611220731
REPS = 1000
ABS_TOL = 1e-9
REL_TOL = 1e-9
LP_TOL = 1e-10
SCENARIOS = (
    ("univariate_equal", 24, 6, 1),
    ("multivariate_equal", 36, 6, 4),
    ("multivariate_near_equal", 38, 6, 4),
)
POINT_TYPES = ("frozen_glp", "clustered", "iid_uniform")
TRANSPORT = {1: (4, 8, 16, 32, 64), 2: (4, 7, 11)}
GRID_SIDES = (12, 24)
SCORES = ("projection_plus_noise", "right_cluster", "zero_gain", "wrong_cluster")
GLP_CASES = ((1, 4), (4, 3), (8, 3), (5, 2), (7, 3), (5, 3), (2, 3))


class Checks:
    def __init__(self):
        self.metrics = {}
        self.failures = []
        self.case = None

    def le(self, name, left, right):
        left, right = float(left), float(right)
        slack = right - left
        metric = self.metrics.setdefault(name, {"count": 0, "min_slack": None})
        metric["count"] += 1
        if np.isfinite(slack) and (
            metric["min_slack"] is None or slack < metric["min_slack"]
        ):
            metric.update(min_slack=slack, min_slack_case=self.case)
        tolerance = ABS_TOL + REL_TOL * max(abs(left), abs(right))
        if not np.isfinite(slack) or slack < -tolerance:
            self.failures.append(dict(case=self.case, check=name, left=repr(left), right=repr(right)))

    def eq(self, name, left, right):
        error = abs(left - right)
        scale = max(abs(left), abs(right))
        self.le(name, error, REL_TOL * scale)
        metric = self.metrics[name]
        if error >= metric.get("max_abs_error", -1):
            metric.update(max_abs_error=float(error), max_error_case=self.case)


def rng(*key):
    return np.random.default_rng(np.random.SeedSequence([SEED, *key]))


@lru_cache(maxsize=32)
def margins(n, h):
    column = np.arange(n * h)
    rows = np.r_[np.repeat(np.arange(n), h), n + np.tile(np.arange(h), n)]
    return csr_matrix(
        (np.ones(2 * n * h), (rows, np.tile(column, 2))), shape=(n + h, n * h)
    )


def transport_lp(x, y, supply, demand, checks):
    """Unexpanded coupling LP with independent HiGHS primal/dual certification."""
    n, h = len(x), len(y)
    cost = cdist(x, y, "sqeuclidean")
    supply, demand = np.asarray(supply), np.asarray(demand)
    mass = float(supply.sum())
    result = linprog(
        cost.ravel(), A_eq=margins(n, h), b_eq=np.r_[supply, demand],
        method="highs-ds", options={
            "primal_feasibility_tolerance": LP_TOL, "dual_feasibility_tolerance": LP_TOL
        },
    )
    if not result.success:
        raise RuntimeError(result.message)
    flow = result.x.reshape(n, h)
    dual_row, dual_col = np.split(result.eqlin.marginals, [n])
    dual_violation = max(0., float(np.max(dual_row[:, None] + dual_col - cost)))
    dual_value = (dual_row @ supply + dual_col @ demand) / mass
    value = result.fun / mass
    checks.eq("lp_row_margin", np.max(np.abs(flow.sum(axis=1) - supply)), 0)
    checks.eq("lp_col_margin", np.max(np.abs(flow.sum(axis=0) - demand)), 0)
    checks.le("lp_nonnegativity", -flow.min(), 0)
    checks.le("lp_dual_feasibility", dual_violation, 0)
    checks.eq("lp_primal_dual_agreement", value, dual_value)
    # Make the grid certificate robust to floating-point LP residuals.
    # Removing excess flow and filling deficits costs at most max(cost) per
    # unit reallocated. Shifting the row potentials restores dual feasibility.
    margin_residual = (
        np.abs(flow.sum(axis=1) - supply).sum()
        + np.abs(flow.sum(axis=0) - demand).sum()
        + 2 * np.maximum(-flow, 0).sum()
    ) / mass
    rounding = ABS_TOL + REL_TOL * max(1., abs(value))
    lower = max(0., dual_value - dual_violation - rounding)
    upper = max(0., value + cost.max() * margin_residual + rounding)
    return value, lower, upper, flow


def centroids(x, labels, h):
    return np.array([x[labels == j].mean(axis=0) for j in range(h)])


def scatter(x, labels, centers):
    return float(np.mean(np.sum((x - centers[labels]) ** 2, axis=1)))


def frozen_design(x, capacities):
    h = len(capacities)
    x, scores, eigenvalues, q = standardized_pca(x)
    u = glp_uniform_design(h, q)
    z = scores[:, :q]
    anchors = inverse_quantile_anchors(z, u)
    initial, cost = capacitated_assignment(z, anchors, capacities)
    centers = centroids(x, initial, h)
    final, refine_cost = capacitated_assignment(x, centers, capacities)
    return x, z, eigenvalues[q:].sum(), u, anchors, initial, cost, centers, final, refine_cost


def adversarial_swap(x, labels, centers, capacities):
    """Choose the cross-cell unit swap that maximizes recentered scatter."""
    candidate_i, candidate_j = np.triu_indices(len(x), k=1)
    cross = labels[candidate_i] != labels[candidate_j]
    i, j = candidate_i[cross], candidate_j[cross]
    left, right = labels[i], labels[j]
    delta = x[j] - x[i]
    increase = 2 * np.einsum("ij,ij->i", centers[right] - centers[left], delta)
    increase -= np.sum(delta**2, axis=1) * (1 / capacities[left] + 1 / capacities[right])
    worst = int(np.argmax(increase))
    a, b = int(i[worst]), int(j[worst])
    swapped = labels.copy()
    swapped[[a, b]] = swapped[[b, a]]
    return swapped, (a, b), float(increase[worst]) / len(x)


def direct_srs(values, m):
    n = len(values)
    f = m / n
    covariance = np.full((n, n), m * (m - 1) / (n * (n - 1)) - f**2)
    np.fill_diagonal(covariance, f * (1 - f))
    centered = values - values.mean(axis=0)
    return np.einsum("ij,ik,kj->j", centered, covariance, centered) / m**2


def pair_moments(values, labels, capacities):
    means, variances = [], []
    for j, size in enumerate(capacities):
        cell = values[labels == j]
        a, b = np.triu_indices(size, k=1)
        pair_means = (cell[a] + cell[b]) / 2
        means.append(pair_means.mean(axis=0))
        variances.append(pair_means.var(axis=0))
    weights = capacities / len(values)
    return weights @ means, weights**2 @ variances


def variance_checks(g, noise, initial, final, capacities, d0, d1, priorities, checks):
    n, h = len(g), len(capacities)
    m = 2 * h
    right = np.array([g[final == j].mean() for j in range(h)])[final]
    wrong = g - right
    # The two components are population-orthogonal and the between-cell
    # component has zero BUD variance. Scale it to cancel the SRS gap.
    components = np.column_stack((right, wrong))
    _, direct = pair_moments(components, final, capacities)
    srs_right, srs_wrong = direct_srs(components, m)
    multiplier = np.sqrt((direct[1] - srs_wrong) / srs_right)
    values = np.column_stack((g + noise, right, multiplier * right + wrong, wrong))
    residual_rms = np.sqrt(np.mean((values - g[:, None]) ** 2, axis=0))
    cn = np.max(capacities * (capacities - 2) / (2 * n * (capacities - 1)))
    equal = bool(np.all(capacities == capacities[0]))
    if equal:
        checks.eq("equal_variance_bound_coefficient", cn, (1 - m / n) * n / (m * (n - h)))
    srs = direct_srs(values, m)
    overall = values.mean(axis=0)
    total = np.sum((values - overall) ** 2, axis=0)
    srs_sample = values[np.argsort(priorities)[:m]].mean(axis=0)
    checks.eq("srs_variance_identity", np.max(np.abs(srs - (1 - m / n) * total / (m * (n - 1)))), 0)
    records = []
    for name, labels, d in (("initial", initial, d0), ("final", final, d1)):
        means = centroids(values, labels, h)
        within = np.array([np.sum((values[labels == j] - means[j]) ** 2, axis=0) for j in range(h)])
        between = capacities @ ((means - overall) ** 2)
        formula = (capacities * (capacities - 2) / (capacities - 1)) @ within / (2 * n**2)
        pair_mean, pair_var = pair_moments(values, labels, capacities)
        bound = cn * (np.sqrt(d) + residual_rms) ** 2
        groups = [np.flatnonzero(labels == j) for j in range(h)]
        pick = [indices[np.argsort(priorities[indices])[:2]] for indices in groups]
        estimate = (capacities / n) @ values[np.asarray(pick)].mean(axis=1)
        gain = srs - pair_var
        w = (within / (capacities[:, None] - 1)).mean(axis=0)
        for j, score in enumerate(SCORES):
            key = f"{name}/{score}"
            checks.eq(f"anova_{'equal' if equal else 'unequal'}/{key}", total[j], within[:, j].sum() + between[j])
            checks.eq(f"pair_mean/{key}", pair_mean[j], overall[j])
            checks.eq(f"pair_variance/{key}", pair_var[j], formula[j])
            checks.le(f"lipschitz_variance_bound/{key}", pair_var[j], bound[j])
            if equal:
                checks.eq(f"equal_anova/{key}", total[j], (n - h) * w[j] + between[j])
                checks.eq(f"equal_srs_gain/{key}", gain[j],
                          (1 - m / n) / m * (between[j] - (h - 1) * w[j]) / (n - 1))
        records.append(dict(
            variance=pair_var, bound=bound, gain=gain, within_ss=within.sum(axis=0),
            between_ss=between, estimate_error=estimate - overall,
        ))
    final_gain = records[-1]["gain"]
    checks.le("right_cluster_positive_gain", ABS_TOL * 2, final_gain[1])
    checks.eq("zero_gain_comparator", final_gain[2], 0)
    checks.le("wrong_cluster_negative_gain", final_gain[3], -ABS_TOL * 2)
    for j, score in enumerate(SCORES):
        checks.le(f"variance_bound_descent/{score}", records[1]["bound"][j], records[0]["bound"][j])
    return srs, residual_rms, srs_sample - overall, records


def summarize_frames(rows):
    matrix = np.asarray(rows)
    return {
        "mean": matrix.mean(axis=0), "min": matrix.min(axis=0), "max": matrix.max(axis=0),
        "mean_mcse": matrix.std(axis=0, ddof=1) / np.sqrt(len(rows)),
    }


def frame_experiment(report, checks):
    for scenario, (name, n, h, p) in enumerate(SCENARIOS):
        capacities = np.full(h, n // h)
        capacities[:n % h] += 1
        group = dict(name=name, N=n, H=h, p=p, capacities=capacities, trials=[])
        report.append(group)
        for rep in range(REPS):
            checks.case = f"{name}/{rep}"
            gen = rng(0, scenario, rep)
            if p == 1:
                raw = gen.normal(size=(n, 1))
            else:
                a, b = gen.normal(size=(2, n))
                raw = np.column_stack((a, a, b, b)) + gen.normal(size=(n, p)) * [.20, .38, .28, .48]
            x, z, eq, u, anchors, initial, tn, centers, final, cost = frozen_design(raw, capacities)
            d0 = scatter(x, initial, centers)
            d1 = scatter(x, final, centroids(x, final, h))
            checks.le("scatter_final_to_initial", d1, d0)
            checks.le("scatter_initial_to_transport_plus_discarded", d0, tn + eq)
            checks.le("refinement_recentering", d1, cost)
            checks.le("refinement_old_center_descent", cost, d0)
            ones = np.ones(n)
            lp0, _, _, _ = transport_lp(z, anchors, ones, capacities, checks)
            lp1, _, _, _ = transport_lp(x, centers, ones, capacities, checks)
            checks.eq("initial_frozen_cost_vs_lp", tn, lp0)
            checks.eq("refinement_frozen_cost_vs_lp", cost, lp1)
            swap, units, predicted = adversarial_swap(x, initial, centers, capacities)
            approx_cost = scatter(x, swap, centers)
            approx_d = scatter(x, swap, centroids(x, swap, h))
            epsilon = max(0., approx_cost - d0)
            optimal_excess = approx_cost - cost
            checks.eq("approximate_capacities", np.max(np.abs(np.bincount(swap) - capacities)), 0)
            checks.eq("approximate_swap_scatter_formula", approx_d - d0, predicted)
            checks.le("approximate_s7_bound", approx_d, d0 + epsilon)
            checks.le("approximate_bound_using_optimum_excess", approx_d, d0 + optimal_excess)
            direction = np.arange(1., p + 1)
            g = x @ (direction / np.linalg.norm(direction))
            srs, residual, srs_error, variances = variance_checks(
                g, .35 * gen.normal(size=n), initial, final, capacities, d0, d1, gen.random(n), checks
            )
            group["trials"].append([
                rep, u.shape[1], tn, eq, d0, d1, cost, lp0, lp1, *units, approx_cost,
                approx_d, epsilon, optimal_excess, *srs, *residual, *srs_error,
                *(value for record in variances for values in record.values() for value in values),
            ])
        group["summary_by_column"] = summarize_frames(group["trials"])
        trials = np.asarray(group["trials"])
        group["retained_q_counts"] = dict(Counter(map(int, trials[:, 1])))
        group["scatter_strict_decreases"] = int(np.sum(trials[:, 4] - trials[:, 5] > ABS_TOL))
        group["adversarial_scatter_increases"] = int(np.sum(trials[:, 12] - trials[:, 4] > ABS_TOL))
        for name in ("initial", "final"):
            start = FRAME_COLUMNS.index(f"{name}_gain_{SCORES[0]}")
            gain = trials[:, start:start + len(SCORES)]
            group[f"{name}_gain_sign_counts"] = {
                score: [int(np.sum(column > ABS_TOL)), int(np.sum(abs(column) <= ABS_TOL)),
                        int(np.sum(column < -ABS_TOL))]
                for score, column in zip(SCORES, gain.T)
            }


FRAME_COLUMNS = [
    "replicate", "retained_q", "initial_transport", "discarded_variance",
    "initial_scatter", "final_scatter", "refinement_cost", "initial_lp", "refinement_lp",
    "swap_first_unit", "swap_second_unit", "approximate_cost", "approximate_scatter", "epsilon",
    "approximate_excess_over_optimum",
    *(f"{metric}_{score}" for metric in ("srs_variance", "residual_rms", "srs_estimate_error")
      for score in SCORES),
    *(f"{stage}_{metric}_{score}" for stage in ("initial", "final")
      for metric in ("variance", "bound", "gain", "within_ss", "between_ss", "estimate_error")
      for score in SCORES),
]


def glp_experiment(report, checks):
    for h, q in GLP_CASES:
        checks.case = f"glp/{h}/{q}"
        design = glp_uniform_design(h, q).copy()
        midpoints = (np.arange(h) + .5) / h
        checks.eq("glp_latin_projections", np.max(abs(np.sort(design, axis=0) - midpoints[:, None])), 0)
        glp_uniform_design.cache_clear()
        checks.eq("glp_deterministic", np.max(abs(design - glp_uniform_design(h, q))), 0)
        if h == 1:
            report.append(dict(H=h, q=q, branch="center", points=design))
            continue
        modulus = h + 1
        units = [i for i in range(1, modulus) if gcd(i, modulus) == 1]
        powers = [tuple(pow(a, j, modulus) for j in range(q)) for a in units]
        candidates = [candidate for candidate in powers if len(set(candidate)) == q]
        branch = "powers"
        if not candidates:
            replace = q > len(units)
            branch = "fallback_with_replacement" if replace else "fallback_without_replacement"
            generator = np.random.default_rng(781211 + 997 * h + q)
            candidates = [tuple(generator.choice(units, size=q, replace=replace)) for _ in range(32)]
        points = np.array([
            ((np.outer(np.arange(1, h + 1), candidate) % modulus) - .5) / h
            for candidate in candidates
        ])
        md = np.array([qmc.discrepancy(candidate, method="MD") for candidate in points])
        winner = int(np.argmin(md))
        checks.eq("glp_first_md_minimizer", np.max(abs(design - points[winner])), 0)
        report.append(dict(
            H=h, q=q, branch=branch, units=units, candidate_count=len(candidates),
            winner_index=winner, generator=candidates[winner], candidate_md=md, points=design,
        ))


def tied_transport_experiment(report, checks):
    fixtures = (
        ("duplicate_scores_equal", np.repeat([[0., 0.], [1., 0.], [2., 0.]], 3, axis=0),
         [[0., 0.], [0., 0.], [2., 0.]], [3, 3, 3]),
        ("duplicate_atoms_unequal", np.repeat([[0.], [1.], [3.]], [3, 4, 5], axis=0),
         [[.5], [.5], [2.]], [3, 3, 6]),
        ("all_costs_tied_near_equal", np.zeros((10, 2)), [[1., 0.], [0., 1.], [-1., 0.]], [4, 3, 3]),
    )
    for name, x, anchors, capacities in fixtures:
        checks.case = name
        anchors = np.array(anchors)
        labels, cost = capacitated_assignment(x, anchors, capacities)
        lp, _, _, flow = transport_lp(x, anchors, np.ones(len(x)), capacities, checks)
        checks.eq("duplicate_fixture_frozen_cost_vs_lp", cost, lp)
        # Coalesce duplicates in both marginals independently to check the
        # atom interpretation even when arbitrary label optima are different.
        ux, nx = np.unique(x, axis=0, return_counts=True)
        ua, code = np.unique(anchors, axis=0, return_inverse=True)
        merged = np.bincount(code, weights=capacities)
        compact, _, _, _ = transport_lp(ux, ua, nx, merged, checks)
        checks.eq("coalesced_atom_transport", lp, compact)
        checks.eq("tied_fixture_lp_integral", np.max(abs(flow - flow.round())), 0)
        report.append(dict(
            name=name, scores=x, anchors=anchors, capacities=capacities, assignment=labels,
            frozen_cost=cost, lp_cost=lp, merged_lp_cost=compact, lp_flow=flow,
        ))


def star_discrepancy(points):
    """Exact finite anchored-grid search, including both one-sided limits."""
    q = points.shape[1]
    boundaries = [np.unique(np.r_[0., points[:, j], 1.]) for j in range(q)]
    corners = np.array(list(product(*boundaries)))
    volume = np.prod(corners, axis=1)
    inside_open = np.all(points[None, :, :] < corners[:, None, :], axis=2).mean(axis=1)
    inside_closed = np.all(points[None, :, :] <= corners[:, None, :], axis=2).mean(axis=1)
    plus, minus = inside_closed - volume, volume - inside_open
    i, j = int(np.argmax(plus)), int(np.argmax(minus))
    return dict(value=max(plus[i], minus[j]), positive=plus[i], negative=minus[j],
                positive_corner=corners[i], negative_corner=corners[j], corners=len(corners))


def monotone_atom_w2(points, constant=0., linear=1., quadratic=0.):
    """Integrate (Q(t)-x_(i))² exactly on each uniform mass interval."""
    values = np.sort(np.ravel(points))
    half_width = .5 / len(values)
    midpoint = (np.arange(len(values)) + .5) / len(values)
    # Around the midpoint the difference is a + b*v + c*v².
    a = constant + linear * midpoint + quadratic * midpoint**2 - values
    b = linear + 2 * quadratic * midpoint
    integral = np.mean(a**2 + (b**2 + 2 * a * quadratic) * half_width**2 / 3
                       + quadratic**2 * half_width**4 / 5)
    return float(np.sqrt(integral))


def empirical_w2_1d(x, y):
    edges = np.unique(np.r_[np.linspace(0, 1, len(x) + 1), np.linspace(0, 1, len(y) + 1)])
    middle = (edges[:-1] + edges[1:]) / 2
    x_quantile = np.sort(x.ravel())[(middle * len(x)).astype(int)]
    y_quantile = np.sort(y.ravel())[(middle * len(y)).astype(int)]
    return float(np.sqrt(np.diff(edges) @ (x_quantile - y_quantile) ** 2))


def reference_map(points):
    q = points.shape[1]
    return np.array([-1., -.9])[:q] + points * np.array([1.6, 2.1])[:q] + points**2 * np.array([.4, .2])[:q]


def grid_points(side):
    midpoint = (np.arange(side) + .5) / side
    return np.array(list(product(midpoint, repeat=2)))


def uniform_grid_lp(x, y, checks):
    n, h = len(x), len(y)
    mass = lcm(n, h)
    return transport_lp(x, y, np.full(n, mass // n), np.full(h, mass // h), checks)


def interval_from_grid(lp, radius):
    _, lower, upper, _ = lp
    return [max(0., float(np.sqrt(lower)) - radius), float(np.sqrt(upper)) + radius]


def transport_1d(points, z, mapped, estimated, lip, direct, proof_bound, checks):
    disc = star_discrepancy(points)["value"]
    ordered = np.sort(points.ravel())
    fraction = np.arange(len(ordered)) / len(ordered)
    checks.eq("star_1d_order_statistic", disc, np.max(np.maximum(
        fraction + 1 / len(ordered) - ordered, ordered - fraction
    )))
    uniform = monotone_atom_w2(points)
    reference = monotone_atom_w2(z, -1., 1.6, .4)
    pushed = monotone_atom_w2(mapped, -1., 1.6, .4)
    exact_map_error = empirical_w2_1d(mapped, estimated)
    error = np.sqrt(np.mean(np.sum((mapped - estimated) ** 2, axis=1)))
    checks.eq("empirical_1d_monotone_vs_assignment", direct, empirical_w2_1d(z, estimated))
    checks.le("discrepancy_bound_1d", uniform, proof_bound)
    checks.le("pushforward_lipschitz_1d", pushed, lip * uniform)
    checks.le("map_error_1d_index_coupling", exact_map_error, error)
    checks.le("triangle_1d", direct, reference + pushed + exact_map_error)
    checks.le("corollary_1d", direct, reference + lip * proof_bound + error)
    return dict(
        W2_uniform_exact=uniform, W2_empirical_to_reference_exact=reference,
        W2_pushed_exact=pushed, W2_anchor_map_error_exact=exact_map_error, e_Q_H=error,
        triangle_rhs=reference + pushed + exact_map_error,
        corollary_rhs=reference + lip * proof_bound + error,
    )


def transport_2d(points, z, mapped, estimated, lip, direct, proof_bound, checks):
    anchor_lp = uniform_grid_lp(mapped, estimated, checks)
    anchor_w2 = np.sqrt(anchor_lp[0])
    anchor_lower, anchor_upper = interval_from_grid(anchor_lp, 0.)
    error = np.sqrt(np.mean(np.sum((mapped - estimated) ** 2, axis=1)))
    checks.le("map_error_2d_index_coupling", anchor_w2, error)
    results = []
    for side in GRID_SIDES:
        grid = grid_points(side)
        mapped_grid = reference_map(grid)
        radius = np.sqrt(2 / 12) / side
        uniform = uniform_grid_lp(points, grid, checks)
        pushed = uniform_grid_lp(mapped, mapped_grid, checks)
        reference = uniform_grid_lp(z, mapped_grid, checks)
        unif_interval = interval_from_grid(uniform, radius)
        push_interval = interval_from_grid(pushed, lip * radius)
        ref_interval = interval_from_grid(reference, lip * radius)
        pushed_plan_cost = np.sum(uniform[3] * cdist(mapped, mapped_grid, "sqeuclidean")) / lcm(len(points), len(grid))
        unif_w2, push_w2, ref_w2 = (np.sqrt(lp[0]) for lp in (uniform, pushed, reference))
        checks.le("discrepancy_bound_2d_certified", unif_interval[1], proof_bound)
        checks.le("pushed_grid_lp_vs_pushed_plan", pushed[0], pushed_plan_cost)
        checks.le("pushed_plan_cost_lipschitz", pushed_plan_cost, lip**2 * uniform[0])
        checks.le("pushforward_lipschitz_2d_grid", push_w2, lip * unif_w2)
        checks.le("triangle_2d_grid", direct, ref_w2 + push_w2 + anchor_w2)
        checks.le("triangle_2d_continuous_certified", direct, ref_interval[1] + push_interval[1] + anchor_upper)
        checks.le("corollary_2d_certified", direct, ref_interval[1] + lip * proof_bound + error)
        results.append(dict(
            side=side, uniform_coupling_radius=radius, mapped_coupling_radius=lip * radius,
            W2_uniform_grid=unif_w2, W2_uniform_interval=unif_interval, W2_pushed_grid=push_w2,
            W2_pushed_interval=push_interval, W2_empirical_to_reference_grid=ref_w2,
            W2_empirical_to_reference_interval=ref_interval, pushed_grid_plan_W2=np.sqrt(pushed_plan_cost),
            triangle_grid_rhs=ref_w2 + push_w2 + anchor_w2,
            triangle_continuous_upper_rhs=ref_interval[1] + push_interval[1] + anchor_upper,
            triangle_continuous_lower_rhs=ref_interval[0] + push_interval[0] + anchor_lower,
            corollary_upper_rhs=ref_interval[1] + lip * proof_bound + error,
        ))
    for metric in ("W2_uniform_interval", "W2_pushed_interval", "W2_empirical_to_reference_interval"):
        checks.le(f"nested_resolution_intervals_overlap/{metric}",
                  max(result[metric][0] for result in results), min(result[metric][1] for result in results))
    return dict(W2_anchor_map_error=anchor_w2, e_Q_H=error, grids=results)


def discrepancy_experiment(report, checks):
    checks.case = "anchored_grid_duplicate_fixture"
    checks.eq("star_duplicate_fixture", star_discrepancy(np.full((3, 2), .5))["value"], .75)
    for q, sizes in TRANSPORT.items():
        for h in sizes:
            n = 4 * h
            capacities = np.full(h, 4)
            raw = rng(1, q, h).normal(size=(n, q))
            if q == 2:
                first = raw[:, 0] - raw[:, 0].mean()
                second = raw[:, 1] - raw[:, 1].mean()
                second -= first * (first @ second) / (first @ first)
                raw[:, 1] = second / second.std() + .25 * first / first.std()
            _, z, _, glp, _, _, frozen_cost, _, _, _ = frozen_design(raw, capacities)
            checks.case = f"discrepancy/{q}/{h}"
            checks.eq("transport_example_retained_q", z.shape[1], q)
            for kind, point_type in enumerate(POINT_TYPES):
                checks.case = f"discrepancy/{q}/{h}/{point_type}"
                if kind == 0:
                    points = glp
                elif kind == 1:
                    points = .12 + .20 * rng(2, q, h, kind).random((h, q))
                else:
                    points = rng(2, q, h, kind).random((h, q))
                discrepancy = star_discrepancy(points)
                delta = discrepancy["value"]
                m = ceil(delta ** (-1 / (q + 2)))
                proof_bound = np.sqrt(q / m**2 + q * 2**(q - 1) * m**q * delta)
                constant_bound = np.sqrt(q * (1 + 2**(2 * q - 1))) * delta ** (1 / (q + 2))
                checks.le("proof_bound_to_power_bound", proof_bound, constant_bound)
                estimated = inverse_quantile_anchors(z, points)
                mapped = reference_map(points)
                lip = 2.4 if q == 1 else 2.5
                _, cost = capacitated_assignment(z, estimated, capacities)
                direct = np.sqrt(cost)
                if kind == 0:
                    checks.eq("corollary_glp_cost_matches_frozen", cost, frozen_cost)
                if q == 1:
                    metrics = transport_1d(points, z, mapped, estimated, lip, direct, proof_bound, checks)
                else:
                    lp, _, _, _ = uniform_grid_lp(z, estimated, checks)
                    checks.eq("corollary_2d_empirical_cost_vs_lp", lp, cost)
                    metrics = transport_2d(points, z, mapped, estimated, lip, direct, proof_bound, checks)
                report.append(dict(
                    q=q, H=h, N=n, point_type=point_type, points=points, pc_scores=z,
                    empirical_anchors=estimated, reference_anchors=mapped, discrepancy=discrepancy,
                    proof_M=m, proof_bound=proof_bound, power_bound=constant_bound,
                    L_Q=lip, sqrt_transport=direct, **metrics,
                ))


def protocol():
    root = Path(__file__).resolve().parents[1]
    sources = (
        "code/theory_check_geometry.py", "code/real_application.py",
        "code/real_core/designs.py", "code/legacy_locked/designs.py",
        "code/legacy_locked/FINAL_METHOD_LOCK.json",
    )
    return dict(
        scope="BUD geometry, fixed-score design variance, and transport, supplement S1–S3 and S7",
        seed=SEED, replicates_per_condition=REPS, scenarios=SCENARIOS,
        rng="NumPy default_rng with SeedSequence([seed, *key]); independent frame keys (0, scenario index, replicate)",
        frame_dgp={
            "univariate": "N iid standard normals",
            "multivariate": "Independent standard normal latent vectors a,b and iid normal errors. "
                            "X=(a,a,b,b)+errors scaled by (.20,.38,.28,.48).",
            "budget": "m=2H, with two SRSWOR draws per cell. First N mod H capacities are ceil(N/H).",
            "design": "Frozen standardization, oriented q85 PCA, MD-selected GLP, linear empirical quantiles, "
                      "expanded exact capacitated assignment, one full-X centroid reassignment.",
        },
        score_dgp={
            "projection": "g(X)=standardized X dotted with (1,...,p)/norm(1,...,p); L=1.",
            "projection_plus_noise": "g plus iid N(0,.35²) from the frame stream after generating X.",
            "right_cluster": "Final-cell mean of g for each unit.",
            "wrong_cluster": "g minus its final-cell mean.",
            "zero_gain": "Wrong score plus right score scaled by sqrt((V_BUD(wrong)-V_SRS(wrong))/V_SRS(right)).",
            "residual": "For each score, r=score-g. All scores are fixed after design construction. "
                        "The three constructed scores are diagnostic fixtures on the same frozen X.",
            "draw": "One independent continuous priority per unit from the frame stream after the noise. "
                    "Share priorities across scores, initial/final BUD, and the SRS variance reference. "
                    "Exact conditional moments are the primary checks.",
            "sign_count_order": ["positive", "zero within absolute tolerance", "negative"],
        },
        replications="Every prespecified frame and case is retained in order. No exclusions, retries, or imputation. "
                     "On exception, save all completed rows and report the terminating case.",
        approximate_reassignment="Adversarial single cross-cell swap of initial labels maximizing the increase "
                                 "in recentered scatter. Ties use first row-index pair. Epsilon is the positive "
                                 "part of the achieved old-center cost excess over initial cost. Also record excess "
                                 "over the exact minimum. Swap indices are zero-based.",
        transport_cases={
            "dimensions_and_H": TRANSPORT, "point_types": POINT_TYPES,
            "points": "Frozen GLP or independently generated H by q uniforms, with clustered coordinates "
                      "on [.12,.32) and iid coordinates on [0,1); key (2,q,H,type index).",
            "pc_frames": "Equal capacity 4. Start with iid normal raw matrix from key (1,q,H). "
                         "In 2D, center both columns, remove the first-column projection from the second, "
                         "then replace column 2 by standardized residual plus .25 times standardized column 1. "
                         "Apply the frozen PCA. Clustered and iid points are diagnostic alternatives for S3.",
            "reference_map": "Q1(t)=-1+1.6t+.4t²; Q2(t)=-.9+2.1t+.2t² coordinatewise. "
                             "Maximum derivatives 2.4 and 2.5 bound the Euclidean Lipschitz constants.",
            "grid_sides": GRID_SIDES,
            "grid_certificate": "Discretize uniform mass onto equal square midpoint cells. The exact "
                                "uniform-to-midpoint W2 is sqrt(2/12)/M from the within-square second moment "
                                "and nearest-center optimality. Mapping the coupling gives error at most L_Q "
                                "times this radius. The LP primal/dual objectives are padded for numerical "
                                "residuals before applying the reverse triangle inequality.",
            "proof_bound": "Use S3's ceil(delta^(-1/(q+2))) in the explicit proof inequality, without "
                           "optimizing M or substituting mixture discrepancy.",
        },
        glp_cases=GLP_CASES,
        tolerances={
            "absolute": ABS_TOL, "relative": REL_TOL,
            "rule": "abs_tol + rel_tol * max(abs(lhs),abs(rhs)); empirical sign threshold abs_tol",
            "highs_primal_dual_feasibility": LP_TOL,
        },
        source_sha256={f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in sources},
        versions={"Python": platform.python_version(), "NumPy": np.__version__, "SciPy": scipy.__version__},
    )


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value {type(value)}")


def json_report(report):
    # Keep the summary readable and give each frame a single line.
    tables = []
    for group in report["frames"]:
        tables.append(group.pop("trials"))
        group["trials"] = f"<geometry frame rows {len(tables) - 1}>"
    formatted = json.dumps(report, default=json_value, indent=2, allow_nan=False)
    for i, table in enumerate(tables):
        block = "[\n" + ",\n".join(
            "        " + json.dumps(row, default=json_value, allow_nan=False, separators=(",", ":"))
            for row in table
        ) + "\n      ]" if table else "[]"
        formatted = formatted.replace(f'"<geometry frame rows {i}>"', block, 1)
    return formatted + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="fresh output JSON path")
    args = parser.parse_args()
    if args.out.suffix.lower() != ".json":
        parser.error("--out must be a .json file")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        output = args.out.open("x", encoding="utf-8", newline="\n")
    except FileExistsError:
        parser.error(f"output already exists: {args.out}")
    checks = Checks()
    spec = protocol()
    started = datetime.now(timezone.utc).isoformat()
    start_clock = perf_counter()
    report = dict(
        protocol=spec, invocation=sys.argv, started_utc=started, status="running",
        frame_columns=FRAME_COLUMNS, frames=[], tie_cases=[], glp_cases=[], transport_cases=[],
    )
    with output:
        # Reserve a fresh name and commit the protocol before any experiment.
        json.dump(report, output, default=json_value, indent=2, allow_nan=False)
        output.flush()
        try:
            glp_experiment(report["glp_cases"], checks)
            tied_transport_experiment(report["tie_cases"], checks)
            discrepancy_experiment(report["transport_cases"], checks)
            frame_experiment(report["frames"], checks)
        except Exception:
            report["exception"] = dict(case=checks.case, traceback=traceback.format_exc())
        report["check_metrics"] = checks.metrics
        report["failed_checks"] = checks.failures
        report["status"] = "failed" if checks.failures or "exception" in report else "passed"
        report["seconds"] = perf_counter() - start_clock
        final_json = json_report(report)
        output.seek(0)
        output.write(final_json)
        output.truncate()
    print(f"{report['status']}: wrote {args.out} ({report['seconds']:.2f}s)")
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
