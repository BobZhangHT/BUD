"""Finite-population stratified sampling and positive rescaled KM bootstrap.

The estimand is the complete-cohort Kaplan-Meier RMST, not the
superpopulation RMST.  The integration below is exact for the KM step
function; no time discretisation or trapezoidal approximation is used.

Arrays returned by ``draw_sample`` are grouped by stratum, and ``meta``
records that order.  Rao-Wu resampling is conditional on these fixed strata.
"""
from __future__ import annotations

import itertools
import json
import numpy as np


def _check_observations(time, event, tau):
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=bool)
    if t.ndim != 1 or e.shape != t.shape or not len(t):
        raise ValueError("time and event must be nonempty one-dimensional arrays of equal length")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be positive and finite")
    if np.any(np.isnan(t)) or np.any(t < 0):
        raise ValueError("observed times must be nonnegative and not NaN")
    return t, e


def _event_layout(time, event, tau):
    """Indices of distinct event times; ties have a single KM jump."""
    order = np.argsort(time, kind="stable")
    ts = time[order]
    es = event[order]
    first = np.r_[0, np.flatnonzero(ts[1:] != ts[:-1]) + 1]
    all_times = ts[first]
    deaths = np.add.reduceat(es.astype(np.int64), first)
    use = (deaths > 0) & (all_times < tau)
    return order, es, first, use, all_times[use], deaths[use]


def km_rmst(time, event, weights, tau):
    """Exact weighted Kaplan-Meier integral over [0, tau].

    ``weights`` is either (n,) or (B,n); the return is a scalar or (B,).
    All weights must be nonnegative, with a positive total in every row.
    Events at tau do not affect its integral.  Empty risk sets in a
    zero-weight tail use the usual carry-forward KM convention.  Such a
    tail is an estimability issue and should be monitored in simulations.
    """
    t, e = _check_observations(time, event, tau)
    w = np.asarray(weights, dtype=float)
    scalar = w.ndim == 1
    if scalar:
        w = w[None, :]
    if w.ndim != 2 or w.shape[1] != len(t):
        raise ValueError("weights must have shape (n,) or (B,n)")
    if np.any(~np.isfinite(w)) or np.any(w < 0) or np.any(w.sum(axis=1) <= 0):
        raise ValueError("weights must be finite, nonnegative, and have positive row totals")
    order, es, first, use, ev_times, _ = _event_layout(t, e, tau)
    if not len(ev_times):
        ans = np.full(w.shape[0], float(tau))
    else:
        ws = w[:, order]
        risk = np.cumsum(ws[:, ::-1], axis=1)[:, ::-1][:, first[use]]
        deaths = np.add.reduceat(ws * es[None, :], first, axis=1)[:, use]
        hazard = np.divide(deaths, risk, out=np.zeros_like(deaths), where=risk > 0)
        hazard = np.clip(hazard, 0.0, 1.0)
        survival_after = np.cumprod(1.0 - hazard, axis=1)
        widths = np.diff(np.r_[0.0, ev_times, float(tau)])
        ans = widths[0] + survival_after @ widths[1:]
    return float(ans[0]) if scalar else ans


def km_if(time, event, tau):
    """Exact first derivative (population-normalised IF) of cohort KM RMST.

    With base raw observation weights all equal to one, the derivative
    with respect to raw weight i equals phi[i] / n.  This finite-cohort
    derivative is an oracle diagnostic and is not a design input.
    """
    t, e = _check_observations(time, event, tau)
    n = len(t)
    order, es, first, use, ev_times, deaths_count = _event_layout(t, e, tau)
    if not len(ev_times):
        return np.zeros(n)
    risk_count = n - first[use]
    y = risk_count / n
    alpha = deaths_count / risk_count
    sj = np.cumprod(1.0 - alpha)
    widths_after = np.diff(np.r_[ev_times, float(tau)])
    tail_area = np.cumsum((sj * widths_after)[::-1])[::-1]
    den = y * (1.0 - alpha)
    coef = np.divide(tail_area, den, out=np.zeros_like(tail_area), where=den > 0)
    # A jump equal to one must exhaust the risk set.  Its perturbation is
    # identically zero for arbitrary positive subject weights, so its
    # contribution is zero even though the quotient reads 0/0.
    risk_prefix = np.r_[0.0, np.cumsum(coef * alpha)]
    phi = risk_prefix[np.searchsorted(ev_times, t, side="right")]
    failures = e & (t < tau)
    phi[failures] -= coef[np.searchsorted(ev_times, t[failures])]
    phi -= phi.mean()  # Remove only floating-point centering error.
    return phi


def draw_sample(labels, allocation, rng, priorities=None):
    """Independent SRSWOR in each stratum, optionally using common priorities.

    allocation: integer per-stratum sample size; dict keyed by label; or
    vector in the sorted order of np.unique(labels).
    priorities: length-N iid continuous uniforms.  Reusing the same array
    across designs produces valid common-random-number comparisons.

    Returns indices, inverse-probability weights, and list-of-dict meta.
    The indices/weights are grouped by stratum, as described by meta.
    """
    lab = np.asarray(labels)
    if lab.ndim != 1 or not len(lab):
        raise ValueError("labels must be a nonempty one-dimensional array")
    unique, inv, sizes = np.unique(lab, return_inverse=True, return_counts=True)
    h = len(unique)
    if isinstance(allocation, dict):
        alloc = np.asarray([allocation[v.item() if hasattr(v, "item") else v] for v in unique])
    elif np.ndim(allocation) == 0:
        alloc = np.full(h, allocation)
    else:
        alloc = np.asarray(allocation)
    if alloc.shape != (h,) or np.any(alloc != np.floor(alloc)):
        raise ValueError("allocation must contain one integer per stratum")
    alloc = alloc.astype(int)
    if np.any(alloc < 1) or np.any(alloc > sizes):
        raise ValueError("each sample size must lie between one and the stratum population size")
    pri = rng.random(len(lab)) if priorities is None else np.asarray(priorities, dtype=float)
    if pri.shape != lab.shape or np.any(~np.isfinite(pri)):
        raise ValueError("priorities must be a finite length-N array")
    groups = np.argsort(inv, kind="stable")
    edges = np.r_[0, np.cumsum(sizes)]
    selected, weight_parts, meta = [], [], []
    offset = 0
    for j, (label, kh, sh) in enumerate(zip(unique, sizes, alloc)):
        members = groups[edges[j]:edges[j+1]]
        take = members if sh == kh else members[np.argpartition(pri[members], sh - 1)[:sh]]
        take = np.sort(take)
        selected.append(take)
        weight_parts.append(np.full(sh, kh / sh))
        meta.append({"label": label.item() if hasattr(label, "item") else label,
                     "N": int(kh), "s": int(sh), "start": offset,
                     "stop": offset + int(sh), "f": float(sh / kh)})
        offset += int(sh)
    return np.concatenate(selected), np.concatenate(weight_parts), meta


def _validate_meta(weights, meta):
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or np.any(w <= 0) or np.any(~np.isfinite(w)):
        raise ValueError("base weights must be a positive finite vector")
    cursor = 0
    for cell in meta:
        k, s = int(cell["N"]), int(cell["s"])
        start, stop = int(cell["start"]), int(cell["stop"])
        if start != cursor or stop - start != s or not 1 <= s <= k:
            raise ValueError("meta must describe consecutive valid sample slices")
        if s < 2 and s != k:
            raise ValueError("a noncensus stratum requires at least two sampled units")
        if not np.allclose(w[start:stop], k / s, rtol=1e-12, atol=1e-12):
            raise ValueError("base weights must be the stratum inverse sampling probabilities")
        cursor = stop
    if cursor != len(w):
        raise ValueError("meta must cover every sampled observation exactly once")
    return w


def _paired_rescaled_weights(weights, meta, signs):
    """Deterministic paired form, also used for exact covariance verification."""
    signs = np.asarray(signs, dtype=float)
    if signs.ndim != 2 or signs.shape[1] != len(meta) or np.any(np.abs(signs) != 1):
        raise ValueError("signs must be a B-by-H matrix of minus/plus one")
    if any(cell["s"] != 2 for cell in meta):
        raise ValueError("paired form requires two sampled subjects in every stratum")
    a = np.sqrt(1.0 - np.asarray([2.0 / cell["N"] for cell in meta]))
    perturbation = signs * a[None, :]
    out = np.empty((len(signs), len(weights)), dtype=float)
    out[:, 0::2] = weights[None, 0::2] * (1.0 + perturbation)
    out[:, 1::2] = weights[None, 1::2] * (1.0 - perturbation)
    return out


def bootstrap_weights(weights, meta, rng, B):
    """Positive Rao-Wu rescaled weights with exact finite-population factor.

    For a cell of population k and sample s, draw multinomial counts C
    from s-1 iid draws on the s sampled subjects and use
        (k/s) [1-a + a*s*C/(s-1)],  a=sqrt(1-s/k).
    Their total is exactly k.  The paired s=2 case uses Rademacher signs.
    """
    w = _validate_meta(weights, meta)
    if int(B) != B or B < 1:
        raise ValueError("B must be a positive integer")
    B = int(B)
    if all(cell["s"] == 2 for cell in meta):
        signs = 2 * rng.integers(0, 2, size=(B, len(meta)), dtype=np.int8) - 1
        return _paired_rescaled_weights(w, meta, signs)
    out = np.broadcast_to(w, (B, len(w))).copy()
    for cell in meta:
        k, s = cell["N"], cell["s"]
        if s == k:
            continue
        a = np.sqrt(1.0 - s / k)
        counts = rng.multinomial(s - 1, np.full(s, 1.0 / s), size=B)
        factors = 1.0 - a + (a * s / (s - 1.0)) * counts
        out[:, cell["start"]:cell["stop"]] *= factors
    return out


def bootstrap_rmst(time, event, weights, meta, rng, B, tau):
    """Return B exact weighted-KM RMST bootstrap replicates for sampled data."""
    wstar = bootstrap_weights(weights, meta, rng, B)
    return km_rmst(time, event, wstar, tau)


def verification():
    """Meaningful identities for the KM target, derivative and resampling law."""
    rng = np.random.default_rng(471002)
    t = np.array([0.0, .2, .2, .8, 1.1, 1.8, 2.1, 3.0])
    tau = 2.0
    w = rng.uniform(.2, 2.0, size=(17, len(t)))
    observed = km_rmst(t, np.ones(len(t), bool), w, tau)
    expected = (w @ np.minimum(t, tau)) / w.sum(axis=1)
    mean_error = float(np.max(np.abs(observed - expected)))
    assert mean_error < 1e-12, (observed, expected)
    assert km_rmst(t, np.zeros(len(t), bool), np.ones(len(t)), tau) == tau
    # Censoring, ties, a tau event, and an event after tau are all present.
    t2 = np.array([.12, .4, .4, .73, 1.0, 1.31, 1.73, 2.0, 2.4])
    e2 = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1], bool)
    phi = km_if(t2, e2, tau)
    eps = 1e-5
    wplus = np.ones((len(t2), len(t2))) + eps * np.eye(len(t2))
    wminus = np.ones((len(t2), len(t2))) - eps * np.eye(len(t2))
    fd = len(t2) * (km_rmst(t2, e2, wplus, tau) - km_rmst(t2, e2, wminus, tau)) / (2 * eps)
    if_error = float(np.max(np.abs(phi - fd)))
    assert if_error < 2e-8, (phi, fd)
    phi_nc = km_if(t, np.ones(len(t), bool), tau)
    assert np.max(np.abs(phi_nc - (np.minimum(t, tau)-np.minimum(t, tau).mean()))) < 1e-12
    # Enumerating all independent paired sign configurations makes this
    # covariance check exact, with no bootstrap Monte Carlo tolerance.
    labels = np.repeat(np.arange(3), [7, 8, 11])
    idx, base, meta = draw_sample(labels, 2, rng)
    signs = np.asarray(list(itertools.product([-1, 1], repeat=3)))
    ws = _paired_rescaled_weights(base, meta, signs)
    v = rng.normal(size=(len(idx), 2))
    total_n = len(labels)
    estimates = ws @ v / total_n
    cov_actual = np.cov(estimates, rowvar=False, ddof=0)
    cov_expected = np.zeros((2, 2))
    for cell in meta:
        sl = slice(cell["start"], cell["stop"])
        shat = np.cov(v[sl], rowvar=False, ddof=1)
        cov_expected += (cell["N"] / total_n)**2 * (1-cell["f"]) / cell["s"] * shat
        assert np.max(np.abs(ws[:, sl].sum(axis=1)-cell["N"])) < 1e-12
    cov_error = float(np.max(np.abs(cov_actual - cov_expected)))
    assert cov_error < 1e-12
    assert np.min(ws) > 0
    # General s=3/4 law and singleton census: positive, fixed cell mass.
    lab2 = np.repeat(np.arange(3), [12, 13, 1])
    ix2, w2, me2 = draw_sample(lab2, [3, 4, 1], rng)
    bs2 = bootstrap_weights(w2, me2, rng, 500)
    for cell in me2:
        sl = slice(cell["start"], cell["stop"])
        assert np.max(np.abs(bs2[:, sl].sum(axis=1)-cell["N"])) < 1e-12
    assert np.min(bs2) > 0
    # Identical supplied priorities reproduce the exact sample, independent
    # of rng state (valid coupled sampling for paired method comparisons).
    pri = rng.uniform(size=len(labels))
    a = draw_sample(labels, 2, rng, priorities=pri)[0]
    b = draw_sample(labels, 2, np.random.default_rng(5), priorities=pri)[0]
    assert np.array_equal(a, b)
    return {"no_censor_mean_identity_max_error": mean_error,
            "if_finite_difference_max_error": if_error,
            "exact_paired_covariance_max_error": cov_error,
            "all_checks_passed": True}


if __name__ == "__main__":
    print(json.dumps(verification(), indent=2))
