"""Check finite-cohort precision and horizon support for the BUD budget rule."""

import argparse
from itertools import combinations, product
import hashlib
import json
from math import comb, exp, floor, sqrt
from pathlib import Path
from statistics import NormalDist
import sys

import bud_experiment

sys.path.insert(0, str(bud_experiment.LEGACY))
import numpy as np
import pandas as pd
from scipy.special import expit
from refinement import refined_design
from survival import km_if, km_rmst, _paired_rescaled_weights

ROOT = bud_experiment.PROJECT
N, COHORTS, DRAWS, BOOTSTRAP, PILOT_N = 120, 50, 20, 1000, 2400
GRID = np.array([1., 2., 3.])
TAU = 4.
BUDGETS = (12, 20, 30, 40, 48, 60, 80)
TARGET, BETA = .32, .05
Z = NormalDist().inv_cdf(.975)
SEED = 840917


def cohort(rng, n, late_events=False, support=.9):
    x = rng.standard_normal((n, 4))
    alive = rng.random(n) < expit(-.15 + .6*x[:, 0] - .4*x[:, 1])
    early = expit(.18 + .45*x[:, 2])
    middle = expit(.18 - .35*x[:, 3])
    u, v = rng.random((2, n))
    failure = np.where(u < early, 1., np.where(v < middle, 2., 3.))
    follow_up = np.where(rng.random(n) < support, 4.5, 3.5)
    return x, np.where(alive, follow_up, 3. if late_events else failure), ~alive


def empty_pair_probability(counts, k):
    missed_pairs = (k-counts) * (k-counts-1)
    return float(np.prod(missed_pairs / (k*(k-1))))


def estimates(event_risk):
    hazard = np.divide(event_risk[..., :3], event_risk[..., 3:],
                       out=np.zeros_like(event_risk[..., :3]), where=event_risk[..., 3:] > 0)
    return 1 + np.cumprod(1 - hazard, axis=-1).sum(axis=-1)


def coefficient(k):
    return k*(k-2) / (2*N*(k-1))


def planned_candidates(moment, risk):
    rows = []
    for m in BUDGETS:
        k = N//(m//2)
        srs_variance = (1/m - 1/N) * (N/(N-1)) * moment
        bound = coefficient(k)*moment
        risk_paired = exp(-2*risk/k)
        risk_srs = comb(N-risk, m)/comb(N, m)
        width = Z*sqrt(max(bound, srs_variance))
        risk_bound = max(risk_paired, risk_srs)
        rows.append(dict(m=m, max_projected_half_width=width, max_projected_empty=risk_bound,
                         passes=width <= TARGET and risk_bound <= BETA))
    return rows


def enumerate_small():
    cells = ((3, 1), (4, 1), (5, 0))
    flags = [np.arange(k) < risk for k, risk in cells]
    pair_counts = []
    for frame in flags:
        pair_counts.append([int(frame[list(ids)].sum()) for ids in combinations(range(len(frame)), 2)])
    outcomes = np.array(list(product(*pair_counts)))
    p_pair = float(np.mean(outcomes.sum(axis=1) == 0))
    exact_pair = np.prod([comb(k-r, 2)/comb(k, 2) for k, r in cells])
    upper_pair = exp(-2*sum(r for k, r in cells)/max(k for k, r in cells))
    all_flags = np.concatenate(flags)
    srs_empty = [not all_flags[list(ids)].any() for ids in combinations(range(len(all_flags)), 6)]
    p_srs = float(np.mean(srs_empty))
    exact_srs = comb(len(all_flags)-int(all_flags.sum()), 6)/comb(len(all_flags), 6)
    p_either = float(np.logical_or.outer(outcomes.sum(axis=1) == 0, srs_empty).mean())
    exact_either = 1-(1-exact_pair)*(1-exact_srs)
    errors = np.abs(np.array([p_pair-exact_pair, p_srs-exact_srs, p_either-exact_either]))
    assert errors.max() < 1e-14 and p_pair <= upper_pair
    return dict(paired=p_pair, paired_upper=upper_pair, srs=p_srs, either=p_either,
                max_formula_error=float(errors.max()))


def phase_ii(cohort_id, m, time, e, grouped, phi, truth, check_weights):
    h, k = grouped.shape
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 2, cohort_id]))
    priorities = rng.random((DRAWS, N))
    order = np.argsort(priorities[:, grouped], axis=2)[..., :2]
    selected = grouped[np.arange(h)[None, :, None], order]
    v = np.c_[e[:, None] & (time[:, None] == GRID), time[:, None] >= GRID].astype(float)
    pair_v = v[selected]
    observed = pair_v.mean(axis=(1, 2))
    estimate = estimates(observed)
    signs = 2*rng.integers(0, 2, size=(DRAWS, BOOTSTRAP, h), dtype=np.int8).astype(float)-1
    scale = sqrt(1 - 2/k)/(2*h)
    increments = signs @ (scale*(pair_v[:, :, 0, :]-pair_v[:, :, 1, :]))
    boot = estimates(observed[:, None, :] + increments)
    if check_weights:
        weights = np.full(m, k / 2)
        meta = [dict(start=2*j, stop=2*j+2, N=k, s=2, f=2/k) for j in range(h)]
        ids = selected[0].ravel()
        reweighted = _paired_rescaled_weights(weights, meta, signs[0, :4])
        np.testing.assert_allclose(km_rmst(time[ids], e[ids], reweighted, TAU), boot[0, :4], atol=1e-12, rtol=0)
    qlo, qhi = np.quantile(boot-estimate[:, None], [.025, .975], axis=1)
    error = estimate-truth
    linear = phi[selected].mean(axis=(1,2))
    return dict(sample_index=np.arange(DRAWS), error=error, half_width=(qhi-qlo)/2,
                covered=(qlo <= error) & (error <= qhi),
                empty=(time[selected] < TAU).all(axis=(1,2)),
                linear_exceed=np.abs(linear) >= TARGET)


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    scenario_specs = {"matched": (False, .9), "low_pilot_variation": (True, .9),
                      "optimistic_pilot_support": (False, .025)}
    diagnostics, draws, projections = [], [], {}
    for name, (pilot_late, target_support) in scenario_specs.items():
        _, p_time, p_event = cohort(np.random.default_rng(np.random.SeedSequence([SEED, 0])),
                                    PILOT_N, late_events=pilot_late)
        score = km_if(p_time, p_event, TAU)
        moment = float(np.mean(score**2))
        risk = floor(N*np.mean(p_time >= TAU))
        plans = planned_candidates(moment, risk)
        chosen = next(row["m"] for row in plans if row["passes"])
        preceding = BUDGETS[BUDGETS.index(chosen)-1] if chosen > BUDGETS[0] else None
        projections[name] = dict(pilot_score_moment=moment, pilot_projected_risk=risk,
                                 chosen=chosen, preceding=preceding, candidates=plans)
        for j in range(COHORTS):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, 1, j]))
            x, time, event = cohort(rng, N, support=target_support)
            truth = km_rmst(time, event, np.ones(N), TAU)
            phi = km_if(time, event, TAU)
            actual_moment = float(np.mean(phi**2))
            r_true = int(np.sum(time >= TAU))
            exact = {}
            for m in BUDGETS:
                labels, _, _ = refined_design(x, m)
                grouped = np.argsort(labels, kind="stable").reshape(m//2, -1)
                k = grouped.shape[1]
                W = phi[grouped].var(axis=1, ddof=1).mean()
                V = (1/m-1/N)*W
                count = np.sum(time[grouped] >= TAU, axis=1)
                exact[m] = (grouped, V, empty_pair_probability(count,k), W,
                            coefficient(k)*actual_moment, exp(-2*r_true/k))
            W0 = max(row[3] for row in exact.values())
            cutoff = Z**2*W0/(TARGET**2 + Z**2*W0/N)
            for m, (_, v, p0, _, upper, risk_bound) in exact.items():
                assert v <= upper + 1e-12 and p0 <= risk_bound + 1e-12
                if m >= cutoff:
                    assert Z*sqrt(v) <= TARGET + 1e-12
            for role, m in (("planned", chosen), ("preceding", preceding)):
                if m is None:
                    continue
                grouped, v, p0, _, upper, risk_bound = exact[m]
                diagnostic = dict(regime=name, role=role, m=m, cohort=j, exact_half_width=Z*sqrt(v),
                                  exact_empty=p0, oracle_empty_bound=risk_bound, target_risk=r_true,
                                  oracle_width_bound=Z*sqrt(upper), chebyshev_bound=min(1., upper/TARGET**2),
                                  meets_exact=(Z*sqrt(v) <= TARGET and p0 <= BETA))
                diagnostics.append(diagnostic)
                draws.append(pd.DataFrame(phase_ii(j,m,time,event,grouped,phi,truth,j == 0)).assign(
                    regime=name, role=role, m=m, cohort=j))
        print(f"{name}: selected {chosen}, preceding {preceding}", flush=True)
    data, diagnostic = pd.concat(draws, ignore_index=True), pd.DataFrame(diagnostics)
    keys = ["regime", "role", "m", "cohort"]
    assert not data.duplicated(keys+["sample_index"]).any()
    assert data.groupby(keys).size().eq(DRAWS).all()
    blocks = data.groupby(keys, as_index=False).agg(half_width=("half_width","mean"),
                   coverage=("covered","mean"), empty=("empty","mean"), linear_exceed=("linear_exceed","mean"))
    blocks = blocks.merge(diagnostic, on=keys, validate="one_to_one")
    summary = []
    metrics = ("half_width","coverage","empty","exact_empty","exact_half_width",
               "oracle_width_bound","oracle_empty_bound","chebyshev_bound","linear_exceed","meets_exact")
    for (name, role, m), group in blocks.groupby(keys[:3]):
        entry = dict(regime=name, role=role, m=int(m), cohorts=len(group), draws=len(group)*DRAWS)
        for col in metrics:
            entry[col] = float(group[col].mean())
            entry[col+"_mcse"] = float(group[col].std(ddof=1)/sqrt(len(group)))
        discrepancy = group["empty"] - group["exact_empty"]
        entry["empty_discrepancy_mcse"] = float(discrepancy.std(ddof=1) / sqrt(len(group)))
        summary.append(entry)
    protocol = ROOT/"handoff/THEORY_VALIDATION_PROTOCOL.md"
    report = dict(target=TARGET, beta=BETA, n=N, pilot_n=PILOT_N, B=BOOTSTRAP, seed=SEED,
                  budget_candidates=BUDGETS, exact_checks=enumerate_small(), projections=projections,
                  summary=summary, source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                for p in [Path(__file__).resolve(), protocol]})
    data.to_csv(output/"budget_draws.csv.gz", index=False)
    blocks.to_csv(output/"budget_cohorts.csv", index=False)
    (output/"budget.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
