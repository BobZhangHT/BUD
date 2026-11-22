#!/usr/bin/env python3
"""Continuity check against the project's historical process-PC1 comparator.

The original synthetic generators are preserved in legacy/.  Their pe/py are
DGP-derived conditional-hazard predictions, so MVRS-PROXY has extra prognostic
information unavailable to the covariate-only methods.  It is a deliberately
strong historical comparator, not the published MVRS algorithm and not a
same-information comparison.  It must never be relabelled PC1-X.

All methods estimate the same complete-cohort exact KM RMST contrast.  Legacy
integer times u are mapped to u*DT and administratively capped at tau=3.  This
step-function integral differs from the historical right-endpoint grid sum;
both targets and their difference are saved for transparency.
"""

import os
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import norm

from designs import make_design, design_variance, equal_quantile_labels
from legacy import formal_vo_dds as legacy_generator
from legacy.run_study import DT, J, TGRID, predict_if
from run_experiments import stable_seed, summarize
from survival import km_rmst, km_if, draw_sample, bootstrap_rmst


ROOT = Path(__file__).resolve().parent
SCENES = ("banana", "saddle3d", "multimodal", "gaussian")
PROXY_METHOD = "MVRS-PROXY"


def legacy_grid_mean(u, event):
    """Historical right-endpoint sum, retained only to record target changes."""
    risk = (u[:, None] >= TGRID[None, :]).sum(axis=0)
    deaths = ((u[:, None] == TGRID[None, :]) & event[:, None].astype(bool)).sum(axis=0)
    hazard = deaths / np.maximum(risk, 1)
    return float(DT * np.cumprod(1.0 - hazard).sum())


def proxy_partition(x, pe, py, m):
    """Exactly the historical predicted IF-process PCA direction, H=m/2.

    This uses the unstandardized IF-process coordinates at t >= 0.5, as in
    legacy run_study.mvrs_labels.  A sign convention is immaterial because
    equal quantile partitions are invariant to reversal up to cell labels.
    """
    start = perf_counter()
    if m % 2:
        raise ValueError("Paired process-PC1 requires an even sampling budget")
    mu = predict_if(pe, py)
    eligible = TGRID * DT >= 0.5
    centered = mu[:, eligible] - mu[:, eligible].mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    score = centered @ vt[0]
    h = m // 2
    labels = equal_quantile_labels(score, h)
    allocation = np.full(h, 2, dtype=int)
    sizes = np.bincount(labels)
    explained = float(singular[0]**2 / max(float(np.sum(singular**2)), 1e-30))
    info = dict(variant=PROXY_METHOD, N=len(x), m=m, p=x.shape[1], q=1, H=h,
                min_population_cell=int(sizes.min()), max_population_cell=int(sizes.max()),
                min_sample_cell=2, max_sample_cell=2,
                proxy_process_first_fraction=explained,
                seconds=perf_counter()-start)
    return labels, allocation, info


def one_historical_cohort(task):
    scene, cohort, spec = task
    rng = np.random.default_rng(stable_seed(spec["seed"], scene, cohort, "legacy-cohort"))
    # This global is local to each process; the original generator body stays
    # unmodified and already derives every array size from N.
    legacy_generator.N = 2 * spec["n_group"]
    group, x, u, legacy_event, pe, py = legacy_generator.generate(rng, scene)
    tau = 3.0
    uncapped_time = u.astype(float) * DT
    observed = np.minimum(uncapped_time, tau)
    event = (legacy_event.astype(bool) & (uncapped_time <= tau)).astype(np.int8)

    truth_exact = 0.0
    truth_grid = 0.0
    groups, diagnostics = [], []
    for g in (0, 1):
        ids = np.flatnonzero(group == g)
        truth_exact += (2*g-1) * km_rmst(observed[ids], event[ids], np.ones(len(ids)), tau)
        truth_grid += (2*g-1) * legacy_grid_mean(u[ids], legacy_event[ids])
        phi = km_if(observed[ids], event[ids], tau)
        partitions = {}
        for method in spec["methods"]:
            if method == PROXY_METHOD:
                labels, allocation, info = proxy_partition(x[ids], pe[ids], py[ids], spec["m_group"])
            else:
                labels, allocation, info = make_design(x[ids], spec["m_group"], method)
            info.update(scene=scene, cohort=cohort, group=g,
                        oracle_if_variance=design_variance(phi, labels, allocation),
                        event_fraction=float(np.mean(event[ids])),
                        censor_before_tau=float(np.mean((event[ids] == 0) & (observed[ids] < tau))),
                        risk_at_tau=float(np.mean(observed[ids] >= tau)))
            diagnostics.append(info)
            partitions[method] = labels, allocation
        groups.append((ids, partitions))

    for record in diagnostics:
        record.update(truth=truth_exact, legacy_grid_truth=truth_grid,
                      target_difference=truth_exact-truth_grid)
    records = []
    for draw in range(spec["draws"]):
        priorities = [np.random.default_rng(stable_seed(spec["seed"], scene, cohort, draw, g,
                                                        "legacy-sample")).random(spec["n_group"])
                      for g in (0, 1)]
        for method in spec["methods"]:
            estimate = 0.0
            boot_theta = np.zeros(spec["B"]) if spec["B"] else None
            for g, (ids, partitions) in enumerate(groups):
                labels, allocation = partitions[method]
                srng = np.random.default_rng(stable_seed(spec["seed"], scene, cohort, draw, g, "unused"))
                sampled, weights, meta = draw_sample(labels, allocation, srng, priorities=priorities[g])
                stime, sevent = observed[ids[sampled]], event[ids[sampled]]
                estimate += (2*g-1) * km_rmst(stime, sevent, weights, tau)
                if spec["B"]:
                    brng = np.random.default_rng(stable_seed(spec["seed"], scene, cohort, draw, g,
                                                            method, "legacy-bootstrap"))
                    boot_theta += (2*g-1) * bootstrap_rmst(stime, sevent, weights, meta, brng,
                                                          spec["B"], tau)
            record = dict(scene=scene, cohort=cohort, draw=draw, method=method,
                          truth=truth_exact, legacy_grid_truth=truth_grid,
                          target_difference=truth_exact-truth_grid,
                          estimate=estimate, error=estimate-truth_exact)
            if spec["B"]:
                delta = boot_theta-estimate
                lo, hi = np.quantile(delta, [.025, .975])
                lower, upper = estimate-hi, estimate-lo
                se = float(np.std(boot_theta, ddof=1))
                record.update(coverage=int(lower <= truth_exact <= upper), width=upper-lower,
                              lower=lower, upper=upper, boot_variance=se**2,
                              normal_coverage=int(abs(estimate-truth_exact) <= norm.ppf(.975)*se),
                              normal_width=2*norm.ppf(.975)*se,
                              bootstrap_bias=float(delta.mean()))
            records.append(record)
    return records, diagnostics


def oracle_summary(diagnostics, out):
    # Sum group variances within a cohort before evaluating cohort MC errors.
    cohort = diagnostics.groupby(["scene", "variant", "cohort"], as_index=False).agg(
        oracle_variance=("oracle_if_variance", "sum"),
        mean_q=("q", "mean"), mean_H=("H", "mean"),
        design_seconds=("seconds", "sum"),
        target_difference=("target_difference", "first"))
    summary = cohort.groupby(["scene", "variant"], as_index=False).agg(
        oracle_variance=("oracle_variance", "mean"),
        oracle_mcse=("oracle_variance", lambda s: float(s.std(ddof=1)/np.sqrt(len(s)))),
        cohorts=("cohort", "size"), mean_q=("mean_q", "mean"),
        mean_H=("mean_H", "mean"), mean_seconds=("design_seconds", "mean"),
        target_difference=("target_difference", "mean"))
    cohort.to_csv(out / "oracle_cohort_metrics.csv", index=False)
    summary.to_csv(out / "oracle_summary.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bud", default="BUD-inverse2")
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument("--n-group", type=int, default=960)
    parser.add_argument("--m-group", type=int, default=120)
    parser.add_argument("--cohorts", type=int, default=24)
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--B", type=int, default=199)
    parser.add_argument("--seed", type=int, default=8114021)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", default="results/historical")
    args = parser.parse_args()
    if args.workers > 2:
        parser.error("This continuity check is limited to at most two workers")
    if args.cohorts < 2 or args.draws < 0 or args.B < 0:
        parser.error("Require at least two cohorts and nonnegative draws/B")
    spec = vars(args).copy()
    spec["methods"] = ["SRS", "PC1-X", PROXY_METHOD, "KMEANS-X", args.bud]
    spec["notes"] = {
        "status": "Locked historical continuity check after development-only choice of BUD-inverse2",
        "target": "Finite-cohort exact KM RMST(group1)-RMST(group0), tau=3.0",
        "time_mapping": "time=min(u*DT,3.0), event=legacy_event & (u*DT<=3.0); DT=0.125",
        "difference_from_old": "Exact KM step integral replaces the old sum_j S(j*DT)*DT; both targets saved",
        "proxy_comparator": "MVRS-style PC1 of legacy predict_if(pe,py) for t>=0.5; H=m/2, two per stratum",
        "information": "Legacy pe/py derive from DGP conditional hazards; MVRS-PROXY has extra information, not a same-info comparison",
        "literature": "Historical generators are project synthetic experiments, not published DDS/MVRS simulation replications",
        "inference": "Independent SRSWOR per fixed cell; positive Rao-Wu rescaled within-cell bootstrap; basic 95% CI",
        "pairing": "Independent unit priorities shared across methods within each cohort/draw/group",
        "monte_carlo_cluster": "Complete cohort; samples nested within cohort are not independent cohort replicates",
    }
    sources = ("historical_check.py", "designs.py", "survival.py", "run_experiments.py",
               "legacy/run_study.py", "legacy/formal_vo_dds.py")
    spec["source_sha256"] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    protocol = out / "protocol.json"
    if protocol.exists():
        if json.loads(protocol.read_text()) != spec:
            raise RuntimeError("Existing historical protocol differs; choose a new output directory")
    else:
        # This happens before generating the first cohort or reading any result.
        protocol.write_text(json.dumps(spec, indent=2))
    tasks = [(scene, cohort, spec) for scene in args.scenes for cohort in range(args.cohorts)]
    allraw, alldiag = [], []
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one_historical_cohort, task): task[:2] for task in tasks}
        for done, future in enumerate(as_completed(futures), 1):
            scene, cohort = futures[future]
            records, diagnostics = future.result()
            allraw.extend(records)
            alldiag.extend(diagnostics)
            if records:
                pd.DataFrame(records).to_csv(out/f"raw_{scene}_{cohort:03d}.csv", index=False)
            pd.DataFrame(diagnostics).to_csv(out/f"diag_{scene}_{cohort:03d}.csv", index=False)
            print(f"historical {done}/{len(tasks)} {scene} cohort={cohort} "
                  f"elapsed={perf_counter()-start:.1f}s", flush=True)
    diagnostics = pd.DataFrame(alldiag).sort_values(["scene", "cohort", "group", "variant"])
    diagnostics.to_csv(out / "diagnostics.csv", index=False)
    oracle = oracle_summary(diagnostics, out)
    if allraw:
        raw = pd.DataFrame(allraw).sort_values(["scene", "cohort", "draw", "method"])
        raw.to_csv(out / "raw.csv", index=False)
        result = summarize(raw, diagnostics, out)
        print(result.round(5).to_string(index=False), flush=True)
    else:
        print(oracle.round(6).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
