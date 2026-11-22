#!/usr/bin/env python3
"""Locked, covariate-only anchor ablations with posterior IF-variance diagnostics.

This finite diagnostic does not fit, select, or modify the locked main method.
It creates no bootstrap samples. Complete-cohort IF scores are computed only
for evaluation; only x (and the historical phase-I pe/py for MVRS-PROXY) enter
partitions. Validation outputs and validation scenarios are not read.
"""
import os
for _variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_variable] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import qmc, t as student_t

from designs import (make_design, standardized_pca, glp_uniform_design,
                     inverse_quantile_anchors, balanced_assignment,
                     design_variance, _within_scatter)
from scenarios import DEVELOPMENT, generate as generate_new
from historical_check import (SCENES as LEGACY_SCENES, legacy_generator,
                              proxy_partition, PROXY_METHOD)
from legacy.run_study import DT
from run_experiments import stable_seed
from survival import km_if, km_rmst

ROOT = Path(__file__).resolve().parent
BASELINE = 'BUD-inverse2'
VARIANTS = ('BUD-inverse2', 'BUD-rank2', 'BUD-lloyd2', 'BUD-sliced2',
            'BUD-iid2', 'BUD-lhs2', 'BUD-inverse-allPC2')


def diagnostic_anchor_design(x, m, variant, seed):
    """Three prespecified ablations, with the same assignment and s=2."""
    start = perf_counter()
    z, scores, eigenvalues, q85 = standardized_pca(x)
    h = m // 2
    if m % 2 or len(x) % h:
        raise ValueError('Require even m and N divisible by H=m/2.')
    q = scores.shape[1] if variant == 'BUD-inverse-allPC2' else q85
    if variant == 'BUD-iid2':
        design = np.random.default_rng(seed).uniform(size=(h, q))
    elif variant == 'BUD-lhs2':
        design = qmc.LatinHypercube(d=q, seed=np.random.default_rng(seed)).random(h)
    elif variant == 'BUD-inverse-allPC2':
        design = glp_uniform_design(h, q)
    else:
        raise ValueError(variant)
    coordinates = scores[:, :q]
    anchors = inverse_quantile_anchors(coordinates, design)
    labels, cost = balanced_assignment(coordinates, anchors)
    allocation = np.full(h, 2, dtype=int)
    sizes = np.bincount(labels)
    info = dict(variant=variant, N=len(x), m=m, p=x.shape[1], q=q, q85=q85,
                H=h, min_population_cell=int(sizes.min()),
                max_population_cell=int(sizes.max()), min_sample_cell=2,
                max_sample_cell=2, assignment_cost=cost,
                retained_fraction=float(eigenvalues[:q].sum()/eigenvalues.sum()),
                within_full_x=_within_scatter(z, labels),
                within_retained_pc=_within_scatter(coordinates, labels),
                ud_mixture_discrepancy=float(qmc.discrepancy(design, method='MD')),
                seconds=perf_counter()-start)
    return labels, allocation, info


def generate_cohort(family, scene, cohort, spec):
    rng = np.random.default_rng(stable_seed(spec['seed'], family, scene, cohort,
                                           'anchor-ablation-cohort'))
    if family == 'legacy':
        legacy_generator.N = 2 * spec['n_group']
        group, x, u, legacy_event, pe, py = legacy_generator.generate(rng, scene)
        tau = 3.0
        uncapped = u.astype(float)*DT
        time = np.minimum(uncapped, tau)
        event = (legacy_event.astype(bool) & (uncapped <= tau)).astype(np.int8)
        return dict(x=x, group=group, time=time, event=event, tau=tau, pe=pe, py=py)
    return generate_new(scene, rng, spec['n_group'], proxy=False)


def one_cohort(task):
    family, scene, cohort, spec = task
    dat = generate_cohort(family, scene, cohort, spec)
    records = []
    for g in (0, 1):
        ids = np.flatnonzero(dat['group'] == g)
        x = dat['x'][ids]
        z, pc, eigenvalues, q85 = standardized_pca(x)
        parts = {}
        # The entire partition loop precedes evaluation of complete-cohort IFs.
        for method in spec['methods']:
            if method in ('BUD-iid2','BUD-lhs2','BUD-inverse-allPC2'):
                labels, allocation, info = diagnostic_anchor_design(
                    x, spec['m_group'], method,
                    stable_seed(spec['seed'], family, scene, cohort, g, method, 'anchors'))
            else:
                labels, allocation, info = make_design(x, spec['m_group'], method)
            parts[method] = (labels, allocation, info)
        if family == 'legacy':
            parts[PROXY_METHOD] = proxy_partition(x, dat['pe'][ids], dat['py'][ids],
                                                 spec['m_group'])
        # Posterior diagnostics only. No output from this line affects a design.
        phi = km_if(dat['time'][ids], dat['event'][ids], dat['tau'])
        truth = km_rmst(dat['time'][ids], dat['event'][ids], np.ones(len(ids)), dat['tau'])
        for method, (labels, allocation, info) in parts.items():
            info = dict(info)
            info.update(family=family, scene=scene, cohort=cohort, group=g,
                        q85=q85, oracle_if_variance=design_variance(phi, labels, allocation),
                        complete_if_variance=float(np.var(phi, ddof=1)),
                        within_full_x=_within_scatter(z,labels),
                        within_q85_pc=_within_scatter(pc[:,:q85],labels),
                        within_pc1=_within_scatter(pc[:,:1],labels),
                        retained_fraction85=float(eigenvalues[:q85].sum()/eigenvalues.sum()),
                        truth_group=truth,
                        event_fraction=float(np.mean(dat['event'][ids])),
                        risk_at_tau=float(np.mean(dat['time'][ids] >= dat['tau'])))
            records.append(info)
    return records


def summarize(diag, out):
    # Each complete cohort is the independent Monte Carlo unit. Arm variances
    # add for the independent two-group contrast; geometries are arm averages.
    cohorts = diag.groupby(['family','scene','cohort','variant'],as_index=False).agg(
        oracle_variance=('oracle_if_variance','sum'),
        mean_q=('q','mean'), mean_q85=('q85','mean'),
        within_full_x=('within_full_x','mean'),
        within_q85_pc=('within_q85_pc','mean'),
        within_pc1=('within_pc1','mean'),
        anchor_md=('ud_mixture_discrepancy','mean'),
        assignment_cost=('assignment_cost','mean'),
        seconds=('seconds','sum'))
    cohorts.to_csv(out/'cohort_metrics.csv',index=False)
    rows=[]
    for (family,scene,method),frame in cohorts.groupby(['family','scene','variant']):
        base=cohorts[(cohorts.family==family)&(cohorts.scene==scene)&
                     (cohorts.variant==BASELINE)]
        paired=frame.merge(base,on=['family','scene','cohort'],suffixes=('_m','_b'))
        diff=paired.oracle_variance_m-paired.oracle_variance_b
        c=len(frame);qt=student_t.ppf(.975,c-1)
        se=float(diff.std(ddof=1)/np.sqrt(c))
        row=dict(family=family,scene=scene,method=method,cohorts=c,
                 oracle_variance=float(frame.oracle_variance.mean()),
                 oracle_mcse=float(frame.oracle_variance.std(ddof=1)/np.sqrt(c)),
                 oracle_se_ratio_to_inverse=float(np.sqrt(frame.oracle_variance.mean()/base.oracle_variance.mean())),
                 delta_variance_to_inverse=float(diff.mean()),
                 paired_delta_lower=float(diff.mean()-qt*se),
                 paired_delta_upper=float(diff.mean()+qt*se),
                 q=float(frame.mean_q.mean()),q85=float(frame.mean_q85.mean()),
                 within_full_x=float(frame.within_full_x.mean()),
                 full_scatter_ratio_to_inverse=float(frame.within_full_x.mean()/base.within_full_x.mean()),
                 within_pc1=float(frame.within_pc1.mean()),
                 anchor_md=float(frame.anchor_md.mean()),
                 design_seconds=float(frame.seconds.mean()))
        proxy=cohorts[(cohorts.family==family)&(cohorts.scene==scene)&
                      (cohorts.variant==PROXY_METHOD)]
        if len(proxy):
            row['oracle_se_ratio_to_proxy']=float(np.sqrt(frame.oracle_variance.mean()/proxy.oracle_variance.mean()))
        rows.append(row)
    summary=pd.DataFrame(rows).sort_values(['family','scene','method'])
    summary.to_csv(out/'summary.csv',index=False)
    print(summary[['family','scene','method','oracle_se_ratio_to_inverse',
                   'oracle_se_ratio_to_proxy','full_scatter_ratio_to_inverse',
                   'paired_delta_lower','paired_delta_upper']].round(6).to_string(index=False),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n-group',type=int,default=960)
    p.add_argument('--m-group',type=int,default=120)
    p.add_argument('--cohorts',type=int,default=12)
    p.add_argument('--seed',type=int,default=631421)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--out',default='results/anchor_ablation')
    args=p.parse_args()
    if args.workers not in (1,2):p.error('Use at most two workers')
    spec=vars(args).copy()
    spec.update(families={'legacy':list(LEGACY_SCENES),'new-development':list(DEVELOPMENT)},
                methods=list(VARIANTS)+['SRS','PC1-X'],
                notes={
                    'purpose':'Post-development mechanism diagnostic, not main validation or a new selected method',
                    'main_method':'BUD-inverse2 remains unchanged; no main validation output is read',
                    'all_candidates':'Same H=m/2,s=2,q85, except explicit all-PC diagnostic; SRS and PC1-X are references',
                    'iid':'One independent H-point uniform anchor set per cohort and group; no anchor selection',
                    'lhs':'One independent random Latin hypercube per cohort/group; no anchor selection',
                    'allPC':'Same GLP MD selection and unwhitened-PC inverse quantiles, retaining all PCs',
                    'proxy':'Legacy-only predicted IF-process PC1 from phase-I pe/py; gold IF never used in a partition',
                    'metric':'Exact design variance of complete-cohort exact-KM RMST influence mean, two arms summed',
                    'target':'Exact step-function KM RMST at tau=3; this is an IF diagnostic, not RMSE or CI validation',
                    'monte_carlo':'12 independent finite cohorts per scene; no phase-II draws or bootstrap',
                    'randomization':'Every random anchor set frozen in its cohort; independent stable seeds per variant/group',
                    'prohibition':'No DGP coefficient changes and no performance-driven anchor/candidate selection'})
    sources=['anchor_ablation.py','designs.py','scenarios.py','survival.py','historical_check.py',
             'run_experiments.py','legacy/run_study.py','legacy/formal_vo_dds.py']
    spec['source_sha256']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources}
    out=ROOT/args.out
    out.mkdir(parents=True,exist_ok=True)
    protocol=out/'protocol.json'
    if protocol.exists():
        if json.loads(protocol.read_text())!=spec:
            raise RuntimeError('Existing protocol differs; no silent overwrite is allowed')
    else:
        # Frozen before any cohort generation or numerical result is available.
        protocol.write_text(json.dumps(spec,indent=2))
    tasks=[(family,scene,cohort,spec) for family,scenes in spec['families'].items()
           for scene in scenes for cohort in range(args.cohorts)]
    records=[];start=perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(one_cohort,task):task[:3] for task in tasks}
        for done,future in enumerate(as_completed(futures),1):
            family,scene,cohort=futures[future]
            rows=future.result();records.extend(rows)
            pd.DataFrame(rows).to_csv(out/f'diag_{family}_{scene}_{cohort:03d}.csv',index=False)
            print(f'anchor {done}/{len(tasks)} {family} {scene} cohort={cohort} '
                  f'elapsed={perf_counter()-start:.1f}s',flush=True)
    diag=pd.DataFrame(records).sort_values(['family','scene','cohort','group','variant'])
    diag.to_csv(out/'diagnostics.csv',index=False)
    summarize(diag,out)


if __name__=='__main__':
    main()
