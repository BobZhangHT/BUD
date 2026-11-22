#!/usr/bin/env python3
"""Mechanism ablation of the FINAL frozen one-pass design; no method selection.

Only the initial anchor configuration changes (GLP, iid Uniform or random LHS).
The same q85 PCA, inverse quantile transform, exact balanced assignment and
exactly one full-standardized-X centroid reassignment are used throughout.
No phase-II sampling or bootstrap is performed: scores are posterior oracles.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from time import perf_counter
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from designs import standardized_pca,balanced_assignment,_within_scatter,design_variance
from anchor_ablation import diagnostic_anchor_design,generate_cohort
from refinement import refined_design
from historical_check import SCENES as LEGACY_SCENES
from scenarios import VALIDATION
from run_experiments import stable_seed
from survival import km_if

ROOT=Path(__file__).resolve().parent
METHODS=('BUD-final-GLP','BUD-final-iid','BUD-final-LHS')


def final_anchor_design(x,m,anchor,seed):
    if anchor=='GLP':
        labels,allocation,info=refined_design(x,m,space='full')
        info=dict(info)
        info.update(variant='BUD-final-GLP',initial_anchor_md=info['ud_mixture_discrepancy'])
        return labels,allocation,info
    start=perf_counter()
    variant={'iid':'BUD-iid2','LHS':'BUD-lhs2'}[anchor]
    initial,allocation,info=diagnostic_anchor_design(x,m,variant,seed)
    z,pc,eigenvalues,q85=standardized_pca(x)
    h=len(allocation)
    centers=np.array([z[initial==j].mean(axis=0) for j in range(h)])
    labels,cost=balanced_assignment(z,centers)
    before,after=_within_scatter(z,initial),_within_scatter(z,labels)
    if after>before+1e-10*max(1.,before):
        raise RuntimeError('Centroid reassignment increased full-X SSE')
    info=dict(info)
    info.update(variant=f'BUD-final-{anchor}',q=z.shape[1],q85=q85,initial_q=q85,
                assignment_cost=cost,initial_anchor_md=info['ud_mixture_discrepancy'],
                full_before=before,full_after=after,within_full_x=after,
                within_retained_pc=_within_scatter(pc[:,:q85],labels),
                seconds=perf_counter()-start)
    return labels,allocation,info


def one_cohort(task):
    family,scene,cohort,spec=task
    data=generate_cohort(family,scene,cohort,spec)
    rows=[]
    for g in (0,1):
        ids=np.flatnonzero(data['group']==g);x=data['x'][ids]
        parts={}
        for anchor in ('GLP','iid','LHS'):
            seed=stable_seed(spec['seed'],family,scene,cohort,g,anchor,'final-anchors')
            parts[anchor]=final_anchor_design(x,spec['m_group'],anchor,seed)
        # Outcomes and oracle IFs are evaluated after all three partitions exist.
        phi=km_if(data['time'][ids],data['event'][ids],data['tau'])
        for anchor,(labels,allocation,info) in parts.items():
            record=dict(info)
            record.update(family=family,scene=scene,cohort=cohort,group=g,
                          oracle_if_variance=design_variance(phi,labels,allocation))
            rows.append(record)
    return rows


def summarize(diag,out):
    cohorts=diag.groupby(['family','scene','cohort','variant'],as_index=False).agg(
        oracle_variance=('oracle_if_variance','sum'),
        full_before=('full_before','mean'),full_after=('full_after','mean'),
        initial_md=('initial_anchor_md','mean'),mean_q85=('q85','mean'),seconds=('seconds','sum'))
    cohorts.to_csv(out/'cohort_metrics.csv',index=False)
    rows=[]
    for (family,scene,method),frame in cohorts.groupby(['family','scene','variant']):
        base=cohorts[(cohorts.family==family)&(cohorts.scene==scene)&(cohorts.variant=='BUD-final-GLP')]
        pair=frame.merge(base,on=['family','scene','cohort'],suffixes=('_m','_b'))
        a,b=pair.oracle_variance_m,pair.oracle_variance_b
        delta=a-b;n=len(delta);crit=student_t.ppf(.975,n-1)
        delta_se=float(delta.std(ddof=1)/np.sqrt(n))
        ratio=float(np.sqrt(a.mean()/b.mean()))
        linearized=(ratio/2)*(a/a.mean()-b/b.mean())
        ratio_se=float(linearized.std(ddof=1)/np.sqrt(n))
        rows.append(dict(family=family,scene=scene,method=method,cohorts=n,
                         oracle_variance=float(a.mean()),oracle_se_ratio_to_GLP=ratio,
                         ratio_mcse_delta=ratio_se,ratio_lower_delta=ratio-crit*ratio_se,
                         ratio_upper_delta=ratio+crit*ratio_se,
                         delta_variance_to_GLP=float(delta.mean()),
                         paired_delta_lower=float(delta.mean()-crit*delta_se),
                         paired_delta_upper=float(delta.mean()+crit*delta_se),
                         initial_scatter_ratio_to_GLP=float(frame.full_before.mean()/base.full_before.mean()),
                         final_scatter_ratio_to_GLP=float(frame.full_after.mean()/base.full_after.mean()),
                         initial_md=float(frame.initial_md.mean()),q85=float(frame.mean_q85.mean()),
                         seconds=float(frame.seconds.mean())))
    summary=pd.DataFrame(rows).sort_values(['family','scene','method'])
    summary.to_csv(out/'summary.csv',index=False)
    aggregate=summary.groupby('method',as_index=False).agg(
        geomean_se_ratio=('oracle_se_ratio_to_GLP',lambda v:float(np.exp(np.log(v).mean()))),
        min_se_ratio=('oracle_se_ratio_to_GLP','min'),max_se_ratio=('oracle_se_ratio_to_GLP','max'),
        mean_initial_scatter_ratio=('initial_scatter_ratio_to_GLP','mean'),
        mean_final_scatter_ratio=('final_scatter_ratio_to_GLP','mean'))
    aggregate.to_csv(out/'aggregate.csv',index=False)
    print(summary[['family','scene','method','oracle_se_ratio_to_GLP','ratio_lower_delta',
                   'ratio_upper_delta','initial_scatter_ratio_to_GLP','final_scatter_ratio_to_GLP']]
          .round(6).to_string(index=False),flush=True)
    print(aggregate.round(6).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=975314)
    parser.add_argument('--n-group',type=int,default=960)
    parser.add_argument('--m-group',type=int,default=120)
    parser.add_argument('--cohorts',type=int,default=12)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--out',default='results/final_anchor_ablation')
    args=parser.parse_args()
    if args.workers not in (1,2):parser.error('Use at most two workers')
    spec=vars(args).copy()
    spec.update(families={'legacy':list(LEGACY_SCENES),'new':list(VALIDATION)},methods=list(METHODS),
                notes={
                    'status':'Post-lock mechanism ablation, no candidate changes permitted',
                    'purpose':'Assess incremental initial-UD anchor effect after the frozen full-X correction',
                    'shared_rules':'PCA q85, marginal inverse quantile, exact equal-capacity assignment, one full-X centroid reassignment, H=m/2 and s=2',
                    'anchor_randomness':'One iid/LHS set per independent cohort and arm; no discrepancy/outcome selection of random sets',
                    'glp':'Exactly the frozen refined_design implementation, not a rewritten approximation',
                    'outcome':'Posterior exact complete-cohort KM-RMST IF design variance; no RMSE, coverage or bootstrap',
                    'cohort_unit':'Independent complete cohort; two group variances summed for contrast',
                    'interpretation':'No claim of new DGP validation; intervals are descriptive Monte Carlo intervals, not multiplicity adjusted',
                    'prohibition':'No result-driven DGP or algorithm modifications'})
    sources=['final_anchor_ablation.py','refinement.py','anchor_ablation.py','designs.py','scenarios.py',
             'historical_check.py','survival.py','run_experiments.py','legacy/formal_vo_dds.py','legacy/run_study.py']
    spec['source_sha256']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources}
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    protocol=out/'protocol.json'
    if protocol.exists():
        if json.loads(protocol.read_text())!=spec:raise RuntimeError('Existing protocol differs')
    else:
        protocol.write_text(json.dumps(spec,indent=2))
    tasks=[(f,s,c,spec) for f,scenes in spec['families'].items() for s in scenes for c in range(spec['cohorts'])]
    rows=[];start=perf_counter()
    with ProcessPoolExecutor(max_workers=spec['workers']) as pool:
        futures={pool.submit(one_cohort,t):t[:3] for t in tasks}
        for done,future in enumerate(as_completed(futures),1):
            family,scene,cohort=futures[future];records=future.result();rows.extend(records)
            pd.DataFrame(records).to_csv(out/f'diag_{family}_{scene}_{cohort:03d}.csv',index=False)
            print(f'final-anchor {done}/{len(tasks)} {family} {scene} cohort={cohort} '
                  f'elapsed={perf_counter()-start:.1f}s',flush=True)
    diag=pd.DataFrame(rows).sort_values(['family','scene','cohort','group','variant'])
    diag.to_csv(out/'diagnostics.csv',index=False)
    summarize(diag,out)


if __name__=='__main__':
    main()
