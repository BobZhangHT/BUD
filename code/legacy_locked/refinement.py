#!/usr/bin/env python3
"""Second development stage: one full-covariate centroid reassignment.

The candidate was motivated by the first validation's omitted-PC risk.
These prespecified scenarios are reused for method refinement, not represented
as fresh external validation. No outcome enters a partition. The locked
BUD-inverse2 implementation is imported without modification.
"""
import os
for _variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_variable] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from designs import make_design, standardized_pca, balanced_assignment, _within_scatter, design_variance
from anchor_ablation import diagnostic_anchor_design, generate_cohort
from scenarios import VALIDATION
from historical_check import SCENES as LEGACY_SCENES, proxy_partition, PROXY_METHOD
from run_experiments import stable_seed
from survival import km_if, km_rmst

ROOT=Path(__file__).resolve().parent
REFINED='BUD-fullspace-onepass'
BASE='BUD-inverse2'


def refined_design(x,m,space='full'):
    """One exact balanced reassignment to centroids of the inverse2 cells.

    space='full' uses every standardized raw-X coordinate. This guarantees
    full-X within-cell SSE cannot increase relative to inverse2, regardless of
    which PCs the initial PCA retained. space='retained' is a diagnostic
    equivalent of the existing retained-PC one-pass variant.
    """
    if space not in ('full','retained'):
        raise ValueError("space must be 'full' or 'retained'")
    start=perf_counter()
    initial,allocation,info=make_design(x,m,'BUD-inverse2')
    z,pc,eigenvalues,q85=standardized_pca(x)
    coords=z if space=='full' else pc[:,:q85]
    h=len(allocation)
    centroids=np.array([coords[initial==j].mean(axis=0) for j in range(h)])
    labels,cost=balanced_assignment(coords,centroids)
    before=_within_scatter(coords,initial)
    after=_within_scatter(coords,labels)
    if after > before+1e-10*max(1.0,before):
        raise RuntimeError('Exact centroid reassignment violated its SSE inequality')
    result=dict(info)
    result.update(variant=REFINED if space=='full' else 'BUD-retained-onepass',
                  q=coords.shape[1],q85=q85,initial_q=q85,
                  refinement_space=space,assignment_cost=cost,
                  refinement_before=before,refinement_after=after,
                  full_before=_within_scatter(z,initial),full_after=_within_scatter(z,labels),
                  within_full_x=_within_scatter(z,labels),
                  within_retained_pc=_within_scatter(pc[:,:q85],labels),
                  moved_fraction=float(np.mean(labels!=initial)),
                  seconds=perf_counter()-start)
    return labels,allocation,result


def one_cohort(task):
    family,scene,cohort,spec=task
    # generate_cohort consumes the fixed seed and unaltered scenario parameters.
    data=generate_cohort(family,scene,cohort,spec)
    rows=[]
    for g in (0,1):
        ids=np.flatnonzero(data['group']==g)
        x=data['x'][ids]
        z,pc,eig,q85=standardized_pca(x)
        parts={}
        for method in spec['methods']:
            if method==REFINED:
                parts[method]=refined_design(x,spec['m_group'],space='full')
            elif method=='BUD-inverse-allPC2':
                parts[method]=diagnostic_anchor_design(x,spec['m_group'],method,
                    stable_seed(spec['seed'],family,scene,cohort,g,method,'anchor'))
            else:
                parts[method]=make_design(x,spec['m_group'],method)
        if family=='legacy':
            parts[PROXY_METHOD]=proxy_partition(x,data['pe'][ids],data['py'][ids],spec['m_group'])
        # Gold IF scores are posterior diagnostics and cannot affect partitions.
        phi=km_if(data['time'][ids],data['event'][ids],data['tau'])
        truth=km_rmst(data['time'][ids],data['event'][ids],np.ones(len(ids)),data['tau'])
        for method,(labels,allocation,info) in parts.items():
            record=dict(info)
            record.update(family=family,scene=scene,cohort=cohort,group=g,q85=q85,
                          oracle_if_variance=design_variance(phi,labels,allocation),
                          within_full_x=_within_scatter(z,labels),
                          within_q85_pc=_within_scatter(pc[:,:q85],labels),
                          within_pc1=_within_scatter(pc[:,:1],labels),
                          complete_if_variance=float(np.var(phi,ddof=1)),truth_group=truth,
                          event_fraction=float(np.mean(data['event'][ids])),
                          risk_at_tau=float(np.mean(data['time'][ids]>=data['tau'])))
            rows.append(record)
    return rows


def summarize(diag,out):
    cohorts=diag.groupby(['family','scene','cohort','variant'],as_index=False).agg(
        oracle_variance=('oracle_if_variance','sum'),mean_q=('q','mean'),mean_q85=('q85','mean'),
        within_full_x=('within_full_x','mean'),within_q85_pc=('within_q85_pc','mean'),
        within_pc1=('within_pc1','mean'),seconds=('seconds','sum'))
    cohorts.to_csv(out/'cohort_metrics.csv',index=False)
    rows=[]
    for (family,scene,method),frame in cohorts.groupby(['family','scene','variant']):
        scene_rows=cohorts[(cohorts.family==family)&(cohorts.scene==scene)]
        base=scene_rows[scene_rows.variant==BASE]
        pair=frame.merge(base,on=['family','scene','cohort'],suffixes=('_m','_b'))
        delta=pair.oracle_variance_m-pair.oracle_variance_b
        n=len(frame); tcrit=student_t.ppf(.975,n-1)
        dse=float(delta.std(ddof=1)/np.sqrt(n))
        variance=float(frame.oracle_variance.mean())
        core=scene_rows[scene_rows.variant.isin(['SRS','PC1-X','BALANCED-KMEANS-X2'])]
        core_means=core.groupby('variant').oracle_variance.mean()
        allbench=scene_rows[scene_rows.variant.isin(['SRS','PC1-X','BALANCED-KMEANS-X2',PROXY_METHOD])]
        bench_means=allbench.groupby('variant').oracle_variance.mean()
        record=dict(family=family,scene=scene,method=method,cohorts=n,
                    oracle_variance=variance,
                    oracle_se_ratio_to_inverse=float(np.sqrt(variance/base.oracle_variance.mean())),
                    oracle_se_ratio_to_best_sameinfo=float(np.sqrt(variance/core_means.min())),
                    best_sameinfo_benchmark=str(core_means.idxmin()),
                    oracle_se_ratio_to_best_including_proxy=float(np.sqrt(variance/bench_means.min())),
                    best_including_proxy=str(bench_means.idxmin()),
                    delta_variance_to_inverse=float(delta.mean()),paired_delta_lower=float(delta.mean()-tcrit*dse),
                    paired_delta_upper=float(delta.mean()+tcrit*dse),
                    full_scatter_ratio_to_inverse=float(frame.within_full_x.mean()/base.within_full_x.mean()),
                    mean_q=float(frame.mean_q.mean()),mean_q85=float(frame.mean_q85.mean()),
                    seconds=float(frame.seconds.mean()))
        rows.append(record)
    summary=pd.DataFrame(rows).sort_values(['family','scene','method'])
    summary.to_csv(out/'summary.csv',index=False)
    worst=summary.groupby('method',as_index=False).agg(
        worst_se_ratio_to_sameinfo=('oracle_se_ratio_to_best_sameinfo','max'),
        worst_se_ratio_including_proxy=('oracle_se_ratio_to_best_including_proxy','max'),
        mean_se_ratio_to_inverse=('oracle_se_ratio_to_inverse','mean'),
        max_se_ratio_to_inverse=('oracle_se_ratio_to_inverse','max'),
        mean_geometry_ratio=('full_scatter_ratio_to_inverse','mean'))
    worst.to_csv(out/'worst_case_summary.csv',index=False)
    print(summary[summary.method.isin([BASE,REFINED,'BUD-lloyd2','BUD-inverse-allPC2'])]
          [['family','scene','method','oracle_se_ratio_to_inverse','oracle_se_ratio_to_best_sameinfo',
            'oracle_se_ratio_to_best_including_proxy','full_scatter_ratio_to_inverse']]
          .round(5).to_string(index=False),flush=True)
    print(worst.round(5).to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=421793)
    p.add_argument('--n-group',type=int,default=960)
    p.add_argument('--m-group',type=int,default=120)
    p.add_argument('--cohorts',type=int,default=12)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--out',default='results/development2')
    args=p.parse_args()
    if args.workers not in (1,2):p.error('At most two workers are permitted')
    spec=vars(args).copy()
    spec.update(families={'legacy':list(LEGACY_SCENES),'new-reused-validation':list(VALIDATION)},
                methods=[BASE,'BUD-lloyd2',REFINED,'BUD-inverse-allPC2','BALANCED-KMEANS-X2','PC1-X','SRS'],
                notes={
                   'status':'Development stage 2, explicitly informed by omitted-PC risk observed in validation stage 1',
                   'not_fresh_validation':'The eight original validation DGPs are reused for refinement; only seeds/cohorts are new',
                   'bounded_iteration':'One final prespecified geometry refinement; no further parameter or DGP changes',
                   'refinement':'Initial inverse2 q85 cells, full standardized-X centroids, exactly one exact balanced reassignment',
                   'guarantee':'Full-X within-cell SSE is nonincreasing; no claim of every-target IF variance dominance',
                   'comparator':'Existing lloyd2 reassigns in retained PC space; all-PC inverse is imported from anchor_ablation',
                   'information':'All candidate designs X-only; historical proxy comparator alone uses phase-I pe/py',
                   'target':'Posterior exact finite-cohort exact-KM RMST influence mean design variance; no sample RMSE/Bootstrap',
                   'benchmark':'SRS, PC1-X, balanced full-X Kmeans; legacy also reports extra-information MVRS-PROXY',
                   'gold_if':'Computed only after all partitions; never used for partition/anchor selection',
                   'independence':'12 independent complete cohorts per DGP, two arms summed as independent design variances',
                   'validation_requirement':'Any newly selected candidate needs separate final new-seed sampling and CI validation'})
    sources=['refinement.py','anchor_ablation.py','designs.py','scenarios.py','historical_check.py',
             'survival.py','run_experiments.py','legacy/formal_vo_dds.py','legacy/run_study.py']
    spec['source_sha256']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources}
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    protocol=out/'protocol.json'
    if protocol.exists():
        if json.loads(protocol.read_text())!=spec:
            raise RuntimeError('A different protocol already exists; no silent overwrite')
    else:
        # Written before any cohort generation or diagnostic result is read.
        protocol.write_text(json.dumps(spec,indent=2))
    tasks=[(family,scene,c,spec) for family,scenes in spec['families'].items()
           for scene in scenes for c in range(spec['cohorts'])]
    all_rows=[];start=perf_counter()
    with ProcessPoolExecutor(max_workers=spec['workers']) as pool:
        futures={pool.submit(one_cohort,t):t[:3] for t in tasks}
        for done,future in enumerate(as_completed(futures),1):
            family,scene,cohort=futures[future]
            records=future.result();all_rows.extend(records)
            pd.DataFrame(records).to_csv(out/f'diag_{family}_{scene}_{cohort:03d}.csv',index=False)
            print(f'development2 {done}/{len(tasks)} {family} {scene} cohort={cohort} '
                  f'elapsed={perf_counter()-start:.1f}s',flush=True)
    diag=pd.DataFrame(all_rows).sort_values(['family','scene','cohort','group','variant'])
    diag.to_csv(out/'diagnostics.csv',index=False)
    summarize(diag,out)


if __name__=='__main__':
    main()
