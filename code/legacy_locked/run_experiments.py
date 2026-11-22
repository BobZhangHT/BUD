#!/usr/bin/env python3
"""Reproducible, paired design-conditional survival simulation.

Partitions depend on covariates only. Complete-cohort influence scores are
diagnostics computed after partitioning, never inputs to the sampling method.
Repeated samples are nested within independent complete cohorts. Uncertainty
summaries therefore retain the cohort as the Monte Carlo cluster.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse, hashlib, json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
from designs import make_design, design_variance, equal_quantile_labels
from survival import km_rmst, km_if, draw_sample, bootstrap_rmst
from scenarios import generate, DEVELOPMENT, VALIDATION, DESCRIPTIONS

ROOT = Path(__file__).resolve().parent


def stable_seed(*parts):
    blob = '|'.join(map(str,parts)).encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], 'little')


def one_cohort(task):
    scene, cohort, spec = task
    rng = np.random.default_rng(stable_seed(spec['seed'],scene,cohort,'cohort'))
    dat = generate(scene,rng,spec['n_group'])
    x, group, observed, event, tau = (dat[k] for k in ('x','group','time','event','tau'))
    groups = []
    true_theta = 0.0
    diagnostics = []
    for g in (0,1):
        ids = np.flatnonzero(group == g)
        truth = km_rmst(observed[ids],event[ids],np.ones(len(ids)),tau)
        true_theta += (2*g-1)*truth
        phi = km_if(observed[ids],event[ids],tau)
        parts = {}
        for meth in spec['methods']:
            labels, allocation, info = make_design(x[ids],spec['m_group'],meth)
            info.update(scene=scene,cohort=cohort,group=g,
                        oracle_if_variance=design_variance(phi,labels,allocation),
                        censor_before_tau=float(np.mean((event[ids]==0)&(observed[ids]<tau))),
                        event_fraction=float(np.mean(event[ids])),
                        risk_at_tau=float(np.mean(observed[ids]>=tau)))
            diagnostics.append(info)
            parts[meth] = (labels,allocation)
        groups.append((ids,parts))
    records = []
    for draw in range(spec['draws']):
        # The same random priority for a unit in every competing design gives
        # valid SRSWOR and a paired Monte Carlo comparison.
        priorities = [np.random.default_rng(stable_seed(spec['seed'],scene,cohort,draw,g,'sample'))
                      .uniform(size=spec['n_group']) for g in (0,1)]
        for meth in spec['methods']:
            estimate = 0.0
            boot_theta = np.zeros(spec['B']) if spec['B'] else None
            for g,(ids,parts) in enumerate(groups):
                labels,allocation = parts[meth]
                srng=np.random.default_rng(stable_seed(spec['seed'],scene,cohort,draw,g,'unused'))
                sampled,weights,meta = draw_sample(labels,allocation,srng,priorities=priorities[g])
                us,ds=observed[ids[sampled]],event[ids[sampled]]
                estimate += (2*g-1)*km_rmst(us,ds,weights,tau)
                if spec['B']:
                    brng=np.random.default_rng(stable_seed(spec['seed'],scene,cohort,draw,g,meth,'bootstrap'))
                    boot_theta += (2*g-1)*bootstrap_rmst(us,ds,weights,meta,brng,spec['B'],tau)
            record = dict(scene=scene,cohort=cohort,draw=draw,method=meth,
                          truth=true_theta,estimate=estimate,error=estimate-true_theta)
            if spec['B']:
                delta=boot_theta-estimate
                lower,upper=np.quantile(delta,[.025,.975])
                low,high=estimate-upper,estimate-lower
                se=float(np.std(boot_theta,ddof=1))
                record.update(coverage=int(low<=true_theta<=high),width=high-low,
                              lower=low,upper=high,boot_variance=se**2,
                              normal_coverage=int(abs(estimate-true_theta)<=norm.ppf(.975)*se),
                              normal_width=2*norm.ppf(.975)*se,
                              bootstrap_bias=float(np.mean(delta)))
            records.append(record)
    return records,diagnostics


def summarize(raw,diagnostics,out):
    raw['sq_error']=raw.error**2
    cols=['sq_error','error']
    if 'coverage' in raw:
        cols += ['coverage','width','boot_variance','normal_coverage','normal_width','bootstrap_bias']
    block=raw.groupby(['scene','method','cohort'])[cols].mean().reset_index()
    rows=[]
    for (scene,meth),v in block.groupby(['scene','method'],sort=False):
        mse=v.sq_error.mean();se=v.sq_error.std(ddof=1)/np.sqrt(len(v))
        q=student_t.ppf(.975,len(v)-1)
        rec=dict(scene=scene,method=meth,cohorts=len(v),draws_total=len(raw[(raw.scene==scene)&(raw.method==meth)]),
                 rmse=np.sqrt(mse),bias=v.error.mean(),mse=mse,mse_mcse=se,
                 rmse_lower=np.sqrt(max(0,mse-q*se)),rmse_upper=np.sqrt(max(0,mse+q*se)))
        for col in cols[2:]:
            rec[col]=v[col].mean()
            rec[col+'_mcse']=v[col].std(ddof=1)/np.sqrt(len(v))
        rows.append(rec)
    summary=pd.DataFrame(rows)
    dd=diagnostics.groupby(['scene','variant']).agg(oracle_variance=('oracle_if_variance','mean'),
                                                   mean_q=('q','mean'),mean_H=('H','mean'),
                                                   mean_seconds=('seconds','mean')).reset_index()
    # Group diagnostic variance was per arm; contrast variance is their sum.
    dd.oracle_variance*=2
    summary=summary.merge(dd,left_on=['scene','method'],right_on=['scene','variant']).drop(columns='variant')
    paired=[]
    for scene,v in block.groupby('scene'):
        tab=v.pivot(index='cohort',columns='method',values='sq_error')
        methods=list(tab.columns)
        for a in methods:
            for b in methods:
                if a==b:continue
                dif=tab[a]-tab[b]
                se=dif.std(ddof=1)/np.sqrt(len(dif));q=student_t.ppf(.975,len(dif)-1)
                paired.append(dict(scene=scene,method=a,baseline=b,delta_mse=dif.mean(),
                                   lower=dif.mean()-q*se,upper=dif.mean()+q*se,
                                   rmse_ratio=np.sqrt(tab[a].mean()/tab[b].mean())))
    summary.to_csv(out/'summary.csv',index=False)
    block.to_csv(out/'cohort_metrics.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'paired_mse.csv',index=False)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['development','validation'],default='development')
    parser.add_argument('--scenes',nargs='+')
    parser.add_argument('--methods',nargs='+',default=['SRS','PC1-X','PC1-X4','KMEANS-X',
                         'BUD-rank4','BUD-rank2','BUD-inverse4','BUD-inverse2','BUD-lloyd2','BUD-sliced2'])
    parser.add_argument('--n-group',type=int,default=640)
    parser.add_argument('--m-group',type=int,default=80)
    parser.add_argument('--cohorts',type=int,default=16)
    parser.add_argument('--draws',type=int,default=30)
    parser.add_argument('--B',type=int,default=0)
    parser.add_argument('--seed',type=int,default=421105)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--out',default='results/development')
    args=parser.parse_args()
    spec=vars(args).copy()
    scenes=args.scenes or (DEVELOPMENT if args.stage=='development' else VALIDATION)
    spec['scenes']=list(scenes)
    out=ROOT/args.out
    out.mkdir(parents=True,exist_ok=True)
    spec['source_sha256']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
                           for f in ['designs.py','scenarios.py','survival.py','run_experiments.py']}
    # Record the exact experiment before any result is inspected.
    specfile=out/'protocol.json'
    if specfile.exists():
        old=json.loads(specfile.read_text())
        if old!=spec:raise RuntimeError('An existing experiment may not be silently overwritten with a new protocol.')
    else:specfile.write_text(json.dumps(spec,indent=2))
    tasks=[(s,c,spec) for s in scenes for c in range(args.cohorts)]
    allraw,alldiag=[],[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs={pool.submit(one_cohort,task):task[:2] for task in tasks}
        for done,fut in enumerate(as_completed(futs),1):
            scene,cohort=futs[fut]
            rows,diag=fut.result()
            allraw.extend(rows);alldiag.extend(diag)
            pd.DataFrame(rows).to_csv(out/f'raw_{scene}_{cohort:03d}.csv',index=False)
            pd.DataFrame(diag).to_csv(out/f'diag_{scene}_{cohort:03d}.csv',index=False)
            print(f'{done}/{len(tasks)} {scene} cohort={cohort} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    raw=pd.DataFrame(allraw).sort_values(['scene','cohort','draw','method'])
    diagnostics=pd.DataFrame(alldiag).sort_values(['scene','cohort','group','variant'])
    raw.to_csv(out/'raw.csv',index=False)
    diagnostics.to_csv(out/'diagnostics.csv',index=False)
    result=summarize(raw,diagnostics,out)
    print(result.round(5).to_string(index=False),flush=True)


if __name__=='__main__':main()
