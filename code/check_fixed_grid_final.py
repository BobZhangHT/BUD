#!/usr/bin/env python3
"""Supplemental fixed-J mechanism check; never used for choosing the method.

Observed times from dev_additive are rounded UP to the fixed grid
0.25, ..., 3.00; the observed cause indicator is retained.  The target is
KM RMST of this coarsened complete cohort.  This deliberately differs from
continuous-time primary experiments.  The exact integral is
0.25 * (1 + S(t_1) + ... + S(t_11)), not the right-endpoint rectangle sum.
"""
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'legacy_locked'))
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
import hashlib
import json
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.stats import t as student_t, norm
from designs import make_design, design_variance
from refinement import refined_design
from scenarios import generate
from survival import draw_sample, km_rmst, km_if, bootstrap_rmst

ROOT = Path(__file__).resolve().parent / 'legacy_locked'
PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / 'results' / 'fixed_grid_final_reproduction'
SEED = 0x78c404c9
SIZES = [(480,60),(960,120),(1920,240)]
COHORTS = 32
DRAWS = 20
B = 499
DT = 0.25
TAU = 3.0


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8], 'little')


def grid_functional(time_obs,event,weights,tau=TAU,dt=DT):
    """The explicit smooth fixed-J map used to check exact integration."""
    grid = np.arange(1, int(round(tau/dt))+1)*dt
    w=np.asarray(weights,dtype=float)
    y=(time_obs[:,None]>=grid[None,:]).T@w
    d=((time_obs[:,None]==grid[None,:])&event[:,None].astype(bool)).T@w
    a=np.divide(d,y,out=np.zeros_like(d),where=y>0)
    s=np.cumprod(1-a)
    return float(dt*(1+s[:-1].sum())), y/w.sum(), a


def protocol():
    return dict(purpose='Frozen final-BUD fixed-grid mechanism check, no method selection; specified before outcomes',
                scope='Fixed J=12 observed-time grid; exact coarsened-cohort KM RMST',
                scenario='dev_additive',method='BUD-fullspace-onepass',
                sizes=[dict(n_group=n,m_group=m,H=m//2,k=16,s=2) for n,m in SIZES],
                cohorts=COHORTS,draws_per_cohort=DRAWS,B=B,seed=SEED,workers=1,
                tau=TAU,dt=DT,
                coarsening='U_grid=min(tau,dt*ceil(U/dt)); preserve original observed event indicator',
                exact_integral='dt*(1+sum_{j=1}^{J-1} S(j*dt))',
                target='difference of complete-cohort coarsened-time KM RMST (group 1 minus 0)',
                primary_ci='basic rescaled bootstrap, 95%',
                secondary_ci='normal with bootstrap standard deviation, 95%',
                uncertainty='cohort-cluster Monte Carlo standard errors; all draws retained',
                source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
                               for f in ('refinement.py','survival.py','designs.py','scenarios.py')})


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('results/fixed_grid_final_reproduction'),
                        help='Output directory relative to the project root (or an absolute path); existing protocols are never overwritten.')
    args=parser.parse_args()
    OUT=PROJECT/args.out
    OUT.mkdir(parents=True,exist_ok=True)
    spec=protocol()
    p=OUT/'protocol.json'
    if p.exists():
        raise RuntimeError('Refusing to overwrite an existing experiment')
    p.write_text(json.dumps(spec,indent=2))
    all_rows=[]
    all_diag=[]
    start=time.perf_counter()
    for n,m in SIZES:
        for cohort in range(COHORTS):
            rng=np.random.default_rng(stable_seed(SEED,n,m,cohort,'cohort'))
            dat=generate('dev_additive',rng,n,proxy=False)
            observed=np.minimum(TAU,DT*np.ceil(dat['time']/DT))
            event=dat['event'].copy()
            groups=[]
            truth=0.
            continuous_truth=0.
            for g in (0,1):
                ids=np.flatnonzero(dat['group']==g)
                ug,dg=observed[ids],event[ids]
                true_g=km_rmst(ug,dg,np.ones(n),TAU)
                fixed,y,alpha=grid_functional(ug,dg,np.ones(n))
                if abs(fixed-true_g)>2e-12:
                    raise AssertionError('Explicit fixed-J map and exact KM integral disagree')
                truth+=(2*g-1)*true_g
                continuous_truth+=(2*g-1)*km_rmst(dat['time'][ids],dg,np.ones(n),TAU)
                labels,allocation,info=refined_design(dat['x'][ids],m,space='full')
                phi=km_if(ug,dg,TAU)
                info.update(n_group=n,m_group=m,cohort=cohort,group=g,
                            min_normalized_risk=float(y.min()),max_event_jump=float(alpha.max()),
                            oracle_if_variance=design_variance(phi,labels,allocation),
                            exact_fixed_grid_difference=abs(fixed-true_g))
                all_diag.append(info)
                groups.append((ug,dg,labels,allocation))
            rows=[]
            for draw in range(DRAWS):
                estimate=0.
                boot=np.zeros(B)
                min_sample_risk=1.
                for g,(ug,dg,labels,allocation) in enumerate(groups):
                    srng=np.random.default_rng(stable_seed(SEED,n,m,cohort,draw,g,'sample'))
                    ix,w,meta=draw_sample(labels,allocation,srng)
                    est_g=km_rmst(ug[ix],dg[ix],w,TAU)
                    fixed,ys,ahs=grid_functional(ug[ix],dg[ix],w)
                    if abs(fixed-est_g)>2e-12:
                        raise AssertionError('Sample KM integral differs from fixed-grid map')
                    min_sample_risk=min(min_sample_risk,float(ys.min()))
                    estimate+=(2*g-1)*est_g
                    brng=np.random.default_rng(stable_seed(SEED,n,m,cohort,draw,g,'bootstrap'))
                    boot+=(2*g-1)*bootstrap_rmst(ug[ix],dg[ix],w,meta,brng,B,TAU)
                delta=boot-estimate
                qlo,qhi=np.quantile(delta,[.025,.975])
                low,high=estimate-qhi,estimate-qlo
                sd=float(np.std(boot,ddof=1))
                rows.append(dict(n_group=n,m_group=m,H=m//2,k=16,s=2,cohort=cohort,draw=draw,
                                 truth=truth,continuous_truth=continuous_truth,
                                 estimate=estimate,error=estimate-truth,sq_error=(estimate-truth)**2,
                                 coverage=int(low<=truth<=high),width=high-low,
                                 lower=low,upper=high,boot_variance=sd**2,
                                 normal_coverage=int(abs(estimate-truth)<=norm.ppf(.975)*sd),
                                 normal_width=2*norm.ppf(.975)*sd,
                                 sample_min_normalized_risk=min_sample_risk,
                                 bootstrap_bias=float(delta.mean())))
            all_rows.extend(rows)
            pd.DataFrame(rows).to_csv(OUT/f'raw_N{n}_c{cohort:02d}.csv',index=False)
            print(f'fixed-grid Ngroup={n} cohort={cohort+1}/{COHORTS} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    raw=pd.DataFrame(all_rows)
    diag=pd.DataFrame(all_diag)
    raw.to_csv(OUT/'raw.csv',index=False)
    diag.to_csv(OUT/'diagnostics.csv',index=False)
    cols=['error','sq_error','coverage','width','boot_variance','normal_coverage','normal_width','bootstrap_bias']
    block=raw.groupby(['n_group','m_group','cohort'])[cols].mean().reset_index()
    block.to_csv(OUT/'cohort_metrics.csv',index=False)
    result=[]
    for (n,m),v in block.groupby(['n_group','m_group']):
        q=student_t.ppf(.975,len(v)-1)
        rec=dict(n_group=n,m_group=m,H=m//2,k=16,s=2,cohorts=COHORTS,draws=DRAWS,
                 repetitions=COHORTS*DRAWS,B=B,rmse=float(np.sqrt(v.sq_error.mean())),
                 bias=float(v.error.mean()),oracle_if_variance=float(2*diag[diag.n_group==n].oracle_if_variance.mean()),
                 min_cohort_normalized_risk=float(diag[diag.n_group==n].min_normalized_risk.min()),
                 max_cohort_event_jump=float(diag[diag.n_group==n].max_event_jump.max()),
                 min_sample_normalized_risk=float(raw[raw.n_group==n].sample_min_normalized_risk.min()))
        for col in cols[2:]:
            val=float(v[col].mean());se=float(v[col].std(ddof=1)/np.sqrt(len(v)))
            rec[col]=val;rec[col+'_mcse']=se
            if col.endswith('coverage'):
                rec[col+'_mc_lower']=max(0.,val-q*se)
                rec[col+'_mc_upper']=min(1.,val+q*se)
        rec['bootstrap_to_oracle_variance']=rec['boot_variance']/rec['oracle_if_variance']
        result.append(rec)
    summary=pd.DataFrame(result)
    summary.to_csv(OUT/'summary.csv',index=False)
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':main()
