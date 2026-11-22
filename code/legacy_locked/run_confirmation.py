#!/usr/bin/env python3
"""Independent Monte Carlo confirmation after development stage two.

Scenario families have been examined during development. New seeds and
cohorts protect against Monte Carlo selection noise, not against overfitting
the finite collection of scenario families. This distinction is explicit.
"""
import os
for var in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[var]='1'
import argparse,hashlib,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.stats import norm
from designs import make_design,design_variance
from refinement import refined_design
from scenarios import VALIDATION
from historical_check import SCENES,PROXY_METHOD,proxy_partition
from anchor_ablation import generate_cohort
from survival import km_rmst,km_if,draw_sample,bootstrap_rmst
from run_experiments import stable_seed,summarize

ROOT=Path(__file__).resolve().parent

def design_dispatch(x,m,method):
    if method=='BUD-fullspace-onepass':return refined_design(x,m,space='full')
    return make_design(x,m,method)

def worker(task):
    family,scene,cohort,spec=task
    dat=generate_cohort(family,scene,cohort,spec)
    methods=spec['methods']+([PROXY_METHOD] if family=='legacy' else [])
    true=0.0;groups=[];diags=[]
    for g in (0,1):
        ix=np.flatnonzero(dat['group']==g)
        t,d=dat['time'][ix],dat['event'][ix]
        true+=(2*g-1)*km_rmst(t,d,np.ones(len(ix)),dat['tau'])
        phi=km_if(t,d,dat['tau'])
        plans={}
        for method in methods:
            if method==PROXY_METHOD:
                lab,alloc,info=proxy_partition(dat['x'][ix],dat['pe'][ix],dat['py'][ix],spec['m_group'])
            else:lab,alloc,info=design_dispatch(dat['x'][ix],spec['m_group'],method)
            info.update(family=family,scene=scene,cohort=cohort,group=g,
                        oracle_if_variance=design_variance(phi,lab,alloc),
                        event_fraction=float(d.mean()),risk_at_tau=float(np.mean(t>=dat['tau'])))
            diags.append(info);plans[method]=(lab,alloc)
        groups.append((ix,plans))
    records=[]
    for draw in range(spec['draws']):
        priorities=[np.random.default_rng(stable_seed(spec['seed'],family,scene,cohort,draw,g,'sample')).random(spec['n_group']) for g in (0,1)]
        for method in methods:
            estimate=0.0;boot=np.zeros(spec['B']);risk=[]
            for g,(ix,plans) in enumerate(groups):
                lab,alloc=plans[method]
                sampled,w,meta=draw_sample(lab,alloc,np.random.default_rng(0),priorities=priorities[g])
                t,d=dat['time'][ix[sampled]],dat['event'][ix[sampled]]
                risk.append(int(np.sum(t>=dat['tau'])))
                estimate+=(2*g-1)*km_rmst(t,d,w,dat['tau'])
                brng=np.random.default_rng(stable_seed(spec['seed'],family,scene,cohort,draw,g,method,'bootstrap'))
                boot+=(2*g-1)*bootstrap_rmst(t,d,w,meta,brng,spec['B'],dat['tau'])
            delta=boot-estimate;lo,hi=np.quantile(delta,[.025,.975]);se=float(np.std(boot,ddof=1))
            record=dict(family=family,scene=scene,cohort=cohort,draw=draw,method=method,
                        truth=true,estimate=estimate,error=estimate-true,
                        coverage=int(estimate-hi<=true<=estimate-lo),width=hi-lo,
                        lower=estimate-hi,upper=estimate-lo,boot_variance=se**2,
                        normal_coverage=int(abs(estimate-true)<=norm.ppf(.975)*se),
                        normal_width=2*norm.ppf(.975)*se,bootstrap_bias=float(delta.mean()),
                        sample_risk_tau_min=min(risk),sample_risk_tau_0=risk[0],sample_risk_tau_1=risk[1])
            records.append(record)
    return records,diags

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cohorts',type=int,default=64);ap.add_argument('--draws',type=int,default=20)
    ap.add_argument('--B',type=int,default=499);ap.add_argument('--n-group',type=int,default=960)
    ap.add_argument('--m-group',type=int,default=120);ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--seed',type=int,default=33891027)
    ap.add_argument('--methods',nargs='+',default=['SRS','PC1-X','KMEANS-X','BALANCED-KMEANS-X2','BUD-inverse2','BUD-fullspace-onepass'])
    ap.add_argument('--family',choices=['all','new','legacy'],default='all')
    ap.add_argument('--out',default='results/confirmation')
    args=ap.parse_args();spec=vars(args).copy()
    tasks=[]
    if args.family in ('all','new'):tasks += [('new-confirmation',s,c,spec) for s in VALIDATION for c in range(args.cohorts)]
    if args.family in ('all','legacy'):tasks += [('legacy',s,c,spec) for s in SCENES for c in range(args.cohorts)]
    spec['interpretation']='Independent new cohorts and Monte Carlo seeds after method refinement; the DGP families are reused, not unseen external validation.'
    spec['source_sha256']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['designs.py','refinement.py','scenarios.py','survival.py','anchor_ablation.py','run_confirmation.py']}
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise RuntimeError('Refuse to overwrite an existing confirmation experiment.')
    (out/'protocol.json').write_text(json.dumps(spec,indent=2))
    start=time.perf_counter();allr=[];alld=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(worker,t):t[:3] for t in tasks}
        for number,fut in enumerate(as_completed(futures),1):
            fam,scene,cohort=futures[fut];r,d=fut.result();allr+=r;alld+=d
            pd.DataFrame(r).to_csv(out/f'raw_{scene}_{cohort:03d}.csv',index=False)
            pd.DataFrame(d).to_csv(out/f'diag_{scene}_{cohort:03d}.csv',index=False)
            print(f'{number}/{len(tasks)} {fam} {scene} cohort={cohort} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    raw=pd.DataFrame(allr).sort_values(['family','scene','cohort','draw','method'])
    diag=pd.DataFrame(alld).sort_values(['family','scene','cohort','group','variant'])
    raw.to_csv(out/'raw.csv',index=False);diag.to_csv(out/'diagnostics.csv',index=False)
    summ=summarize(raw,diag,out)
    print(summ.round(5).to_string(index=False),flush=True)

if __name__=='__main__':main()
