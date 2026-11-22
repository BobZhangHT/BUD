#!/usr/bin/env python3
"""Independent direct KM integration and first-draw probability-design checks."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/real_core'))
from survival import km_rmst,verification

def direct_integral(time,event,weights,tau):
    survival,area,previous=1.,0.,0.
    for instant in sorted(set(time[(event==1)&(time<tau)])):
        area+=survival*(instant-previous)
        survival*=1-weights[(time==instant)&(event==1)].sum()/weights[time>=instant].sum()
        previous=instant
    return area+survival*(tau-previous)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('results/real_application'),
                        help='Result folder relative to the project root (or an absolute path).')
    args=parser.parse_args()
    results=ROOT/args.results
    protocol=json.loads((results/'protocol.json').read_text())
    tau=float(protocol['tau'])
    checks={'core_identity_checks':verification(),'independent_full_KM':[],'sampling_capacity_checks':[]}
    for name,time_column,event_column in [('nwtco','edrel','rel'),('flchain','futime','death')]:
        frame=pd.read_csv(ROOT/'data'/f'{name}.csv')
        time=frame[time_column].to_numpy();event=frame[event_column].to_numpy()
        a=km_rmst(time,event,np.ones(len(time)),tau)
        b=direct_integral(time,event,np.ones(len(time)),tau)
        assert abs(a-b)<1e-8
        checks['independent_full_KM'].append(dict(dataset=name,vectorized=a,direct_event_loop=b,absolute_difference=abs(a-b)))
        for folder in sorted(results.glob(f'{name}_m*')):
            first=pd.read_csv(folder/'first_draw_selected.csv')
            partition=pd.read_csv(folder/'partition_labels.csv')
            m=int(folder.name.split('_m')[1])
            for method in ['SRS','PC1-X','BUD']:
                selected=first[first.method==method]
                indices=selected.source_row.to_numpy(int)-1
                weights=selected.weight.to_numpy()
                assert len(indices)==m and len(np.unique(indices))==m
                a=km_rmst(time[indices],event[indices],weights,tau)
                b=direct_integral(time[indices],event[indices],weights,tau)
                assert abs(a-b)<1e-8
                counts=partition[method].value_counts()
                if method!='SRS':
                    assert counts.max()-counts.min()<=1
                    assert (selected.stratum.value_counts()==2).all()
                assert abs(weights.sum()-len(frame))<1e-8
                checks['sampling_capacity_checks'].append(dict(dataset=name,m=m,method=method,selected_unique=len(indices),weighted_mass=weights.sum(),capacity_min=int(counts.min()),capacity_max=int(counts.max()),first_draw_KM_absolute_difference=abs(a-b)))
    path=results/'validation_checks.json'
    path.write_text(json.dumps(checks,indent=2))
    print('PASS: cohort integrals, sampled integrals, distinctness, weights, capacities, paired covariance and finite-difference influence scores')

if __name__=='__main__':main()
