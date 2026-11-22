#!/usr/bin/env python3
"""Real fixed-cohort evaluation of locked BUD, PC1-X and SRS.

Run from any directory. The protocol is written before constructing partitions
or computing outcome estimates. No outcome is an input to build_partitions.
The analysis estimates design error relative to each complete cohort's KM-RMST.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import norm
sys.path.insert(0, str(Path(__file__).resolve().parent / 'real_core'))
from designs import standardized_pca, glp_uniform_design, inverse_quantile_anchors, _within_scatter
from survival import km_rmst, draw_sample, bootstrap_weights, verification

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('SRS', 'PC1-X', 'BUD')
SOURCE_HASHES = {
    'nwtco': '2484933ed730f05f73fbce6af476a205fb29fef1b7ad75cd966a1a835161fa7a',
    'flchain': 'a96bcc58addb4c127e5012c5974aec8c7daf66123ddb104f309a78f158daa563',
}

def seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'little')

def load_frame(name, base_seed):
    path = ROOT / 'data' / f'{name}.csv'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_HASHES[name]:
        raise ValueError(f'{name}: input hash differs from the recorded data source')
    frame = pd.read_csv(path)
    frame['_source_row'] = np.arange(1, len(frame) + 1)
    frame = frame.iloc[np.random.default_rng(seed(base_seed, name, 'row_order')).permutation(len(frame))].reset_index(drop=True)
    if name == 'nwtco':
        xframe = pd.concat([
            frame[['age']].astype(float),
            pd.get_dummies(frame[['instit', 'stage', 'study']].astype(str), dtype=float)
        ], axis=1)
        time, event = frame.edrel.to_numpy(float), frame.rel.to_numpy(int)
    elif name == 'flchain':
        xframe = frame[['age', 'sample.yr', 'mgus']].astype(float).copy()
        xframe['sex_M'] = (frame.sex == 'M').astype(float)
        for key in ('kappa', 'lambda', 'creatinine'):
            a = frame[key].astype(float)
            if a.isna().any():
                xframe[f'{key}_missing'] = a.isna().astype(float)
            xframe[f'log1p_{key}'] = np.log1p(a.fillna(a.median()))
        time, event = frame.futime.to_numpy(float), frame.death.to_numpy(int)
    else:
        raise ValueError(name)
    assert len(frame) == {'nwtco': 4028, 'flchain': 7874}[name]
    assert np.isfinite(xframe.to_numpy()).all() and np.isfinite(time).all()
    assert (time >= 0).all() and np.isin(event, (0, 1)).all()
    return frame, xframe, time, event, digest

def capacitated_assignment(coords, centers, capacities):
    """Exact quadratic assignment for specified, possibly unequal capacities."""
    d = cdist(coords, centers, metric='sqeuclidean')
    slots = np.repeat(np.arange(len(centers)), capacities)
    row, col = linear_sum_assignment(d[:, slots])
    labels = np.empty(len(coords), dtype=int)
    labels[row] = slots[col]
    assert np.array_equal(np.bincount(labels), capacities)
    return labels, float(d[np.arange(len(coords)), labels].mean())

def build_partitions(x, m, design_seed):
    """The entire function is measurable with respect to X and fixed seeds."""
    n = len(x)
    if m % 2 or not 4 <= m < n:
        raise ValueError('Require an even sampling budget between 4 and N-1')
    h = m // 2
    capacities = np.full(h, n // h, dtype=int)
    capacities[:n % h] += 1
    if capacities.min() <= 2:
        raise ValueError('Every population cell must contain at least 3 units')
    start = perf_counter()
    z, pc, eig, q = standardized_pca(x)
    pca_seconds = perf_counter() - start
    ties = np.random.default_rng(seed(design_seed, 'pc1_ties')).random(n)
    order = np.lexsort((ties, pc[:, 0]))
    pc1 = np.empty(n, dtype=int)
    pc1[order] = np.repeat(np.arange(h), capacities)
    pc1_seconds = perf_counter() - start
    before_bud = perf_counter()
    ud = glp_uniform_design(h, q)
    anchors = inverse_quantile_anchors(pc[:, :q], ud)
    initial, transport = capacitated_assignment(pc[:, :q], anchors, capacities)
    centers = np.vstack([z[initial == j].mean(axis=0) for j in range(h)])
    final, refined_cost = capacitated_assignment(z, centers, capacities)
    before, after = _within_scatter(z, initial), _within_scatter(z, final)
    assert after <= before + 1e-9 * max(1., before)
    diagnostics = dict(N=n, p=x.shape[1], m=m, H=h, q85=q,
                       capacity_min=int(capacities.min()), capacity_max=int(capacities.max()),
                       pca_seconds=pca_seconds, pc1_total_seconds=pc1_seconds,
                       bud_total_seconds=pca_seconds + perf_counter() - before_bud,
                       initial_transport=transport, refined_assignment_cost=refined_cost,
                       full_scatter_initial=before, full_scatter_final=after,
                       eigenvalues=eig.tolist(), pc1_scatter=_within_scatter(z, pc1))
    return {'SRS': (np.zeros(n, int), np.array([m])),
            'PC1-X': (pc1, np.full(h, 2)), 'BUD': (final, np.full(h, 2))}, diagnostics

def one_case(case):
    name, m, spec = case
    out = ROOT / spec['out'] / f'{name}_m{m}'
    out.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    frame, xframe, time, event, digest = load_frame(name, spec['seed'])
    parts, diag = build_partitions(xframe.to_numpy(), m, seed(spec['seed'], name, m, 'design'))
    diag['predictors'] = xframe.columns.tolist()
    (out / 'design_diagnostics.json').write_text(json.dumps(diag, indent=2))
    # Outcomes enter only after all partitions have been fixed.
    tau = spec['tau']
    full = km_rmst(time, event, np.ones(len(time)), tau)
    diag.update(full_rmst=full, tau=tau, deaths=int(event.sum()),
                risk_at_tau=int(np.sum(time >= tau)), input_sha256=digest)
    labels_frame = frame[['_source_row']].copy()
    for method, (lab, _) in parts.items(): labels_frame[method] = lab
    labels_frame.to_csv(out / 'partition_labels.csv', index=False)
    rows, selected_first = [], []
    for r in range(spec['replicates']):
        sampling_rng = np.random.default_rng(seed(spec['seed'], name, m, r, 'sample'))
        priorities = sampling_rng.random(len(frame))
        for method, (labels, allocation) in parts.items():
            idx, weights, meta = draw_sample(labels, allocation, sampling_rng, priorities=priorities)
            estimate = km_rmst(time[idx], event[idx], weights, tau)
            boot_rng = np.random.default_rng(seed(spec['seed'], name, m, r, method, 'bootstrap'))
            wstar = bootstrap_weights(weights, meta, boot_rng, spec['bootstrap_replicates'])
            boot = km_rmst(time[idx], event[idx], wstar, tau)
            qlo, qhi = np.quantile(boot - estimate, [.025, .975], method='linear')
            lo, hi = estimate - qhi, estimate - qlo
            se = float(boot.std(ddof=1))
            rows.append(dict(dataset=name, m=m, replicate=r, method=method,
                             full_rmst=full, estimate=estimate, error=estimate-full,
                             squared_error=(estimate-full)**2, lower=lo, upper=hi,
                             covered=int(lo <= full <= hi), interval_length=hi-lo,
                             bootstrap_se=se, boot_mean=float(boot.mean()),
                             sample_risk_at_tau=int(np.sum(time[idx] >= tau)),
                             tail_empty=int(np.sum(time[idx] >= tau) == 0)))
            if r == 0:
                for j, ix in enumerate(idx):
                    selected_first.append(dict(method=method, source_row=int(frame._source_row.iloc[ix]),
                                               stratum=int(labels[ix]), weight=float(weights[j])))
    raw = pd.DataFrame(rows)
    raw.to_csv(out / 'raw.csv', index=False)
    pd.DataFrame(selected_first).to_csv(out / 'first_draw_selected.csv', index=False)
    raw[raw.replicate == 0].to_csv(out / 'first_draw_estimates.csv', index=False)
    diag['total_seconds'] = perf_counter() - start
    (out / 'cohort_diagnostics.json').write_text(json.dumps(diag, indent=2))
    return name, m, diag['total_seconds']

def summarize(spec):
    out = ROOT / spec['out']
    raw = pd.concat([pd.read_csv(p) for p in sorted(out.glob('*_m*/raw.csv'))], ignore_index=True)
    raw.to_csv(out / 'real_raw.csv.gz', index=False)
    rows, pairs = [], []
    for (name, m, method), d in raw.groupby(['dataset', 'm', 'method']):
        r = len(d)
        rows.append(dict(dataset=name, N={'nwtco':4028,'flchain':7874}[name], m=m, method=method,
                         replicates=r, full_rmst=d.full_rmst.iloc[0], bias=d.error.mean(),
                         rmse=np.sqrt(d.squared_error.mean()), sd=d.estimate.std(ddof=1),
                         basic_coverage=d.covered.mean(), coverage_mcse=np.sqrt(d.covered.mean()*(1-d.covered.mean())/r),
                         mean_interval_length=d.interval_length.mean(),
                         mean_bootstrap_se=d.bootstrap_se.mean(),
                         rms_bootstrap_se=np.sqrt(np.mean(d.bootstrap_se**2)),
                         tail_empty_count=d.tail_empty.sum()))
    for (name, m), d in raw.groupby(['dataset', 'm']):
        w = d.pivot(index='replicate', columns='method', values='squared_error')
        rng = np.random.default_rng(seed(spec['seed'], name, m, 'paired_mc'))
        resampled = rng.integers(0, len(w), size=(4000, len(w)))
        for benchmark in ('SRS', 'PC1-X'):
            a, b = w.BUD.to_numpy(), w[benchmark].to_numpy()
            dif = a-b
            reduction = 100*(1-np.sqrt(a.mean()/b.mean()))
            bt = 100*(1-np.sqrt(a[resampled].mean(axis=1)/b[resampled].mean(axis=1)))
            qlo, qhi = np.quantile(bt, [.025,.975])
            se = dif.std(ddof=1)/np.sqrt(len(dif))
            pairs.append(dict(dataset=name, m=m, benchmark=benchmark,
                              rmse_reduction_percent=reduction, reduction_mc_low=qlo, reduction_mc_high=qhi,
                              mse_difference=dif.mean(), mse_difference_mcse=se,
                              mse_difference_mc_low=dif.mean()-1.96*se,
                              mse_difference_mc_high=dif.mean()+1.96*se,
                              paired_p=2*norm.sf(abs(dif.mean()/se))))
    summary = pd.DataFrame(rows)
    pair = pd.DataFrame(pairs)
    order = np.argsort(pair.paired_p.to_numpy())
    adjusted = np.minimum(1, np.maximum.accumulate(pair.paired_p.to_numpy()[order]*(len(pair)-np.arange(len(pair)))))
    pair['paired_p_holm'] = 1.
    pair.loc[order, 'paired_p_holm'] = adjusted
    summary.to_csv(out / 'real_summary.csv', index=False)
    pair.to_csv(out / 'real_paired.csv', index=False)
    print(summary.to_string(index=False), flush=True)
    print(pair.to_string(index=False), flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replicates', type=int, default=1000)
    p.add_argument('--bootstrap-replicates', type=int, default=499)
    p.add_argument('--seed', type=int, default=0x78c404c9)
    p.add_argument('--tau', type=float, default=1825.)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--out', default='results/real_application')
    args = p.parse_args()
    spec = vars(args).copy()
    spec.update(cases=[['nwtco',200],['nwtco',400],['flchain',400],['flchain',800]],
                target='Single-cohort full-data exact weighted KM RMST on [0,1825] days',
                method='Locked BUD85 inverse marginal quantile GLP32 MD + exactly one full-X refinement',
                capacities='First N mod H cells have ceil(N/H); other cells have floor(N/H); H=m/2',
                bootstrap='Positive finite-population corrected Rao-Wu, s-1 multinomial draws; paired for s=2',
                interval='Basic 95 percent, linear empirical quantiles, B fixed in computation',
                pairing='Common iid priorities per replicate across all three methods',
                preprocessing={'nwtco':'age numeric + all-level dummies instit,stage,study; excludes histol,in.subcohort,IDs',
                               'flchain':'age,sample.yr,mgus,sex_M; log1p kappa,lambda,creatinine; median creatinine plus missing flag; excludes flc.grp,chapter,IDs'},
                source_hashes=SOURCE_HASHES,
                source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                               for f in [Path(__file__),Path(__file__).parent/'real_core/designs.py',Path(__file__).parent/'real_core/survival.py']})
    out = ROOT / spec['out']; out.mkdir(parents=True, exist_ok=True)
    protocol = out / 'protocol.json'
    if protocol.exists():
        prior = json.loads(protocol.read_text()); prior.pop('created_utc',None)
        if prior != spec: raise RuntimeError('Protocol differs; use a new output directory')
    else:
        protocol.write_text(json.dumps({**spec,'created_utc':datetime.now(timezone.utc).isoformat()}, indent=2))
    verification()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one_case,(name,m,spec)) for name,m in spec['cases']]
        for future in as_completed(futures): print('COMPLETE',future.result(),flush=True)
    summarize(spec)

if __name__ == '__main__': main()
