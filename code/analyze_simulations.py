#!/usr/bin/env python3
"""Reproduce the three-method manuscript analysis from archived raw results.

The independent Monte Carlo unit is a complete cohort, not an individual
phase-II draw. All draws from the same cohort stay together in each uncertainty calculation.
This is a post-lock reanalysis of a broader recorded experiment.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
METHODS=['SRS','PC1-X','BUD-fullspace-onepass']
LABELS={'SRS':'SRS','PC1-X':'First PC','BUD-fullspace-onepass':'BUD'}
BUD=METHODS[-1]
SCENES=['val_pc1','val_multivariate','val_banana','val_saddle','val_mixture',
        'val_nuisance','val_lowvar','val_null','banana','saddle3d','gaussian','multimodal']
SCENE_LABELS=['Leading factor','Several prognostic factors','Banana, 2D','Saddle, 6D',
              'Unequal mixture','Many nuisance variables','Low-variance signal','Null signal',
              'Curved trajectories','Saddle trajectories','Correlated trajectories','Mixture trajectories']
COLORS={'SRS':'#7B8490','PC1-X':'#D18424','BUD-fullspace-onepass':'#2166AC'}
COMPACT_LABELS=['Leading factor','Several factors','Banana (2D)','Saddle (6D)',
                'Unequal mixture','Nuisance variables','Low-variance signal','Null signal',
                'Curved, varying effect','Saddle, varying effect','Correlated, varying effect','Mixture, varying effect']


def write_grouped_table(summary, path):
    """Report the same three performance summaries for each comparator."""
    metrics = ['rmse', 'width', 'coverage']
    method_order = [BUD, 'PC1-X', 'SRS']
    lines = [
        r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrrrrr@{}}',
        r'\toprule',
        r'& \multicolumn{3}{c}{BUD} & \multicolumn{3}{c}{PC1-X} & \multicolumn{3}{c}{SRS} \\',
        r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}',
        r'Setting & RMSE & Width & Cov. & RMSE & Width & Cov. & RMSE & Width & Cov. \\',
        r'\midrule',
    ]
    labels = COMPACT_LABELS[:8] + ['Curved (TV)', 'Saddle (TV)', 'Correlated (TV)', 'Mixture (TV)']
    for scene, label in zip(SCENES, labels):
        values = summary[summary.scene == scene].set_index('method').loc[method_order, metrics]
        cells = [f'{rmse:.3f} & {width:.3f} & {coverage*100:.1f}'
                 for rmse, width, coverage in values.itertuples(index=False, name=None)]
        lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular*}'])
    path.write_text('\n'.join(lines))

def holm(p):
    p=np.asarray(p);order=np.argsort(p);q=np.empty(len(p));last=0.
    for rank,i in enumerate(order):
        last=max(last,min(1.,p[i]*(len(p)-rank)));q[i]=last
    return q

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirmation',type=Path,default=Path('results/original/confirmation'),
                        help='Confirmation input folder containing raw.csv[.gz] and diagnostics.csv[.gz].')
    parser.add_argument('--out',type=Path,default=Path('results'),help='Analysis output folder, relative to the project root or absolute.')
    parser.add_argument('--figures',type=Path,default=Path('figures'),help='Figure output folder, relative to the project root or absolute.')
    parser.add_argument('--cohorts', type=int, default=64, help='Expected independent cohorts per scenario.')
    parser.add_argument('--draws', type=int, default=20, help='Expected phase-II draws per cohort.')
    parser.add_argument('--planned-table', action='store_true',
                        help='Also write the grouped table of all three methods at the planned budget.')
    args=parser.parse_args()
    if args.cohorts < 2 or args.draws < 1:
        parser.error('At least two cohorts and one draw per cohort are required.')
    source=ROOT/args.confirmation
    def input_csv(stem):
        for suffix in ('.csv','.csv.gz'):
            path=source/(stem+suffix)
            if path.exists():return pd.read_csv(path)
        raise FileNotFoundError(f'Missing {stem}.csv or {stem}.csv.gz in {source}')
    out=ROOT/args.out;figs=ROOT/args.figures;out.mkdir(parents=True,exist_ok=True);figs.mkdir(parents=True,exist_ok=True)
    rawall=input_csv('raw')
    raw=rawall[rawall.method.isin(METHODS)].copy()
    raw['squared_error']=raw.error**2
    assert set(raw.scene) == set(SCENES)
    assert len(raw)==len(SCENES)*args.cohorts*args.draws*len(METHODS)
    assert not raw.duplicated(['scene','cohort','draw','method']).any()
    assert raw.groupby(['scene','method']).cohort.nunique().eq(args.cohorts).all()
    assert raw.groupby(['scene','cohort','method']).size().eq(args.draws).all()
    assert raw.groupby(['scene','cohort','draw']).method.nunique().eq(len(METHODS)).all()
    assert np.max(np.abs(raw.error-(raw.estimate-raw.truth)))<1e-12
    assert raw[['error','coverage','width','boot_variance']].notna().all().all()
    raw.sort_values(['scene','cohort','draw','method']).to_csv(out/'main_three_method_raw.csv.gz',index=False)
    blocks=raw.groupby(['scene','cohort','method'],as_index=False).agg(
        mse=('squared_error','mean'),bias=('error','mean'),coverage=('coverage','mean'),
        width=('width','mean'),boot_variance=('boot_variance','mean'),
        min_sample_risk=('sample_risk_tau_min','min'))
    blocks.to_csv(out/'cohort_metrics.csv',index=False)
    summary=[]
    for (scene,method),v in blocks.groupby(['scene','method']):
        q=student_t.ppf(.975,len(v)-1)
        d=dict(scene=scene,method=method,cohorts=len(v),draws_per_cohort=args.draws,
               repetitions=args.draws*len(v),rmse=float(np.sqrt(v.mse.mean())))
        for metric in ['mse','bias','coverage','width','boot_variance']:
            est=float(v[metric].mean());se=float(v[metric].std(ddof=1)/np.sqrt(len(v)))
            d.update({metric:est,metric+'_mcse':se,metric+'_lower':est-q*se,metric+'_upper':est+q*se})
        d['min_sample_risk']=int(v.min_sample_risk.min())
        summary.append(d)
    summary=pd.DataFrame(summary)
    paired=[];plot_ratios=[]
    rng=np.random.default_rng(622113)
    for scene in SCENES:
        v=blocks[blocks.scene==scene]
        pivot=v.pivot(index='cohort',columns='method',values='mse').sort_index()
        wpivot=v.pivot(index='cohort',columns='method',values='width').sort_index()
        resamples=rng.integers(0,len(pivot),size=(8000,len(pivot)))
        for benchmark in ['SRS','PC1-X']:
            d=pivot[BUD]-pivot[benchmark]
            se=float(d.std(ddof=1)/np.sqrt(len(d)));crit=student_t.ppf(.975,len(d)-1)
            ratio=float(np.sqrt(pivot[BUD].mean()/pivot[benchmark].mean()))
            ratios=np.sqrt(pivot[BUD].to_numpy()[resamples].mean(axis=1)/pivot[benchmark].to_numpy()[resamples].mean(axis=1))
            lo,hi=np.quantile(ratios,[.025,.975])
            paired.append(dict(scene=scene,benchmark=benchmark,rmse_ratio=ratio,ratio_lower=lo,ratio_upper=hi,
                relative_rmse_reduction=1-ratio,mse_difference=float(d.mean()),mse_difference_mcse=se,
                mse_difference_lower=float(d.mean()-crit*se),mse_difference_upper=float(d.mean()+crit*se),
                p_two_sided=float(2*student_t.sf(abs(d.mean()/se),len(d)-1)),
                ci_width_ratio=float(wpivot[BUD].mean()/wpivot[benchmark].mean())))
        for method in ['PC1-X',BUD]:
            for metric,piv,transform in [('rmse',pivot,np.sqrt),('width',wpivot,lambda z:z)]:
                ratio=float(transform(piv[method].mean()/piv['SRS'].mean()))
                r=transform(piv[method].to_numpy()[resamples].mean(axis=1)/piv['SRS'].to_numpy()[resamples].mean(axis=1))
                lo,hi=np.quantile(r,[.025,.975])
                plot_ratios.append(dict(scene=scene,method=method,metric=metric,ratio=ratio,lower=lo,upper=hi))
    paired=pd.DataFrame(paired)
    for b in ['SRS','PC1-X']:
        ix=paired.benchmark.eq(b);paired.loc[ix,'p_holm_12']=holm(paired.loc[ix,'p_two_sided'])
    paired.to_csv(out/'paired_evidence.csv',index=False)
    ratios=pd.DataFrame(plot_ratios);ratios.to_csv(out/'plot_ratios.csv',index=False)
    summary.to_csv(out/'simulation_summary.csv',index=False)
    pivot=summary.pivot(index='scene',columns='method',values='rmse')
    rmse_vs_pc1=pivot[BUD]/pivot['PC1-X']
    order=[s for s in SCENES]
    b=summary[summary.method==BUD].set_index('scene')
    headline=dict(number_scenarios=len(SCENES),independent_cohorts_per_scenario=args.cohorts,conditional_draws_per_cohort=args.draws,
        estimates_three_methods=len(raw),bud_point_wins=int(pivot.idxmin(axis=1).eq(BUD).sum()),
        bud_within_5pct_of_best=int((pivot[BUD]/pivot.min(axis=1)<=1.05).sum()),
        rmse_geometric_ratio_to_srs=float(np.exp(np.log(pivot[BUD]/pivot.SRS).mean())),
        rmse_geometric_ratio_to_pc1=float(np.exp(np.log(rmse_vs_pc1).mean())),
        width_geometric_ratio_to_srs=float(np.exp(np.log(paired[paired.benchmark=='SRS'].ci_width_ratio).mean())),
        coverage_range=[float(b.coverage.min()),float(b.coverage.max())],
        coverage_mean=float(b.coverage.mean()),
        significant_improvements_srs=int(((paired.benchmark=='SRS')&(paired.mse_difference<0)&(paired.p_holm_12<.05)).sum()),
        significant_improvements_pc1=int(((paired.benchmark=='PC1-X')&(paired.mse_difference<0)&(paired.p_holm_12<.05)).sum()),
        negative_pc1_scene=rmse_vs_pc1.idxmax() if rmse_vs_pc1.max()>1 else None,
        pc1_friendly_rmse_ratio=float(rmse_vs_pc1.loc['val_pc1']),
        all_draws_retained=True,cluster_unit='complete cohort',ratio_mc_replicates=8000,
        three_method_selection='Current comparison set requested after the broader experiment; historical archive retained.')
    (out/'simulation_headline.json').write_text(json.dumps(headline,indent=2))
    diag=input_csv('diagnostics')
    # Geometry is measured on both arms of each independent cohort.
    dg=diag[diag.variant==BUD].copy()
    assert len(dg)==len(SCENES)*args.cohorts*2 and (dg.full_after<=dg.full_before+1e-10).all()
    geometry=dg.groupby('scene',as_index=False).agg(q_initial=('q85','mean'),
        scatter_before=('full_before','mean'),scatter_after=('full_after','mean'),
        event_fraction=('event_fraction','mean'),risk_at_tau=('risk_at_tau','mean'),
        seconds_design_per_arm=('seconds','mean'))
    geometry['scatter_ratio']=geometry.scatter_after/geometry.scatter_before
    geometry.to_csv(out/'geometry_diagnostics.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(11.5,5.4),sharey=True,gridspec_kw={'width_ratios':[1.15,1,1]})
    y=np.arange(12)
    for j,(metric,title) in enumerate([('rmse','A  RMSE relative to SRS'),('coverage','B  95% interval coverage'),('width','C  Mean length relative to SRS')]):
        ax=axes[j]
        if metric=='coverage':
            ax.axvline(.95,color='#404040',lw=1,ls='--',zorder=0)
            for method,off in zip(METHODS,[-.20,0,.20]):
                v=summary[summary.method==method].set_index('scene').loc[order]
                ax.errorbar(v.coverage,y+off,xerr=np.vstack([v.coverage-v.coverage_lower,v.coverage_upper-v.coverage]),
                    fmt='o',ms=3.6,color=COLORS[method],elinewidth=.8,capsize=1.5,label=LABELS[method])
            ax.set_xlim(.905,.982);ax.set_xticks([.91,.93,.95,.97]);ax.set_xlabel('Coverage probability')
        else:
            ax.axvline(1,color='#404040',lw=1,ls='--',zorder=0)
            for method,off in [('PC1-X',-.11),(BUD,.11)]:
                v=ratios[(ratios.method==method)&(ratios.metric==metric)].set_index('scene').loc[order]
                ax.errorbar(v.ratio,y+off,xerr=np.vstack([v.ratio-v.lower,v.upper-v.ratio]),
                    fmt='o',ms=4,color=COLORS[method],elinewidth=1,capsize=2,label=LABELS[method])
            ax.set_xlim(.57,1.08);ax.set_xticks([.6,.8,1.]);ax.set_xlabel('Smaller indicates lower '+('error' if metric=='rmse' else 'length'))
        ax.axhline(7.5,color='#BCC1C7',lw=.8);ax.set_title(title,loc='left',fontsize=10,fontweight='bold',pad=10)
        ax.grid(axis='x',alpha=.18);ax.set_yticks(y);ax.set_ylim(11.6,-.6)
    axes[0].set_yticklabels(SCENE_LABELS)
    axes[1].legend(loc='lower center',bbox_to_anchor=(.5,-.24),ncol=3,frameon=False)
    fig.subplots_adjust(left=.21,right=.99,top=.93,bottom=.17,wspace=.14)
    fig.savefig(figs/'simulation_main.pdf',bbox_inches='tight');fig.savefig(figs/'simulation_main.png',dpi=220,bbox_inches='tight');plt.close(fig)
    # A native journal-width variant keeps all twelve settings and readable type.
    fig,axes=plt.subplots(1,3,figsize=(7.15,4.9),sharey=True,gridspec_kw={'width_ratios':[1,1,1]})
    for j,(metric,title) in enumerate([('rmse','A  RMSE / SRS'),('coverage','B  Coverage'),('width','C  Length / SRS')]):
        ax=axes[j]
        if metric=='coverage':
            ax.axvline(.95,color='#404040',lw=.8,ls='--',zorder=0)
            for method,off in zip(METHODS,[-.21,0,.21]):
                v=summary[summary.method==method].set_index('scene').loc[order]
                ax.errorbar(v.coverage,y+off,xerr=np.vstack([v.coverage-v.coverage_lower,v.coverage_upper-v.coverage]),
                    fmt='o',ms=2.8,color=COLORS[method],elinewidth=.7,capsize=1,label=LABELS[method])
            ax.set_xlim(.905,.982);ax.set_xticks([.91,.95,.98]);ax.set_xlabel('Nominal: 0.95',fontsize=9.4)
        else:
            ax.axvline(1,color='#404040',lw=.8,ls='--',zorder=0)
            for method,off in [('PC1-X',-.13),(BUD,.13)]:
                v=ratios[(ratios.method==method)&(ratios.metric==metric)].set_index('scene').loc[order]
                ax.errorbar(v.ratio,y+off,xerr=np.vstack([v.ratio-v.lower,v.upper-v.ratio]),
                    fmt='o',ms=3.1,color=COLORS[method],elinewidth=.8,capsize=1.3,label=LABELS[method])
            ax.set_xlim(.57,1.08);ax.set_xticks([.6,.8,1.]);ax.set_xlabel('Lower is better',fontsize=9.4)
        ax.axhline(7.5,color='#BCC1C7',lw=.7);ax.set_title(title,loc='left',fontsize=10,fontweight='bold',pad=8)
        ax.grid(axis='x',alpha=.15);ax.set_yticks(y);ax.set_ylim(11.6,-.6);ax.tick_params(labelsize=9.4)
    axes[0].set_yticklabels(COMPACT_LABELS,fontsize=9.4)
    axes[1].legend(loc='lower center',bbox_to_anchor=(.5,-.27),ncol=3,frameon=False,fontsize=9.4,
                   handletextpad=.3,columnspacing=1.0)
    fig.subplots_adjust(left=.26,right=.995,top=.91,bottom=.21,wspace=.18)
    fig.savefig(figs/'simulation_main_compact.pdf',bbox_inches='tight');fig.savefig(figs/'simulation_main_compact.png',dpi=240,bbox_inches='tight');plt.close(fig)
    # A focused paired-effect panel makes uncertainty vs both comparators visible.
    fig,axes=plt.subplots(1,2,figsize=(8.4,5.2),sharey=True)
    for ax,bench in zip(axes,['SRS','PC1-X']):
        v=paired[paired.benchmark==bench].set_index('scene').loc[order]
        ax.axvline(0,color='#444444',lw=1,ls='--')
        value=100*(1-v.rmse_ratio);lo=100*(1-v.ratio_upper);hi=100*(1-v.ratio_lower)
        ax.errorbar(value,y,xerr=np.vstack([value-lo,hi-value]),fmt='o',color=COLORS[BUD],ms=4,capsize=2)
        ax.axhline(7.5,color='#BCC1C7',lw=.8);ax.grid(axis='x',alpha=.18)
        ax.set_title('BUD versus '+LABELS[bench]);ax.set_xlabel('RMSE reduction (%)');ax.set_ylim(11.6,-.6)
    axes[0].set_yticks(y);axes[0].set_yticklabels(SCENE_LABELS)
    fig.tight_layout();fig.savefig(figs/'paired_effects.pdf',bbox_inches='tight');fig.savefig(figs/'paired_effects.png',dpi=220,bbox_inches='tight');plt.close(fig)
    # Main three-method table, generated rather than transcribed.
    lines=['\\begin{tabular}{lrrrrr}','\\toprule','Setting & SRS & First PC & BUD & BUD coverage & Length/SRS \\\\','\\midrule']
    for scene,label in zip(order,SCENE_LABELS):
        v=summary[summary.scene==scene].set_index('method');wr=float(v.loc[BUD,'width']/v.loc['SRS','width'])
        lines.append(f"{label} & {v.loc['SRS','rmse']:.4f} & {v.loc['PC1-X','rmse']:.4f} & {v.loc[BUD,'rmse']:.4f} & {v.loc[BUD,'coverage']:.4f} & {wr:.3f} \\\\")
    lines.extend(['\\bottomrule','\\end{tabular}'])
    (out/'table_main.tex').write_text('\n'.join(lines))
    if args.planned_table:
        write_grouped_table(summary, out/'table_planned.tex')
    print(json.dumps(headline,indent=2))

if __name__=='__main__':main()
