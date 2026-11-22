"""Extracted without changing function bodies from the project historical
B_DDS_UD_Bootstrap_Exploration/formal_vo_dds.py.
Original SHA256: 6360e3cbdfe3582af8ff25bd70c9afd2c81093458fa1cabd7c1b0849525e7b2a

Only the legacy synthetic generator / predicted influence-process functions
are retained. They are project exploration code, not an implementation
distributed by the DDS or MVRS paper authors. In particular, pe/py use
DGP-derived conditional hazards and give MVRS-PROXY extra information.
"""
import numpy as np
from . import run_study as old

N=2400
J=old.J
DT=old.DT
TGRID=old.TGRID
EVAL=old.EVAL


def generate(rng,scenario,proxy_noise=0.16):
    n=N
    if scenario in ('banana','gaussian'):
        gg,x,u,event,pe,py=old.data_generation(rng,n,scenario,'strong')
        return gg,x,u,event,pe,py
    g=np.repeat(np.arange(2),n//2)
    t,r,z,w=rng.normal(size=(4,n))
    if scenario=='saddle3d':
        # 3-d non-convex saddle response manifold in 16 dims, x contains nonlinear low-rank features
        x=np.column_stack([t,r,z, .9*(t*t-r*r), t*r, np.sin(1.5*t),np.cos(1.4*r),
             (t*t-1)*z*.5,np.sin(z+r),.6*t+.5*r,.5*z-.3*r,t*r*z*.25,
             .6*np.cos(t+z), .45*t*t+.4*z,.4*r*r+.5*t,.5*np.sin(2*z)+.2*t])
        a_early=1.35*t+.78*np.sin(1.3*r)+.50*z
        a_late= -1.05*t+.9*(r*r-1)/1.6+.7*np.cos(1.5*z)
    elif scenario=='multimodal':
        mode=rng.choice([-1.,1.],size=n)
        # multi-modal nonlinear manifold with correlated repeated noisy measurements.
        # All 16 columns standardized; nuisance measured repeatedly common in EHR.
        x=np.column_stack([t,mode*(1+.7*t*t)+.25*r,mode*t,
            np.sin(1.3*t)+.4*r,np.cos(t)+.15*z,t*r*.45,
            z+.15*t,mode+.3*z, (t*t-1)*.55,
            .5*mode*t+.3*r, .3*r+.6*z, .5*np.sin(z)+.3*t,
            .8*w+.08*z,.83*w+.08*r,.85*w+.08*t,.81*w+.08*mode])
        a_early=1.15*t+.9*mode+.52*np.sin(r)
        a_late=-.9*mode*t+1.0*r+.5*(t*t-1)/2
    else: raise ValueError(scenario)
    x+=rng.normal(scale=.06,size=x.shape)
    x=(x-x.mean(axis=0))/np.maximum(x.std(axis=0),1e-8)
    early=(np.arange(J)<12)[None,:]
    base=np.where(early,.14,.30)
    a_true=np.where(early,a_early[:,None],a_late[:,None])+g[:,None]*np.where(early,.17,-.17)
    # Proxy is external phase-I prognostic output: perturbed true conditional risk scores.
    a_pred=np.where(early,.86*a_early[:,None]+proxy_noise*(x[:,12]-x[:,10])[:,None],
          .86*a_late[:,None]+proxy_noise*(x[:,7]-x[:,11])[:,None])
    a_pred+=g[:,None]*np.where(early,.13,-.13)
    htrue=np.clip(base*np.exp(np.clip(a_true,-3.0,2.2)),.005,4.)
    hpred=np.clip(base*np.exp(np.clip(a_pred,-3.0,2.2)),.005,4.)
    wait=rng.exponential(size=n)
    ch=np.cumsum(htrue*DT,axis=1)
    tbin=np.where(ch[:,-1]>=wait,(ch<wait[:,None]).sum(axis=1)+1,J+1)
    censor=rng.exponential(scale=1/.105,size=n)
    cbin=np.minimum((censor/DT).astype(int)+1,J+1)
    u=np.minimum(tbin,cbin).astype(np.int16)
    event=((tbin<=cbin)&(tbin<=J)).astype(np.int8)
    pe=1-np.exp(-hpred*DT)
    py=np.ones_like(pe);py[:,1:]=np.cumprod((1-pe[:,:-1])*np.exp(-.105*DT),axis=1)
    return g,x,u,event,pe,py
