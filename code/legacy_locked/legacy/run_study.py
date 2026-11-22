"""Extracted without changing function bodies from the project historical
B_DDS_UD_Bootstrap_Exploration/run_study.py.
Original SHA256: 2d218234d7b9a0f7518cf14b1012ce3fe0b18fe718262097cf051da46d222daa

Only the legacy synthetic generator / predicted influence-process functions
are retained. They are project exploration code, not an implementation
distributed by the DDS or MVRS paper authors. In particular, pe/py use
DGP-derived conditional hazards and give MVRS-PROXY extra information.
"""
import numpy as np

J=24
DT=3.0/J
TGRID=np.arange(1,J+1)
EVAL=np.arange(J)>=3


def data_generation(rng, n, scenario, proxy_quality='strong'):
    t=rng.standard_normal(n); r=rng.standard_normal(n); v=rng.standard_normal(n)
    g=np.repeat(np.arange(2),n//2)
    if scenario=='banana':
        raw=np.column_stack([t, .85*(t*t-1)+.45*r, np.sin(1.6*t)+.22*r,
            np.cos(1.1*t)+.3*r,t*r*.45,.55*r+.15*t,
            t*t*t/2.5, np.sin(r)+.2*t,np.cos(.8*r)+.1*t,
            (t*t-1)*r/2,.7*t-.3*r,.7*(t*t-1)+.2*v])
        ae=1.55*t+.55*r; al=1.30*r-.75*t+.35*(t*t-1)
    elif scenario=='s_curve':
        a=rng.uniform(-1.4*np.pi,1.4*np.pi,size=n)
        b=rng.uniform(-1.5,1.5,size=n)
        raw=np.column_stack([np.sin(a),b,np.sign(a)*(np.cos(a)-1),
            .7*np.sin(.7*a)+.2*b,np.cos(.75*a),
            b*np.sin(a), b*np.cos(a),np.sin(2*a),
            b*b-0.75,np.cos(1.7*a),.5*np.sign(a)*(np.cos(a)-1)+.1*b,.3*np.sin(a)+.3*b])
        ae=1.55*np.sin(a)+.65*b; al=1.15*np.cos(a)+.95*b+.25*np.sin(2*a)
    elif scenario=='two_moons':
        z=rng.choice([-1.,1.],size=n)
        raw=np.column_stack([t+.08*r,z*(1.3+.65*t*t)+.22*r,z*t,
            np.sin(1.2*t)+.2*z,np.cos(t)+.1*r,
            r+.2*z,z,t*t-1,t*r*.5,
            .6*z+.4*np.sin(t),.2*v+.7*r,.6*t+.3*z*r])
        ae=1.25*t+.80*z; al=1.05*r+.85*z*t
    elif scenario=='gaussian':
        raw=np.column_stack([t+.1*r,t*.7+.2*r,t*.5+.4*r,*.5*np.ones(n)] ) if False else None
        raw=np.column_stack([1.4*t+.1*r,1.1*t+.2*v,.85*t+.1*r,
             .7*r+.3*t,.6*r+.2*v,.5*t+.15*r, 
             .1*v+.5*r,.2*t+.1*v,.3*r+.05*v,
             .2*t+.1*r,.1*v+.25*t,.3*t+.2*r])
        raw+=rng.normal(scale=.1,size=raw.shape)
        ae=1.55*t+.2*r; al=1.20*t+.2*r
    else: raise ValueError(scenario)
    # Low-dimensional nonlinear manifold embedded in 12 standardized coordinates.
    raw+=rng.normal(scale=.035 if scenario!='gaussian' else .02,size=raw.shape)
    x=(raw-np.mean(raw,axis=0))/np.maximum(np.std(raw,axis=0),1e-10)
    early=(np.arange(J)<12)[None,:]
    # Time-varying contrasts, non-proportional hazards.
    base=np.where(early,.15,.33)
    pred_base=np.where(early,.15,.33)
    a_true=np.where(early,ae[:,None],al[:,None])+g[:,None]*np.where(early,.18,-.18)
    # The external Phase-I surrogate has strong but imperfect prognostic information.
    err=.18*(x[:,10]-x[:,11]); strength=.90 if proxy_quality=='strong' else .40
    a_proxy=np.where(early,strength*ae[:,None]+err[:,None],strength*al[:,None]-err[:,None])
    a_proxy+=g[:,None]*np.where(early,.13,-.13)
    true_h=np.clip(base*np.exp(np.clip(a_true,-3.0,2.2)),.005,4.0)
    pred_h=np.clip(pred_base*np.exp(np.clip(a_proxy,-3.0,2.2)),.005,4.0)
    wait=rng.exponential(size=n)
    csum=np.cumsum(true_h*DT,axis=1)
    tbin=np.where(csum[:,-1]>=wait,(csum<wait[:,None]).sum(axis=1)+1,J+1)
    censor=rng.exponential(scale=1/.105,size=n)
    cbin=np.minimum(np.floor(censor/DT).astype(int)+1,J+1)
    u=np.minimum(tbin,cbin).astype(np.int16)
    event=((tbin<=cbin)&(tbin<=J)).astype(np.int8)
    pe=1-np.exp(-pred_h*DT)
    py=np.ones_like(pe)
    py[:,1:]=np.cumprod((1-pe[:,:-1])*np.exp(-.105*DT),axis=1)
    return g,x,u,event,pe,py


def predict_if(pe,py):
    alpha=(py*pe).sum(axis=0)/np.maximum(py.sum(axis=0),1e-9)
    sy=np.cumprod(1-alpha)
    denom=np.maximum(py.mean(axis=0)*(1-alpha),1e-8)
    mu=-sy[None,:]*np.cumsum(py*(pe-alpha[None,:])/denom,axis=1)
    return mu
