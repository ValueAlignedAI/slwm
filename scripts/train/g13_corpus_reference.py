"""Fit a direct causal raw-signal action predictor and include zero and repeat-last-command references."""
import numpy as np
from training.g13_physics import group_mean
from evals.g13_analysis import raw_x

def design(r):
    hist=r["kin"][:,:,2:].reshape(len(r["kin"]),-1)/.05
    return np.concatenate([raw_x(r),hist,np.tanh(hist),np.linalg.norm(r["kin"][:,:,2:4],axis=2)/.05],1)

def fit(train,dev):
    x=design(train).astype("f8");y=train["plan"].sum(1);g=train["group"]
    _,idx,n=np.unique(g,return_inverse=True,return_counts=True);w=1./n[idx];w/=w.sum()
    mu=(x*w[:,None]).sum(0);sd=np.maximum(np.sqrt(((x-mu)**2*w[:,None]).sum(0)),.04)
    xx=(x-mu)/sd;base=(y*w[:,None]).sum(0)
    gram=xx.T@(xx*w[:,None]);target=xx.T@((y-base)*w[:,None]);z=(design(dev)-mu)/sd
    candidates=[]
    for lam in (.005,.04,.2,1.):
        coef=np.linalg.solve(gram+lam*np.eye(len(mu)),target)
        yp=z@coef+base;err=np.linalg.norm(yp-dev["plan"].sum(1),axis=1)*1000
        candidates.append((group_mean(err,dev["group"]),lam,coef))
    ev,lam,coef=min(candidates,key=lambda r:r[:2])
    return dict(mu=mu,sd=sd,beta=coef,base=base,lam=np.array(lam),development_mm=np.array(ev))

def apply(r,params):return (design(r)-params["mu"])/params["sd"]@params["beta"]+params["base"]
