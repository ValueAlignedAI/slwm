"""Fit small, training-only causal references, then freeze them for both G5C cores."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from training.g4_data import digest


def geometry(history,plan,idx,xp=np):
    block=history[:,-1,2:4]
    agent=history[:,-1,:2]
    rel=plan[:,idx,:]-block[:,None]
    return xp.concatenate((block,agent-block,history[:,-1,4:6],history[:,-1,:4]-history[:,-3,:4],
                           rel.reshape(len(plan),-1),xp.linalg.norm(rel,axis=-1),
                           xp.sum(rel[:,-1]*(block-agent),axis=-1,keepdims=True)),axis=-1)


def fit_priors(root,out):
    paths={f:root/f/"train" for f in ("impact","pusht")}
    cards={k:json.loads((d/"card.json").read_bytes()) for k,d in paths.items()}
    def load(f,k):
        path=paths[f]/(k+".npy")
        if digest(path)!=cards[f]["sha256"][k]:
            raise ValueError("Calibration data hash mismatch")
        return np.load(path,mmap_mode="r",allow_pickle=False)
    n=load("pusht","n")
    plan=load("pusht","plan").reshape(-1,16,2)
    y=load("pusht","outcome").reshape(-1,16,2)
    h=np.repeat(n,3,axis=0)
    coefs=[]
    for t in range(16):
        idx=np.minimum([3,7,11,15],t)
        x=geometry(h,plan,idx).astype(np.float64)
        mean=x.mean(0); std=np.maximum(x.std(0),1e-5)
        v=np.column_stack([np.ones(len(x)),(x-mean)/std])
        reg=np.eye(v.shape[1]); reg[0,0]=0
        w=np.linalg.solve(v.T@v+reg,v.T@y[:,t])
        coefs.append(np.vstack([w[0]-(mean/std)@w[1:],w[1:]/std[:,None]]))
    video=load("impact","v")
    event=load("impact","event")
    xs=[]; ys=[]
    for start in range(0,len(video),128):
        v=np.asarray(video[start:start+128],np.float32)
        xs.append(np.log1p(np.abs(v[:,1:]-v[:,:-1]).mean(-1)).ravel())
        ys.append(event[start:start+128,1:].ravel())
    x=np.concatenate(xs).astype(np.float64); y=np.concatenate(ys).astype(np.float64)
    def value(w):
        logit=w[0]+w[1]*x
        eps=1/(1+np.exp(-logit))-y
        return np.mean(np.logaddexp(0,logit)-y*logit),np.array([eps.mean(),np.mean(eps*x)])
    trained=minimize(value,[-4.,.4],jac=True,bounds=[(-20,0),(1e-5,8)],method="L-BFGS-B")
    if not trained.success:
        raise RuntimeError(trained.message)
    out.parent.mkdir(parents=True,exist_ok=True)
    # Use deterministic contiguous arrays. No held observations enter either fit.
    with out.with_suffix(".partial").open("wb") as fd:
        np.savez(fd,coef=np.asarray(coefs,np.float32),contact=trained.x.astype(np.float32))
    out.with_suffix(".partial").replace(out)
    print(json.dumps({"file_sha256":digest(out),"training_scenes":len(n),"impact_ticks":len(x),
                      "contact_logistic":trained.x.tolist()}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); fit_priors(a.data,a.out)
