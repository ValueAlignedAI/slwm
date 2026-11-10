"""Compare held and in-sample visual contact ranking against the identical physical prior."""
import argparse
import json
from pathlib import Path

import jax
import numpy as np

from models.g4_shared import Spec
from models.g5_residual import world
from training.g4_train import sha_file,write
from training.g5_batches import JointData
from training.g5_eval import average_precision


def metrics(logit,prior,labels,groups):
    logits=np.asarray(logit,np.float64); baseline=np.asarray(prior,np.float64); y=np.asarray(labels,np.float64)
    def summary(v):
        q=1/(1+np.exp(-np.clip(v,-60,60)))
        w=v+np.log(8.)
        return {"ap":average_precision(v,y,np.ones(len(y))),"brier":float(np.mean((q-y)**2)),
                "training_weighted_bce":float(np.mean(np.logaddexp(0,w)*(1-y)+8*np.logaddexp(0,-w)*y))}
    model=summary(logits); base=summary(baseline)
    effects=[]
    for g in np.unique(groups):
        use=groups==g
        if y[use].sum():
            effects.append(average_precision(logits[use],y[use],np.ones(use.sum()))-
                           average_precision(baseline[use],y[use],np.ones(use.sum())))
    x=np.array(effects)
    ix=np.random.default_rng(61123).integers(0,len(x),(2000,len(x)))
    return {"model":model,"motion_prior":base,"ap_gain":model["ap"]-base["ap"],"ticks":len(y),"prevalence":float(y.mean()),
            "dates":len(np.unique(groups)),"dates_with_events":len(x),"equal_date_mean_ap_gain":float(x.mean()),
            "equal_date_ci95":np.quantile(x[ix].mean(1),[.025,.975]).tolist()}


def score(a):
    train=None; selected=None; results={}
    for step in (2048,8192):
        stage=a.run/f"step-{step:06d}"; cfg=json.loads((stage/"config.json").read_bytes())
        proof=json.loads((stage/"verified.json").read_bytes()); model=proof["sha256"]["inference.npz"]
        if proof["step"]!=step or sha_file(stage/"inference.npz")!=model: raise ValueError("Wrong audited weights")
        with np.load(stage/"inference.npz") as f:
            p={k:jax.device_put(f[k]) for k in f if k!="stats" and not k.startswith("prior_")}
            cal={k[6:]:jax.device_put(f[k]) for k in f if k.startswith("prior_")}
            if train is None: train=JointData(a.data,"train",verify=False,stats=f["stats"])
        pack=train.packs[1]
        if selected is None:
            rng=np.random.default_rng(61120)
            selected=np.concatenate([rng.choice(ids,min(30,len(ids)),replace=False) for g in np.unique(pack["group"])
                                     for ids in [np.flatnonzero(pack["group"]==g)]])
        spec=Spec(**cfg["spec"])
        @jax.jit
        def forward(params,b):
            _,logit,_=world(params,spec,b,b["source"],cal)
            return logit[:,8:]
        logits=[]; motion=[]
        for j in range(0,len(selected),32):
            rows=selected[j:j+32]; n=len(rows); idx=np.pad(rows,(0,32-n),mode="edge")
            b=train.gather(np.full(32,7),idx,np.zeros(32,int)); b["source"][:,1]=1; b["a"].fill(0)
            logits.append(np.asarray(forward(p,b))[:n])
            v=b["v"][:n].astype(np.float32)
            motion.append(np.abs(v[:,8:]-v[:,7:-1]).mean(-1))
        w=np.asarray(cal["contact"])
        prior=lambda v:w[0]+w[1]*np.log1p(v)
        labels=pack["event"][selected,8:].reshape(-1); groups=np.repeat(pack["group"][selected],56)
        learn=metrics(np.concatenate(logits).ravel(),prior(np.concatenate(motion).ravel()),labels,groups)
        with np.load(stage/"dev-predictions.npz") as f:
            held=metrics(f["impact_correct"],prior(f["impact_raw_motion"]),f["impact_y"],f["impact_group"])
        results[str(step)]={"weights":model,"train":learn,"held":held,
                            "training_minus_held_ap_gain":learn["ap_gain"]-held["ap_gain"]}
        del p,cal
        jax.clear_caches()
    write(a.out,{"run":cfg["run"],"windows_per_train_date_cap":30,"chosen_training_windows":len(selected),
                 "purpose":"post hoc generalization diagnostic without model selection","steps":results})
    print(cfg["run"],[(s,round(v["train"]["ap_gain"],3),round(v["held"]["ap_gain"],3)) for s,v in results.items()],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--run",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True); p.add_argument("--out",type=Path,required=True); score(p.parse_args())
