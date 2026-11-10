"""Held-separate action and contact tests with whole-scene and whole-date paired controls."""
import argparse
import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from models.g4_shared import Spec
from models.g5_shared import decoded
from training.g4_train import backend, sha_file, write
from training.g5_batches import JointData, load_pack


def average_precision(score,y,weights):
    # Group equal scores before integrating the PR curve so a constant predictor scores prevalence.
    order=np.argsort(-score,kind="stable")
    return ap_table(score[order],y[order],weights[order])


def ap_table(sorted_score,y,w):
    ends=np.r_[sorted_score[:-1]!=sorted_score[1:],True]
    tp=np.cumsum(y*w)[ends]
    n=np.cumsum(w)[ends]
    return float(np.sum(np.diff(tp,prepend=0)*tp/np.maximum(n,1))/max(tp[-1],1))


def raw_features(pack,idx,branch):
    n=pack["n"][idx]
    plan=pack["plan"][idx,branch]
    block=n[:,-1,2:4]
    agent=n[:,-1,:2]
    rel=plan[:,[3,7,11,15],:]-block[:,None]
    return np.column_stack([block,agent-block,n[:,-1,4:6],(n[:,-1,:4]-n[:,-3,:4]),rel.reshape(len(n),-1),
                            np.linalg.norm(rel,axis=-1),np.sum(rel[:,-1]*(block-agent),axis=-1)])


def ridge(train,held):
    rows=np.repeat(np.arange(len(train["n"])),3)
    branches=np.tile(np.arange(3),len(train["n"]))
    x=raw_features(train,rows,branches)
    y=train["outcome"][rows,branches,-1,:]*64
    mean=x.mean(0)
    std=np.maximum(x.std(0),1e-5)
    design=lambda v:np.column_stack([np.ones(len(v)),(v-mean)/std]).astype(np.float64)
    x=design(x)
    penalty=np.eye(x.shape[1]); penalty[0,0]=0
    coef=np.linalg.solve(x.T@x+penalty,x.T@y)
    rows=np.repeat(np.arange(len(held["n"])),3)
    branches=np.tile(np.arange(3),len(held["n"]))
    return (design(raw_features(held,rows,branches))@coef).reshape(-1,3,2)


def evaluate(run,root,out,role,device,gate=None):
    dev=backend(device)
    card=json.loads((run/"config.json").read_bytes())
    done=json.loads((run/"complete.json").read_bytes())
    sha=done["sha256"]["inference.npz"]
    if sha_file(run/"inference.npz")!=sha:
        raise ValueError("Inference hash differs from training completion")
    if role=="test":
        if gate is None:
            raise ValueError("Sealed role requires the previous four-weights promotion lock")
        lock=json.loads(gate.read_bytes())
        if lock.get("study")=="G7":
            from scripts.train.g7_scale_lock import verify
            verify(lock,run,sha,root)
        elif not lock["promote"] or lock["sha256"].get(run.name)!=sha or len(lock["sha256"])!=4:
            raise ValueError("The run is outside the frozen confirmation")
    spec=Spec(**card["spec"])
    with np.load(run/"inference.npz",allow_pickle=False) as f:
        stats=np.array(f["stats"])
        p={k:jax.device_put(f[k]) for k in f if k!="stats" and not k.startswith("prior_")}
        cal={k[6:]:jax.device_put(f[k]) for k in f if k.startswith("prior_")}
    data=JointData(root,role,stats=stats)
    train,_=load_pack(root/"pusht/train",True)
    imp,sim=data.packs[1:]
    @jax.jit
    def infer(b):
        if card.get("variant")=="physical-residual":
            from models.g5_residual import world
            _,onset,delta=world(p,spec,b,b["source"],cal)
            return onset,delta[:,-1]
        pred=decoded(p,spec,b,b["source"])
        return pred[:,:,632],pred[:,31,633:635]
    def outputs(family,indices,branches,source,condition="correct"):
        ev,dy=[],[]
        for j in range(0,len(indices),64):
            idx=indices[j:j+64]
            br=branches[j:j+64]
            # Fixed shape including the last minibatch.
            padded=np.pad(idx,(0,64-len(idx)),mode="edge")
            pb=np.pad(br,(0,64-len(br)),mode="edge")
            b=data.gather(np.full(64,family),padded,pb)
            b["source"][:,source]=1
            if condition=="shift":
                b["v"][:,8:]=b["v"][:,:-8].copy()
            elif condition=="erase":
                b["sense"][:]=0
            elif condition=="no_plan":
                b["u"][:]=0
            event,delta=map(np.asarray,infer(b))
            ev.append(event[:len(idx)])
            dy.append(delta[:len(idx)]*64)
        return np.concatenate(ev),np.concatenate(dy)
    pi=np.repeat(np.arange(len(sim["group"])),3)
    br=np.tile(np.arange(3),len(sim["group"]))
    outcomes={}
    for key,source,condition in (("joint",[1,2],"correct"),("state",[2],"correct"),("video",[1],"correct"),
                                 ("unplanned",[1,2],"no_plan")):
        outcomes[key]=outputs(8,pi,br,source,condition)[1].reshape(-1,3,2)
    truth=sim["outcome"][:,:,-1,:].astype(float)*64
    velocity=(sim["n"][:,-1,2:4]-sim["n"][:,-2,2:4])*256*16
    outcomes.update(persistence=np.zeros_like(truth),velocity=np.broadcast_to(velocity[:,None],truth.shape),
                    ridge=ridge(train,sim))
    diffs=[(0,1),(0,2),(1,2)]
    raw={}
    scores={}
    for k,y in outcomes.items():
        error=y-truth
        raw["push_"+k]=np.linalg.norm(error,axis=-1).mean(1)
        raw["branch_"+k]=np.stack([np.linalg.norm(error[:,a]-error[:,b],axis=-1) for a,b in diffs]).mean(0)
        scores[k]={"endpoint_px":float(raw["push_"+k].mean()),"branch_difference_px":float(raw["branch_"+k].mean())}
    indices=np.arange(len(imp["group"]))
    y=np.array(imp["event"][:,8:],dtype=np.uint8).reshape(-1)
    groups=np.repeat(imp["group"],56)
    raw.update(impact_y=y,impact_group=groups,sim_group=np.asarray(sim["group"]))
    aps={}
    for condition in ("correct","shift","erase","raw_motion"):
        if condition=="raw_motion":
            v=np.asarray(imp["v"],np.float32)
            values=np.abs(v[:,8:]-v[:,7:-1]).mean(-1).reshape(-1)
        else:
            values=outputs(7,indices,np.zeros(len(indices),int),[1],condition)[0][:,8:].reshape(-1)
        raw["impact_"+condition]=values
        aps[condition]=average_precision(values,y,np.ones(len(y)))
    out.mkdir(exist_ok=True,parents=True)
    with (out/(role+"-predictions.partial")).open("wb") as f:
        np.savez_compressed(f,**raw)
    (out/(role+"-predictions.partial")).replace(out/(role+"-predictions.npz"))
    result={"run":run.name,"split":role,"backend":dev.device_kind,"checkpoint":sha,"examples":{"scenes":len(sim["n"]),
            "impact_windows":len(indices),"impact_dates":len(np.unique(groups)),"impact_ticks":len(y),"positive_ticks":int(y.sum())},
            "push":scores,"impact":{"average_precision":aps,"positive_prevalence":float(y.mean())}}
    write(out/(role+"-metrics.json"),result)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True); p.add_argument("--run",required=True,type=Path)
    p.add_argument("--role",choices=["dev","test"],required=True); p.add_argument("--backend",required=True)
    p.add_argument("--out",type=Path,required=True); p.add_argument("--lock",type=Path)
    a=p.parse_args(); evaluate(a.run,a.data,a.out,a.role,a.backend,a.lock)
