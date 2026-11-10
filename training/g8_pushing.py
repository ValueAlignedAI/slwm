"""Shared, causal physical forecasts and paired decision scoring on reserved three-plan scenes."""
from functools import lru_cache
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from models.g5_residual import world
from training.g5_batches import load_pack
from training.g5_eval import raw_features
from training.g4_train import write
from training.g8_models import G8_DATA,read
from scripts.train.g7_lock_heads import sha

VIEWS={"joint":(1,1),"state":(0,1),"video":(1,0),"joint_wrong_camera":(1,1),"video_wrong_camera":(1,0),"no_plan":(1,1),"erased_camera_only":(1,0)}


@lru_cache(maxsize=4)
def kernel(spec):
    @jax.jit
    def apply(p,cal,n,v,u,source,erase):
        b=len(n);ok=jnp.broadcast_to(jnp.arange(64)<32,(b,64)).astype(jnp.uint8)
        sense=jnp.broadcast_to(jnp.arange(64)<16,(b,64)).astype(jnp.uint8)
        features={"a":jnp.zeros((b,64,100),jnp.uint8),"v":jnp.pad(v,((0,0),(0,48),(0,0))),"n":jnp.pad(n,((0,0),(0,48),(0,11))),
                  "u":jnp.pad(u,((0,0),(16,32),(0,0))),"source":source,"sense":sense*(1-erase),"ok":ok,
                  "input_x":jnp.zeros((b,64),jnp.int32),"family":jnp.full((b,),8,jnp.uint8)}
        _,_,dy=world(p,spec,features,source,cal)
        return dy[:,-1]*64
    return apply


def predictions(spec,p,cal,n,v,plans,condition="joint"):
    """n/v describe observed 16-tick histories. No future observations enter this function."""
    use_v,use_n=VIEWS[condition];b,k=plans.shape[:2]
    n=np.repeat(n,k,axis=0);v=np.repeat(v,k,axis=0);u=plans.reshape(b*k,16,2)
    if not use_n:n=np.zeros_like(n)
    if not use_v or condition=="erased_camera_only":v=np.zeros_like(v)
    if condition=="no_plan":u=np.zeros_like(u)
    src=np.tile(np.array([0,use_v,use_n,0],np.uint8),(len(n),1))
    result=kernel(spec)(p,cal,n,v,u,src,np.uint8(condition=="erased_camera_only"))
    return np.asarray(result).reshape(b,k,2)


class Geometry:
    def __init__(self,old):
        train,_=load_pack(old/"pusht/train",True)
        rows=np.repeat(np.arange(len(train["n"])),3);branches=np.tile(np.arange(3),len(train["n"]))
        x=raw_features(train,rows,branches);y=train["outcome"][rows,branches,-1]*64
        self.mean=x.mean(0);self.std=np.maximum(x.std(0),1e-5)
        des=self.design(x);reg=np.eye(des.shape[1]);reg[0,0]=0
        self.coef=np.linalg.solve(des.T@des+reg,des.T@y)
    def design(self,x):return np.column_stack((np.ones(len(x)),(x-self.mean)/self.std)).astype(np.float64)
    def predict(self,n,plans):
        k=plans.shape[1];rows=np.repeat(np.arange(len(n)),k);branch=np.tile(np.arange(k),len(n))
        x=raw_features({"n":n,"plan":plans},rows,branch)
        return (self.design(x)@self.coef).reshape(len(n),k,2)


def summarize(pred,truth,goals):
    preds=pred.astype(np.float64);real=truth.astype(np.float64)
    costs=np.linalg.norm(real[:,None]-goals[:,:,None],axis=-1);dist=np.linalg.norm(preds[:,None]-goals[:,:,None],axis=-1)
    ties=np.isclose(dist,dist.min(-1,keepdims=True),rtol=0,atol=1e-9);choice=ties/ties.sum(-1,keepdims=True)
    best=costs.min(-1);chosen=(choice*costs).sum(-1);e=preds-real
    branch=np.stack([np.linalg.norm(e[:,l]-e[:,r],axis=-1) for l,r in ((0,1),(0,2),(1,2))],axis=1).mean(1)
    return {"regret_px":(chosen-best).mean(1),"endpoint_px":np.linalg.norm(e,axis=-1).mean(1),"branch_error_px":branch,
            "best_plan_rate":(choice*np.isclose(costs,best[...,None],rtol=0,atol=1e-8)).sum(-1).mean(1),"goal_distance_px":chosen.mean(1)}


def panel(task,doc):
    folder=G8_DATA/"panels"/task;meta=read(folder/"card.json")
    if sha(folder/"card.json")!=doc["panels"][task] or sha(folder/"scenes.npz")!=meta["sha256"]:raise ValueError("Confirmation scenes changed")
    with np.load(folder/"scenes.npz") as f:return {k:np.array(v) for k,v in f.items()}


def evaluate(task,tag,spec,p,cal,sim,prior,out,weight):
    dest=out/"confirmation"/task/tag;dest.mkdir(parents=True,exist_ok=True)
    if (dest/"metrics.json").exists():
        d=read(dest/"metrics.json")
        if d["weight"]!=weight or d["row_sha256"]!=sha(dest/"rows.npz"):raise ValueError("Existing scientific result changed")
        return
    rawrows={"group":sim["group"]};scores={};truth=sim["outcome"][:,:,-1]*64
    if tag=="raw":
        values={"geometry":prior.predict(sim["n"],sim["plan"]),"persistence":np.zeros_like(truth)}
        scores["first_proposal"]={"regret_px":float((np.linalg.norm(truth[:,0,None]-sim["goals"],axis=-1)-np.linalg.norm(truth[:,None]-sim["goals"][:,:,None],axis=-1).min(-1)).mean())}
    else:
        values={}
        for cond in VIEWS:
            pred=[]
            for j in range(0,len(truth),64):
                ids=np.arange(j,min(j+64,len(truth)));vid=sim["v"][sim["wrong"][ids] if "wrong" in cond else ids]
                pred.append(predictions(spec,p,cal,sim["n"][ids],vid,sim["plan"][ids],cond))
            values[cond]=np.concatenate(pred)
    for cond,estimate in values.items():
        measures=summarize(estimate,truth,sim["goals"])
        for k,v in measures.items():rawrows[cond+"_"+k]=v.astype(np.float32)
        scores[cond]={k:float(v.mean()) for k,v in measures.items()}
    with (dest/"rows.part").open("wb") as f:np.savez_compressed(f,**rawrows)
    (dest/"rows.part").replace(dest/"rows.npz")
    write(dest/"metrics.json",{"task":task,"model":tag,"weight":weight,"starts":len(truth),"goals_per_start":sim["goals"].shape[1],
           "scores":scores,"row_sha256":sha(dest/"rows.npz")})
