"""Fixed-budget transfer of frozen multimodal states to independently executed Reacher torques."""
import argparse
from functools import lru_cache
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from models.g5_shared import encode
from training.g7_camera import initial,fit,forecast
from training.g7_policy import data,obs_batch
from training.g7_reacher_prior import design,rotate
from training.g7_lookahead import train_model,mlp,save
from training.g8_models import CONFIG,G7_DATA,G8_DATA,read,key,resolve,load_model,sealed
from scripts.train.g6_artifacts import roots
from scripts.train.g7_lock_heads import sha
from training.g4_train import write


def training_rows():
    lo,hi=read(CONFIG)["panels"]["reacher_readout_train"]
    rows=data(G7_DATA/"reacher/train");take=np.flatnonzero((rows["seed"]>=lo)&(rows["seed"]<hi))
    np.testing.assert_array_equal(rows["seed"][take],np.arange(lo,hi))
    return {k:v[take] for k,v in rows.items()}


@lru_cache(maxsize=4)
def encoder(spec):
    @jax.jit
    def apply(p,n,v):
        m=len(n);num=n.reshape(m,32,17);img=v.reshape(m,32,256)
        if spec is None:
            picture=img.reshape(m,32,4,4,4,4).mean((3,5)).reshape(m,32,16)/127.5-1
            z=jnp.pad(jnp.concatenate((num,picture),-1),((0,0),(0,0),(0,95)))
        else:
            batch=obs_batch(num,img);batch["u"]=num[:,:,10:12]
            source=jnp.broadcast_to(jnp.array([0,1,1,0],jnp.uint8),(m,4))
            z,_=encode(p,spec,batch,source)
        z=z.astype(jnp.float32).reshape(m,4,8,128)
        return jnp.concatenate((z.mean(2),z[:,:,-1],n[:,:,:,10:12].reshape(m,4,16)),-1)
    return apply


def fields(rows,rec):
    if rec:spec,p,_,_=load_model(rec)
    else:spec,p=None,{}
    fn=encoder(spec);pieces=[]
    for i in range(0,len(rows["seed"]),64):
        pieces.append(np.asarray(fn(p,rows["n"][i:i+64],rows["v"][i:i+64])))
    return np.concatenate(pieces)


def physical_inputs(rows):
    x,sc=design(rows["n"],rows["actions"],np);y=rotate(rows["end"]/.21,sc,np,True)
    return x.astype(np.float32),y.astype(np.float32),sc


def train(name):
    primary,_=roots("g8");out=primary/"reacher-heads"/name;out.mkdir(exist_ok=True,parents=True)
    if (out/"complete.json").exists():return
    rows=training_rows()
    if name=="physical":
        x,y,_=physical_inputs(rows);flat=x.reshape(-1,x.shape[-1]);target=y.reshape(-1,2)
        mean=flat.mean(0);std=flat.std(0).clip(.05)
        train_model(out,(x.shape[-1],192,192,192,2),(flat-mean)/std,target,np.ones_like(target),1024,811)
        np.savez(out/"normalization.npz",mean=mean,std=std)
        parent="matched-train-only-body-frame";size=sum(v.size for v in np.load(out/"weights.npz").values())
    else:
        core,seed,*arm=name.split("-");seed=int(seed)
        rec=None if core=="raw" else resolve(core,seed,arm[0])
        if core!="raw" and not rec:raise ValueError("Unregistered transfer parent")
        x=fields(rows,rec);net,optimizer=fit(initial(seed+200),x,rows)
        save(out/"weights.npz",net);save(out/"optimizer.npz",{f"v{i}":v for i,v in enumerate(optimizer)})
        parent=rec["weights"] if rec else "raw-observations";size=sum(v.size for v in net.values())
    write(out/"complete.json",{"parent":parent,"weights":sha(out/"weights.npz"),"optimizer":sha(out/"optimizer.npz"),
        "normalization":sha(out/"normalization.npz") if name=="physical" else None,"parameters":size,"updates":1024,
        "train_starts":len(rows["seed"]),"executed_counterfactuals":int(np.prod(rows["end"].shape[:-1])),
        "train_range":read(CONFIG)["panels"]["reacher_readout_train"],"training_data":sha(G7_DATA/"reacher/train/card.json"),
        "training_code":{f:sha(Path(f)) for f in ("training/g8_reacher.py","training/g7_camera.py","training/g7_lookahead.py")}})
    print("Frozen Reacher head",name,flush=True)


def score(name,lockdoc):
    primary,_=roots("g8");folder=primary/"reacher-heads"/name
    if sha(folder/"weights.npz")!=lockdoc["readouts"][name]:raise ValueError("Unregistered transfer head")
    data_folder=G8_DATA/"panels/reacher_test"
    if sha(data_folder/"card.json")!=lockdoc["panels"]["reacher_test"]:raise ValueError("Reacher split changed")
    rows=data(data_folder);n=len(rows["seed"])
    with np.load(folder/"weights.npz") as f:net={k:jnp.asarray(v) for k,v in f.items()}
    if name=="physical":
        norm=folder/"normalization.npz"
        if sha(norm)!=read(folder/"complete.json")["normalization"]:raise ValueError("Physical fit normalization changed")
        with np.load(norm) as f:mu,std=f["mean"],f["std"]
        x,_,sc=physical_inputs(rows);local=np.asarray(jax.jit(mlp)(net,(x-mu)/std))
        y=rotate(local,sc,np)*.21
    else:
        x=fields(rows,lockdoc["models"].get(name))
        y=np.asarray(jax.jit(forecast)(net,x,rows["actions"]))* .21
    # Decisions 2–4 use both independently counterbalanced goals. The predictor never receives goal or future outcomes.
    cost=np.linalg.norm(rows["end"][:,1:,None]-rows["goals"][:,None,:,None],axis=-1)*100
    predicted=np.linalg.norm(y[:,1:,None]-rows["goals"][:,None,:,None],axis=-1)
    tie=np.isclose(predicted,predicted.min(-1,keepdims=True),rtol=0,atol=1e-9);weights=tie/tie.sum(-1,keepdims=True)
    best=cost.min(-1);regret=((weights*cost).sum(-1)-best).mean((1,2))
    endpoint=(np.linalg.norm(y[:,1:]-rows["end"][:,1:],axis=-1)*100).mean((1,2))
    hit=(weights*np.isclose(cost,best[...,None],rtol=0,atol=1e-6)).sum(-1).mean((1,2))
    out=primary/"confirmation"/"reacher"/name;out.mkdir(exist_ok=True,parents=True)
    with (out/"rows.part").open("wb") as f:np.savez_compressed(f,seed=rows["seed"],regret_cm=regret,endpoint_cm=endpoint,hit=hit,
        uniform_regret_cm=(cost.mean(-1)-best).mean((1,2)),passive_regret_cm=(cost[...,-1]-best).mean((1,2)))
    (out/"rows.part").replace(out/"rows.npz")
    write(out/"metrics.json",{"head":name,"weights":lockdoc["readouts"][name],"rows":n,"regret_cm":float(regret.mean()),"endpoint_cm":float(endpoint.mean()),
        "optimal_action_rate":float(hit.mean()),"uniform_regret_cm":float((cost.mean(-1)-best).mean()),"passive_regret_cm":float((cost[...,-1]-best).mean()),"row_sha256":sha(out/"rows.npz")})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("train","score"));p.add_argument("name");a=p.parse_args()
    if a.action=="train":train(a.name)
    else:score(a.name,sealed())
