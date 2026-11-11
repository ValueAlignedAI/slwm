"""Fit, development-select, and score compositional speech on genuinely unseen utterances and speakers."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jnp
from models.g14_cue import initial,objective,forward
from scripts.train.g13_optim import save,load,complete,update
from scripts.train.g9_storage import disks,write,sha
ROOT=Path("/mnt/slwm-data/SLWM/g14")
ARMS=("unaligned","aligned","audiovisual")
SEEDS=(1401,1402,1403);CUTS=(128,640,1536)

def roots(name):return tuple(p.parent.parent/"g14"/name for p in disks())
def read(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
@lru_cache(maxsize=8)
def kernel(arm):
    @jax.jit
    def fit(p,m,v,data,indices,start,seed):
        def step(state,args):
            pp,mm,vv=state;ix,t=args;row={k:x[ix] for k,x in data.items()}
            loss,grad=jax.value_and_grad(objective)(pp,row,arm,t,seed)
            return update(pp,mm,vv,grad,t,1536,.0009),loss
        return jax.lax.scan(step,(p,m,v),(indices,start+1+jnp.arange(len(indices))))
    return fit
@lru_cache(maxsize=32)
def evaluator(arm,mode):
    return jax.jit(lambda p,a,l,n:tuple(jax.nn.softmax(x,axis=-1) for x in forward(p,a,l,n,arm,mode)[0]))
def posterior(p,data,arm,mode):
    out=[[] for _ in range(3)];n=len(data["y"])
    for i in range(0,n,128):
        ix=np.minimum(np.arange(i,i+128),n-1)
        arr=evaluator(arm,mode)(p,jnp.asarray(data["a"][ix]),jnp.asarray(data["l"][ix]),jnp.asarray(data["length"][ix]))
        for target,scores in zip(out,arr):target.append(np.asarray(scores)[:min(128,n-i)])
    return [np.concatenate(x) for x in out]

def scores(p,row):
    y=row["y"];speaker=row["speaker"];hat=np.stack([a.argmax(1) for a in p],1)
    hit=hat==y;ce=sum(-np.log(np.maximum(a[np.arange(len(y)),y[:,i]],1e-8)) for i,a in enumerate(p))
    def avg(a):return float(np.mean([a[speaker==v].mean() for v in np.unique(speaker)]))
    return dict(color=avg(hit[:,0]),letter=avg(hit[:,1]),digit=avg(hit[:,2]),joint=avg(hit.all(1)),nll=avg(ce),
                by_speaker={str(v):float(hit[speaker==v].all(1).mean()) for v in np.unique(speaker)})

def main(stage,root=ROOT,native=False,only=None,semantic=False):
    src=root/"data"/("semantic-utterances" if semantic else "utterances");work=root/"work"/("cue-semantic" if semantic else "cue");work.mkdir(exist_ok=True,parents=True);folders=(root/"checkpoints",) if native else roots("cue-semantic" if semantic else "cue")
    train=read(src/"train.npz");dev=read(src/"dev.npz");g=train["speaker"]
    _,iv,ct=np.unique(g,return_inverse=True,return_counts=True);weights=1./ct[iv];weights/=weights.sum()
    bank={k:jnp.asarray(train[k]) for k in ("a","l","length","alignment","y")};configs={}
    for seed in ((only,) if only else SEEDS):
        ix=np.random.default_rng(seed).choice(len(g),(1536,128),p=weights).astype("i4")
        for arm in (("aligned","audiovisual") if semantic else ARMS):
            tag=f"{arm}-{seed}";p=initial(seed);m=jax.tree.map(jnp.zeros_like,p);v=jax.tree.map(jnp.zeros_like,p);curve=[];start=0
            cfg=dict(seed=seed,arm=arm,train_sha=sha(src/"train.npz"),batch=128,steps=list(CUTS),parameters=sum(x.size for x in p.values()))
            for stop in CUTS:
                if complete(folders,tag,stop):
                    p,m,v,curve=load(folders,tag,stop,cfg);start=stop;continue
                if stage!="train":raise ValueError("Incomplete speech fit")
                tic=time.monotonic()
                for j in range(start,stop,16):
                    (p,m,v),loss=kernel(arm)(p,m,v,bank,jnp.asarray(ix[j:j+16]),jnp.int32(j),jnp.int32(seed))
                    if not np.isfinite(np.asarray(loss)).all():raise ValueError("Nonfinite cue loss")
                elapsed=time.monotonic()-tic;metric=scores(posterior(p,dev,arm,"normal"),dev)
                curve.append(dict(step=stop,seconds=elapsed,**metric));save(folders,tag,stop,p,m,v,cfg,curve);start=stop
                print(tag,stop,round(metric["joint"],4),round(metric["nll"],4),flush=True)
            best=min(curve,key=lambda r:(r["nll"],r["step"]));configs[tag]=dict(arm=arm,seed=seed,selected=best["step"],dev=best,curve=curve,parameters=cfg["parameters"],
                sha256=sha(folders[0]/tag/f"step-{best['step']:05d}"/"weights.npz"),accelerator=jax.devices()[0].device_kind)
    if stage=="train":
        write(work/(f"choices-{only}.json" if only else "choices.json"),configs);return
    lock=json.loads((work/"choices.json").read_bytes())
    chosen=min(lock,key=lambda k:(lock[k]["dev"]["nll"],k));results={};packed={}
    for role in ("train","dev","cal","held","fresh"):
        row=read(src/(role+".npz"))
        for tag,info in configs.items():
            if role=="train" and tag!=chosen:continue
            if info["sha256"]!=lock[tag]["sha256"]:raise ValueError("Selected voice checkpoint changed")
            p={k:jnp.asarray(v) for k,v in read(folders[0]/tag/f"step-{info['selected']:05d}"/"weights.npz").items()}
            modes=("normal","no_lips","no_audio","shifted_lips") if info["arm"]=="audiovisual" and role!="train" else ("normal",)
            for mode in modes:
                pp=posterior(p,row,info["arm"],mode);results.setdefault(tag,{}).setdefault(role,{})[mode]=scores(pp,row)
                if tag==chosen:
                    for h,arr in enumerate(pp):packed[f"{role}_{mode}_{h}"]=arr
        packed[role+"_indices"]=row["global_index"]
    np.savez_compressed(work/"posterior.npz",**packed)
    write(work/"scores.json",dict(selected=chosen,variants=results));print("Selected cue",chosen,results[chosen]["fresh"]["normal"],flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("stage",choices=("train","score"));a.add_argument("--semantic",action="store_true");a.add_argument("--native",action="store_true");a.add_argument("--seed",type=int)
    p=a.parse_args();root=Path("/content/g14") if p.native else ROOT
    if p.native and not any("v6" in d.device_kind.lower() for d in jax.devices()):raise ValueError("Expected v6e")
    jax.config.update("jax_compilation_cache_dir",str(root/"cache"));main(p.stage,root,p.native,p.seed,p.semantic)
