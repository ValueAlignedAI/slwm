"""Two-stage supervised transfer with immutable adapters and original parent identity."""
import argparse
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from models.g9_shared import Spec
from models.g11_instruction import initialize,loss
from scripts.train.g9_fit import numpy_weights,save_json
from scripts.train.g10_fit import adam,restore,snapshot

TREATMENTS={"paired":"correct","deranged":"wrong","absent":"none"}


def read_bank(banks,fold,role,physical=True):
    stem=f"{fold}-{role}"
    with np.load(banks/(stem+".npz"),allow_pickle=False) as f:
        d={k:f[k] for k in f.files if k not in ("names","semantic")};sem=f["semantic"]
    n=len(d["cued"])
    if physical:
        rec=json.loads((banks/(stem+"-effects-receipt.json")).read_text());fid=banks/(stem+"-effects.npz")
        if rec["base_sha256"]!=json.loads((banks/(stem+".json")).read_text())["sha256"]:raise ValueError("Physical targets do not match causal sample IDs")
        with np.load(fid) as f:d.update({k:f[k] for k in f.files})
    else:
        d.update(geometry=np.zeros((n,201),np.float32),effect=np.zeros((n,3),np.float32),valid=np.zeros(n,bool),strict=np.zeros(n,bool),
                 small_words=np.zeros((n,128),np.int8),wrong_small=np.zeros((n,128),np.int8))
    return d,sem


def parent(runs,core,seed):
    path=runs/f"context-{core}-variable-s{seed}"/"step-05120"
    meta=json.loads((path/"receipt.json").read_text())
    with (path/"weights.npz").open("rb") as f:
        if hashlib.file_digest(f,"sha256").hexdigest()!=meta["inference_sha256"]:raise ValueError("Parent identity mismatch")
    return meta,jax.tree.map(jnp.asarray,numpy_weights(path/"weights.npz")),Spec(**meta["spec"])


@lru_cache(maxsize=48)
def kernel(s,mode,nv,nn,physical):
    def step(theta,m,v,base,bank,cued,bg,geo,sem,scale,i,seed):
        rng=jax.random.fold_in(jax.random.PRNGKey(seed+110781),i);a,b,c=jax.random.split(rng,3)
        ix=jnp.r_[cued[jax.random.randint(a,(64,),0,len(cued))],bg[jax.random.randint(b,(32,),0,len(bg))],geo[jax.random.randint(c,(32,),0,len(geo))]]
        rows=jax.tree.map(lambda x:x[ix],bank)
        (value,metrics),g=jax.value_and_grad(loss,has_aux=True)(theta,base,s,rows,mode,nv,nn,sem,scale,physical)
        theta,m,v,norm=adam(theta,m,v,g,i,1024,5e-4)
        return theta,m,v,jnp.r_[value,metrics,norm]
    return jax.jit(step,donate_argnums=(0,1,2))


def run(banks,parents,out,fold,seeds,cores,treatments,phase):
    physical=phase=="joint";values,sem=read_bank(banks,fold,"train",physical);vocab=json.loads((banks/(fold+"-vocabulary.json")).read_text())
    nv=len(vocab["verb"]);nn=len(vocab["noun"])
    if nv>64 or nn>256:raise ValueError("Ontology exceeds registered head capacity")
    ci=np.flatnonzero(values["cued"]).astype("i4");bg=np.flatnonzero(~values["cued"]).astype("i4");good=np.flatnonzero(values["valid"]).astype("i4")
    if physical and len(good)<20:raise ValueError("Insufficient training effect measurements")
    gi=good if physical else ci;scale=np.maximum(values["effect"][good].std(0),.065) if physical else np.ones(3,np.float32)
    wanted=("a","v","n","ac","wc","aw","ww","ah","wh","small_words","wrong_small","geometry","effect","valid","verb","noun","open_noun")
    dev=jax.tree.map(jax.device_put,{k:values[k] for k in wanted});del values
    ci,bg,gi,sem,scale=map(jax.device_put,(ci,bg,gi,sem,scale));until=1024 if physical else 256
    for seed in seeds:
        for core in cores:
            meta,base,s=parent(parents,core,seed)
            for treatment in treatments:
                tag=f"g11-{fold}-{core}-{treatment}-s{seed}";job=out/tag
                if (job/f"step-{until:05d}/receipt.json").exists():continue
                recipe=dict(tag=tag,fold=fold,core=core,seed=seed,treatment=treatment,goal_steps=1024,spec=asdict(s),
                    parent_sha256=meta["inference_sha256"],active_parent_parameters=meta["trainable_parameters"],source_bank_sha256=json.loads((banks/(fold+"-train.json")).read_text())["sha256"],
                    vocabulary_sha256=hashlib.sha256((banks/(fold+"-vocabulary.json")).read_bytes()).hexdigest(),phase="G11 transfer")
                i,theta,m,v=restore(job,lambda:initialize(s,seed),recipe);anchor=i
                if physical and i<256:raise ValueError("A joint fit must resume the exact action rung")
                runmeta={**recipe,"backend":jax.default_backend(),"trainable_parameters":sum(x.size for x in theta.values()),"measured_training_effects":len(good)}
                t=last=time.monotonic()
                while i<until:
                    theta,m,v,stat=kernel(s,TREATMENTS[treatment],nv,nn,physical)(theta,m,v,base,dev,ci,bg,gi,sem,scale,jnp.int32(i),jnp.int32(seed+(0 if fold=="A" else 1777)));i+=1
                    if i==anchor+1 or i%32==0 or i==until:
                        vals=np.asarray(stat);now=time.monotonic()
                        if not np.isfinite(vals).all():raise FloatingPointError((tag,i,vals))
                        row=dict(tag=tag,step=i,loss=float(vals[0]),verb=float(vals[1]),noun=float(vals[2]),semantic=float(vals[3]),effect=float(vals[4]),
                                 gradient=float(vals[5]),seconds=now-t,updates_s=32/(now-last) if i>=anchor+64 else None)
                        with (job/"train.jsonl").open("a") as file:file.write(json.dumps(row)+"\n")
                        save_json(out/"progress.json",row);last=now
                    if i in (256,640,1024):
                        snapshot(job,i,theta,m,v,runmeta,t,anchor);print(tag,i,round(float(vals[0]),3),round(time.monotonic()-t,1),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("banks",type=Path);p.add_argument("parents",type=Path);p.add_argument("out",type=Path);p.add_argument("phase",choices=("action","joint"))
    p.add_argument("--folds",nargs="+",default=["A","B"]);p.add_argument("--seeds",nargs="+",type=int,default=[1001,1002,1003]);p.add_argument("--cores",nargs="+",default=["hybrid","gpt2","gated"])
    p.add_argument("--treatments",nargs="+",default=list(TREATMENTS));p.add_argument("--cache");a=p.parse_args()
    if a.cache:jax.config.update("jax_compilation_cache_dir",a.cache)
    for f in a.folds:run(a.banks,a.parents,a.out,f,a.seeds,a.cores,a.treatments,a.phase)
