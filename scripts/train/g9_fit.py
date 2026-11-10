"""Batched training, deterministic episode streams and atomic, resumable optimizer snapshots."""
import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from models.g9_shared import Spec, objective, parameters
from models.g9_tabletop import LENGTH, mix


def save_json(path,doc):
    tmp=path.with_suffix(".new");tmp.write_text(json.dumps(doc,sort_keys=True,indent=2)+"\n");tmp.replace(path)


def weights(path,values):
    with path.open("wb") as out:
        np.savez(out,**{k:np.asarray(v) for k,v in values.items()})
        out.flush();os.fsync(out.fileno())
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def numpy_weights(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def bank_data(path):
    with np.load(path,allow_pickle=False) as f:
        lengths=f["lengths"].astype(int)
        if (lengths<=LENGTH).any():raise ValueError("Every training recording must have a full causal window")
        offsets=np.cumsum(np.r_[0,lengths[:-1]])
        starts=np.concatenate([o+np.arange(n-LENGTH,dtype=np.int32) for o,n in zip(offsets,lengths)]).astype(np.int32)
        return {k:jax.device_put(f[k]) for k in ("a","v","n","w")},jax.device_put(starts),list(map(str,f["names"]))


def recipe_group(group):
    def cell(size,core,field,act,seed,replay=False):
        return (size,core,field,act,seed,replay)
    if group=="smoke":return [cell("smoke","hybrid",128,"gelu",911)]
    if group=="pilot":
        return [cell("pilot",core,field,act,seed) for seed in (911,912) for field in (128,512)
                for act in ("gelu","snake") for core in ("hybrid","gpt2")]+[
                cell("pilot",core,512,"gelu",seed,True) for seed in (911,912) for core in ("hybrid","gpt2")]+[
                cell("pilot","gated",512,"gelu",seed) for seed in (911,912)]
    if group.startswith("confirm"):
        return [cell("confirm",core,field,"gelu",seed) for seed in (911,912) for field in (128,512)
                for core in ("hybrid","gpt2") if group=="confirm" or group=="confirm-"+str(seed)]
    raise ValueError(group)


def tag(cell):
    size,core,field,act,seed,replay=cell
    return f"{size}-{core}-z{field}-{act}-s{seed}-"+("replay" if replay else "fresh")


@lru_cache(maxsize=48)
def compiled(spec,replay,count,goal):
    @jax.jit
    def update(p,first,second,bank,starts,step,seed):
        r=mix(bank,starts,step,seed,replay,count)
        (loss,metrics),grad=jax.value_and_grad(objective,has_aux=True)(p,spec,r)
        norm=jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree.leaves(grad)))
        scale=jnp.minimum(1.,1./(norm+1e-8))
        grad=jax.tree.map(lambda x:x*scale,grad)
        first=jax.tree.map(lambda m,g:.9*m+.1*g,first,grad)
        second=jax.tree.map(lambda v,g:.98*v+.02*g*g,second,grad)
        t=step+1
        warm=min(100,max(8,goal//12))
        lr=3e-4*jnp.minimum(t/warm,1.)*(.1+.9*.5*(1+jnp.cos(jnp.pi*jnp.maximum(0,t-warm)/max(1,goal-warm))))
        a=1-jnp.power(.9,t);b=1-jnp.power(.98,t)
        p=jax.tree.map(lambda x,m,v:x-lr*(m/a/(jnp.sqrt(v/b)+1e-8)+.01*x),p,first,second)
        return p,first,second,jnp.concatenate([metrics,jnp.array([loss,norm,r["keep"].all(1).sum()])])
    return jax.jit(update.__wrapped__,donate_argnums=(0,1,2))


def one(cell,bank,starts,names,dest,count=128):
    size,core,field,act,seed,replay=cell
    d,layers,goal={"smoke":(128,2,5),"pilot":(512,4,384),"confirm":(896,12,4096)}[size]
    job=dest/tag(cell);job.mkdir(parents=True,exist_ok=True)
    if (job/"done.json").is_file():return
    spec=Spec(core,d,layers,field,act)
    records=sorted(p for p in job.glob("step-*") if p.is_dir() and (p/"receipt.json").exists())
    if records:
        last=records[-1];step=int(last.name.split("-")[1]);p=jax.tree.map(jnp.asarray,numpy_weights(last/"weights.npz"))
        saved=numpy_weights(last/"optimizer.npz")
        m={k:jnp.asarray(saved["m_"+k]) for k in p};v={k:jnp.asarray(saved["v_"+k]) for k in p}
        del saved
    else:
        step=0;p=parameters(spec,seed)
        m=jax.tree.map(jnp.zeros_like,p);v=jax.tree.map(jnp.zeros_like,p)
    num=sum(x.size for x in p.values())
    run=dict(tag=tag(cell),spec=asdict(spec),seed=seed,replay=replay,batch=count,sequence=LENGTH,
             trainable_parameters=num,training_recordings=names,goal_steps=goal,backend=jax.default_backend())
    save_json(job/"recipe.json",run)
    train=compiled(spec,replay,count,goal)
    start=time.perf_counter();anchor=step;last_time=start
    while step<goal:
        p,m,v,metric=train(p,m,v,bank,starts,jnp.int32(step),jnp.int32(seed))
        step+=1
        if step%32==0 or step==goal or step==1:
            vals=np.asarray(metric)
            if not np.all(np.isfinite(vals)):raise FloatingPointError(f"{tag(cell)} nonfinite at {step}")
            now=time.perf_counter()
            entry=dict(step=step,policy=float(vals[0]),reconstruction=float(vals[1]),forecast=float(vals[2]),
                       loss=float(vals[3]),gradient_norm=float(vals[4]),four_present_this_update=int(vals[5]),
                       wall_since_start_s=now-start,warm_updates_per_s=(32/(now-last_time) if step>=64 else None))
            with (job/"train.jsonl").open("a") as f:f.write(json.dumps(entry)+"\n")
            save_json(dest/"progress.json",dict(job=tag(cell),**entry,parameters=num))
            if step==1 or step%128==0 or step==goal:print(tag(cell),step,round(entry["loss"],4),round(now-start,2),flush=True)
            last_time=now
        if step==goal or (size=="confirm" and step%1024==0):
            final=job/f"step-{step:05d}";tmp=job/f".writing-{step}"
            if tmp.exists():shutil.rmtree(tmp)
            tmp.mkdir()
            sha=weights(tmp/"weights.npz",p)
            opt=weights(tmp/"optimizer.npz",{**{"m_"+k:x for k,x in m.items()},**{"v_"+k:x for k,x in v.items()}})
            receipt=dict(step=step,inference_sha256=sha,optimizer_sha256=opt,wall_this_process_s=time.perf_counter()-start,
                         started_from=anchor,**run)
            save_json(tmp/"receipt.json",receipt);tmp.rename(final)
            if step==goal:save_json(job/"done.json",receipt)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data",required=True,type=Path);ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--group",choices=["smoke","pilot","confirm","confirm-911","confirm-912"],required=True)
    ap.add_argument("--only",nargs="*");ap.add_argument("--cache",type=Path)
    ap.add_argument("--batch",type=int,default=128)
    args=ap.parse_args()
    if args.cache:jax.config.update("jax_compilation_cache_dir",str(args.cache))
    if args.group!="smoke" and args.batch!=128:raise ValueError("Registered batch is 128")
    bank,starts,names=bank_data(args.data/"starss-train.npz")
    for cell in recipe_group(args.group):
        if not args.only or tag(cell) in args.only:one(cell,bank,starts,names,args.out,args.batch)


if __name__=="__main__":main()
