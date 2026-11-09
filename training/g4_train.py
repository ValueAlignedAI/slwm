"""Compiled and resumable JAX training with fixed-negative microbatches and bounded host transfers."""
import argparse
from dataclasses import asdict
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from models.g4_shared import Spec, initialize, objective


CFG=Path("configs/t4/crossmodal_tpu.json")
KEYS=("a","v","n","x","ok","has","family","cap")
PROFILES={"gpt2-cross":("gpt2","cross",4),"gpt2-self":("gpt2","self",4),
          "signal-cross":("signal","cross",4),"signal-self":("signal","self",4),
          "sparse-cross":("grouped","cross",4),"gpt2-matched":("gpt2","cross",1)}


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)


def sha_file(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def backend(required):
    dev=jax.devices()[0]
    if required!="any" and dev.platform!=required:
        raise RuntimeError(f"Requested {required}, got {dev.platform} {dev.device_kind}")
    return dev


class Data:
    def __init__(self,root,role,limit=0,verify=True):
        self.dir=root/role
        self.card=read(self.dir/"card.json")
        def load(k):
            file=self.dir/f"{k}.npy"
            if verify and sha_file(file)!=self.card["sha256"][k]:
                raise ValueError(f"Changed G4 input: {file}")
            return np.load(file,mmap_mode="r",allow_pickle=False)
        self.host={k:load(k) for k in KEYS}
        self.group=load("group")
        self.uid=load("uid")
        if limit:
            # A deterministic stratified subset is used exclusively for the mechanical smoke.
            indices=np.concatenate([np.flatnonzero(self.host["family"]==f)[:limit] for f in range(7)])
            self.host={k:v[indices] for k,v in self.host.items()}
            self.group,self.uid=self.group[indices],self.uid[indices]
        self.pools=[np.flatnonzero(self.host["family"]==f) for f in range(7)]
        self.device={k:jax.device_put(np.array(v,copy=True)) for k,v in self.host.items()}
        jax.block_until_ready(self.device)

    def stats(self):
        data=np.asarray(self.host["n"][self.pools[4]],np.float32)
        mask=self.host["ok"][self.pools[4]].astype(bool)
        values=data[mask]
        return np.stack([values.mean(0),np.maximum(values.std(0),.025)]).astype(np.float32)


class Sampler:
    def __init__(self,data,batch,seed,mixture):
        self.data,self.batch,self.seed=data,batch,seed
        self.counts=np.array(mixture)*batch//16
        if self.counts.sum()!=batch:
            raise ValueError("Batch must preserve the registered source proportions")
        self.orders={}

    def sample(self,step):
        chunks=[]
        for fam,count in enumerate(self.counts):
            if not count: continue
            pool=self.data.pools[fam]
            absolute=step*count+np.arange(count)
            sample=np.empty(count,np.int32)
            for epoch in np.unique(absolute//len(pool)):
                key=fam,int(epoch)
                if key not in self.orders:
                    self.orders={k:v for k,v in self.orders.items() if k[0]!=fam}
                    self.orders[key]=np.random.default_rng([self.seed,fam,int(epoch)]).permutation(pool)
                use=absolute//len(pool)==epoch
                sample[use]=self.orders[key][absolute[use]%len(pool)]
            chunks.append(sample)
        rng=np.random.default_rng([self.seed,step,953])
        ids=np.concatenate(chunks)
        rng.shuffle(ids)
        present=self.data.host["has"][ids].astype(bool)
        priorities=np.where(present,rng.random(present.shape),-1)
        first=priorities.argmax(1)
        priorities[np.arange(len(ids)),first]=-2
        second=np.where(present.sum(1)>1,priorities.argmax(1),first)
        erased=(rng.random((len(ids),2))<.05)&(present.sum(1)>1)[:,None]
        views=np.column_stack([first,second,erased]).astype(np.int32)
        return ids,views

    def block(self,step,count):
        batches=[self.sample(s) for s in range(step,step+count)]
        return tuple(np.stack(a) for a in zip(*batches))


def make_block(spec,policy,micro,effective,negatives,lr,horizon):
    if effective%micro or micro%negatives:
        raise ValueError("Microbatch must divide the effective batch and contain complete contrastive groups")
    acc=effective//micro
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.scale_by_adam(b1=.9,b2=.95))
    @partial(jax.jit,donate_argnums=(0,))
    def compiled(state,data,indices,views,stats):
        def step(state,items):
            p,opt,count=state
            ix,vs=items
            def gradient(pair):
                ids,v=pair
                batch=jax.tree.map(lambda x:x[ids],data)
                return jax.value_and_grad(objective,has_aux=True)(p,spec,batch,v,stats,policy,negatives)
            if acc==1:
                (_,metrics),grads=gradient((ix,vs))
            else:
                def add(carry,pair):
                    accum,met=carry
                    (_,m),g=gradient(pair)
                    return (jax.tree.map(jnp.add,accum,g),met+m),None
                zeros=jax.tree.map(jnp.zeros_like,p)
                (grads,metrics),_=jax.lax.scan(add,(zeros,jnp.zeros(4)),(ix.reshape(acc,micro),vs.reshape(acc,micro,4)))
                grads=jax.tree.map(lambda x:x/acc,grads)
                metrics=metrics/acc
            updates,opt=optimizer.update(grads,opt,p)
            warm=jnp.minimum(1.,(count+1)/16)
            rate=lr*warm*(.1+.9*(1+jnp.cos(jnp.pi*jnp.minimum(count/horizon,1)))/2)
            p=jax.tree.map(lambda w,g:w-rate*(g+(.01*w if w.ndim>=2 else 0)),p,updates)
            return (p,opt,count+1),metrics
        state,met=jax.lax.scan(step,state,(indices,views))
        return state,met.mean(0)
    return optimizer,compiled


def config_spec(cfg,scale,profile,precision):
    core,policy,expansion=PROFILES[profile]
    width,depth=cfg["scales"][scale]
    return Spec(core,width,depth,precision,expansion),policy


def checkpoint(out,state,manifest,stats):
    p,opt,count=jax.device_get(state)
    step=int(count)
    opt_arrays=jax.tree.leaves(opt)
    payload={**{f"p/{k}":v for k,v in p.items()},**{f"o/{i}":v for i,v in enumerate(opt_arrays)},
             "step":np.int64(step),"stats":np.asarray(stats),"identity":np.array(manifest)}
    path=out/"resume.npz"
    with (out/"resume.tmp").open("wb") as f:
        np.savez(f,**payload)
    (out/"resume.tmp").replace(path)
    return path,step


def restore(path,state,identity):
    p,opt,count=state
    with np.load(path,allow_pickle=False) as f:
        if str(f["identity"])!=identity:
            raise ValueError("Checkpoint config, data, or code identity changed")
        p={k:jax.device_put(f[f"p/{k}"]) for k in p}
        treedef=jax.tree.structure(opt)
        opt=jax.tree.unflatten(treedef,[jax.device_put(f[f"o/{i}"]) for i in range(treedef.num_leaves)])
        return p,opt,jnp.int32(f["step"])


def archive(run,step,files,spool):
    if spool:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/train"))
        from colab_artifacts import capture
        capture(spool,run,step,files)


def run(args):
    dev=backend(args.backend)
    cfg=read(CFG)
    scale,profile=args.scale,args.profile
    spec,policy=config_spec(cfg,scale,profile,args.precision)
    tag=f"g4-{args.label}-{scale}-{profile}-{args.seed}"
    out=args.output/tag
    out.mkdir(parents=True,exist_ok=True)
    data=Data(args.data,"train",args.limit,verify=not args.checked_data)
    stats=data.stats()
    effective=args.batch
    sampler=Sampler(data,effective,cfg["seed"],cfg["mixture_16"])
    lr=.0006*(256/spec.width)**.5
    opt,step_block=make_block(spec,policy,args.micro,effective,args.negatives,lr,args.steps)
    p=initialize(spec,args.seed)
    state=(p,opt.init(p),jnp.int32(0))
    card={"name":tag,"profile":profile,"policy":policy,"spec":asdict(spec),"seed":args.seed,
          "data_seed":cfg["seed"],"batch":effective,"contrastive_group":args.negatives,"steps":args.steps,
          "limit_per_family":args.limit,"lr":lr,"train_data":sha_file(data.dir/"card.json"),
          "sources":{path:sha_file(Path(path)) for path in ("models/g4_shared.py","training/g4_train.py",str(CFG))},
          "params":sum(x.size for x in p.values()),"label":args.label}
    identity=hashlib.sha256(json.dumps(card,sort_keys=True).encode()).hexdigest()
    write(out/"config.json",card)
    if (out/"resume.npz").exists():
        state=restore(out/"resume.npz",state,identity)
    start=int(state[-1])
    start_clock=time.monotonic()
    timing=read(out/"timing.json") if (out/"timing.json").exists() else {"compiled_s":0,"warm_s":0,"warm_steps":0}
    durations=[]
    for s in range(start,args.steps,args.scan):
        size=min(args.scan,args.steps-s)
        ids,views=sampler.block(s,size)
        before=time.monotonic()
        state,metrics=step_block(state,data.device,ids,views,stats)
        metrics=np.asarray(jax.block_until_ready(metrics))
        elapsed=time.monotonic()-before
        if not np.isfinite(metrics).all():
            raise FloatingPointError((tag,s,metrics))
        durations.append(elapsed)
        if s==start:
            timing["compiled_s"]+=elapsed
        else:
            timing["warm_s"]+=elapsed
            timing["warm_steps"]+=size
        done=s+size
        result={"run":tag,"step":done,"target":args.steps,"device":dev.device_kind,"micro":args.micro,
                "metrics":dict(zip(("objective","supervised","alignment","dynamics"),metrics.tolist())),
                "seconds_this_block":elapsed,"seconds_this_session":time.monotonic()-start_clock,
                "examples_per_second":effective*size/elapsed,"exposure_positions":done*effective*64}
        write(out/"progress.json",result)
        if done%args.save_every==0 or done==args.steps:
            path,_=checkpoint(out,state,identity,stats)
            write(out/"timing.json",timing)
            archive(tag,done,{"resume.npz":path,"config.json":out/"config.json","progress.json":out/"progress.json"},args.spool)
            print(json.dumps(result),flush=True)
    if args.steps:
        write(out/"complete.json",{"run":tag,"step":args.steps,"weight_file_sha256":sha_file(out/"resume.npz"),
                                  "timing":timing,"params":card["params"],"device":dev.device_kind})


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--profile",choices=PROFILES,default="signal-cross")
    p.add_argument("--scale",choices=["pilot","medium","anchor"],default="pilot")
    p.add_argument("--label",default="registered"); p.add_argument("--seed",type=int,default=81)
    p.add_argument("--steps",type=int,default=256); p.add_argument("--batch",type=int,default=256)
    p.add_argument("--micro",type=int,default=256); p.add_argument("--scan",type=int,default=8)
    p.add_argument("--negatives",type=int,default=32); p.add_argument("--save-every",type=int,default=128)
    p.add_argument("--backend",choices=["cpu","gpu","tpu","any"],default="tpu")
    p.add_argument("--precision",choices=["bfloat16","float32"],default="bfloat16")
    p.add_argument("--limit",type=int,default=0); p.add_argument("--spool",type=Path)
    p.add_argument("--checked-data",action="store_true",help="Supervisor has already verified the immutable local arrays")
    return p


if __name__=="__main__":
    run(parser().parse_args())
