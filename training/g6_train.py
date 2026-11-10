"""Same-exposure multimodal scaling with fixed LR horizon, restartable rungs, and timed TPU updates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from models.g4_shared import Spec
from models.g5_residual import initialize, objective
from training.g4_train import checkpoint, restore, sha_file, write
from training.g5_batches import JointData,stack
from training.g6_checkpoint import export

CFG=Path("configs/t6/tpu_scale.json")
SOURCES=("models/g4_shared.py","models/g5_shared.py","models/g5_residual.py","training/g4_train.py",
         "training/g5_batches.py","training/g5_priors.py","training/g6_train.py","training/g6_checkpoint.py",
         "configs/t5/joint_world.json",str(CFG))


def tpu_device(generation):
    dev=jax.devices()[0]
    kind=dev.device_kind.lower()
    good=("v5 lite" in kind or "v5e" in kind) if generation=="v5e" else ("v6" in kind or "trillium" in kind)
    if dev.platform!="tpu" or not good or len(jax.devices())!=1:
        raise RuntimeError(f"Expected a single {generation} TPU, got {jax.devices()}")
    return dev


def run_name(generation,scale,core,seed):
    return f"g6-{generation}-{scale}-{core}-{seed}"


def compiled_steps(spec,opt,cal,micro,batch):
    if batch%micro:
        raise ValueError("Microbatch must divide the effective batch")
    accum=batch//micro
    def update(state,examples):
        p,ost,count=state
        def loss(w,b):
            return objective(w,spec,b,cal)
        grad=jax.value_and_grad(loss,has_aux=True)
        small=jax.tree.map(lambda x:x.reshape(accum,micro,*x.shape[1:]),examples)
        def add(carry,items):
            summed,metrics=carry
            (_,m),g=grad(p,items)
            return (jax.tree.map(jnp.add,summed,g),metrics+m),None
        (g,metrics),_=jax.lax.scan(add,(jax.tree.map(jnp.zeros_like,p),jnp.zeros(4)),small)
        updates,ost=opt.update(jax.tree.map(lambda v:v/accum,g),ost,p)
        return (optax.apply_updates(p,updates),ost,count+1),metrics/accum
    def block(state,examples):
        state,m=jax.lax.scan(update,state,examples)
        return state,m.mean(axis=0)
    return jax.jit(block,donate_argnums=(0,))


def link_exact(path,dest):
    if dest.exists():
        if not dest.samefile(path) and sha_file(dest)!=sha_file(path):
            raise ValueError(f"Conflicting frozen stage {dest}")
    else:
        os.link(path,dest)


def fit(a):
    cfg=json.loads(CFG.read_bytes())
    dev=tpu_device(a.generation)
    spec=Spec(a.core,*cfg["scales"][a.scale],dtype="bfloat16")
    target=a.until
    batch=cfg["batch"]; horizon=cfg["schedule_steps"]
    if target%a.scan or a.scan<1 or target>horizon or target<=0:
        raise ValueError("Invalid update target")
    allowed={n for sc,n in cfg["stages"] if sc==a.scale}
    if not a.probe and target not in allowed:
        raise ValueError("Unregistered training stage")
    # The loader has already verified the frozen archive bytes. Its per-family cards bind the data identity.
    run=run_name(a.generation,a.scale,a.core,a.seed)
    folder=a.output/run
    stage=folder/f"step-{target:06d}"
    if not a.probe and (stage/"complete.json").is_file():
        print("Already trained",run,target,flush=True)
        return
    data=JointData(a.data,"train",verify=not a.checked)
    prior=a.data/"prior.npz"
    if sha_file(prior)!=cfg["prior_sha256"]:
        raise ValueError("Training-only calibration changed")
    with np.load(prior) as f:
        cal={k:jax.device_put(f[k]) for k in ("coef","contact")}
    lr=.0006*(256/cfg["learning_rate_reference_width"])**.5
    rate=lambda t:lr*jnp.minimum(1.,(t+1)/16)*(.1+.45*(1+jnp.cos(jnp.pi*t/horizon)))
    opt=optax.chain(optax.clip_by_global_norm(1),optax.adamw(rate,b1=.9,b2=.95,weight_decay=.01,
                    mask=lambda p:jax.tree.map(lambda w:w.ndim>=2,p)))
    p=initialize(spec,a.seed)
    count=sum(v.size for v in p.values())
    if count!=cfg["expected_parameters"][a.scale]:
        raise ValueError(f"Unexpected parameter count {count}")
    card={"run":run,"variant":"physical-residual","spec":asdict(spec),"params":count,"seed":a.seed,
          "data_seed":data.cfg["seed"],"data":data.identity,"batch":batch,"steps":horizon,"lr":lr,
          "prior_sha256":cfg["prior_sha256"],"families_available":list(map(len,data.pools)),
          "tpu":a.generation,"code":{s:sha_file(Path(s)) for s in SOURCES}}
    identity=hashlib.sha256(json.dumps(card,sort_keys=True).encode()).hexdigest()
    if not a.probe:
        folder.mkdir(exist_ok=True,parents=True)
        fp=folder/"config.json"
        if fp.is_file() and json.loads(fp.read_bytes())!=card:
            raise ValueError("Existing run differs from the frozen training recipe")
        if not fp.is_file():
            write(fp,card)
    state=(p,opt.init(p),jnp.int32(0))
    del p
    if not a.probe and (folder/"resume.npz").is_file():
        state=restore(folder/"resume.npz",state,identity)
    start=int(state[-1])
    if start>target:
        raise ValueError("The requested immutable stage is missing behind the latest training state")
    compiled=compiled_steps(spec,opt,cal,a.micro,batch)
    def sample(i):
        return stack([data.requests(j,batch) for j in range(i,i+a.scan)])
    speeds=[]; cold=None
    begin=time.monotonic()
    print(json.dumps({"starting":run,"device":dev.device_kind,"params":count,"from":start,"to":target,"micro":a.micro,
                      "effective_batch":batch,"probe":a.probe}),flush=True)
    if start<target:
        with ThreadPoolExecutor(max_workers=1) as io:
            prefetched=io.submit(sample,start)
            for s in range(start,target,a.scan):
                before=time.monotonic()
                block=jax.device_put(prefetched.result())
                if s+a.scan<target:
                    prefetched=io.submit(sample,s+a.scan)
                state,metrics=compiled(state,block)
                jax.block_until_ready((state,metrics))
                val=np.asarray(metrics); took=time.monotonic()-before
                if not np.isfinite(val).all():
                    raise FloatingPointError((run,s,val))
                if s==start:
                    cold=took
                else:
                    speeds.append(batch*a.scan/took)
                done=s+a.scan
                progress={"run":run,"step":done,"target_stage":target,"schedule_steps":horizon,"device":dev.device_kind,
                          "micro":a.micro,"examples_per_second":batch*a.scan/took,"warm_examples_per_second":float(np.median(speeds)) if speeds else None,
                          "first_block_seconds":cold,"wall_seconds_since_resume":time.monotonic()-begin,
                          "stream_positions":done*batch*64,"loss":dict(zip(("total","crossmodal","contact","action"),val.tolist()))}
                if not a.probe:
                    write(folder/"progress.json",progress)
                    if done==target or done%512==0:
                        checkpoint(folder,state,identity,data.stats)
                    if done==start+a.scan or done%128==0:
                        print(json.dumps(progress),flush=True)
    if a.probe:
        write(a.output/f"bench-{a.generation}-{a.scale}-{a.core}-{a.micro}.json",progress)
        print(json.dumps(progress),flush=True)
        return
    # Recover cleanly if the optimizer had reached its rung before the inference export completed.
    if not (folder/"progress.json").exists() or json.loads((folder/"progress.json").read_bytes())["step"]!=target:
        write(folder/"progress.json",{"run":run,"step":target,"target_stage":target,"recovered":True,"device":dev.device_kind})
    stage.mkdir(exist_ok=True,parents=True)
    for key in ("resume.npz","config.json","progress.json"):
        link_exact(folder/key,stage/key)
    export(stage/"resume.npz",prior,stage/"inference.npz")
    hashes={k:sha_file(stage/k) for k in ("resume.npz","inference.npz","config.json","progress.json")}
    write(stage/"complete.json",{"run":run,"step":target,"identity":identity,"device":dev.device_kind,"sha256":hashes})
    print("Frozen",stage,hashes["inference.npz"],flush=True)


if __name__=="__main__":
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument("--data",required=True,type=Path); q.add_argument("--output",required=True,type=Path)
    q.add_argument("--generation",required=True,choices=["v5e","v6e"]); q.add_argument("--core",required=True,choices=["hybrid","gpt2"])
    q.add_argument("--scale",required=True,choices=["anchor","larger"]); q.add_argument("--seed",type=int,required=True)
    q.add_argument("--until",type=int,required=True); q.add_argument("--micro",type=int,required=True)
    q.add_argument("--scan",type=int,default=4); q.add_argument("--probe",action="store_true"); q.add_argument("--checked",action="store_true")
    fit(q.parse_args())
