"""Fork exact G6 Adam states and run fixed-schedule continuations with measured, synchronized time."""
import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from models.g4_shared import Spec
from training.g4_train import checkpoint,sha_file,write
from training.g5_batches import JointData,stack
from training.g6_train import compiled_steps,tpu_device,link_exact
from training.g6_checkpoint import export
from training.g7_batches import ContinuationData

CFG=Path("configs/t7/study.json")


def restore(file,identity,opt):
    with np.load(file,allow_pickle=False) as z:
        if str(z["identity"])!=identity:raise ValueError("Optimizer identity mismatch")
        params={k[2:]:jax.device_put(z[k]) for k in z.files if k.startswith("p/")}
        tree=jax.tree.structure(jax.eval_shape(opt.init,params))
        return params,jax.tree.unflatten(tree,[jax.device_put(z[f"o/{i}"]) for i in range(tree.num_leaves)]),jnp.int32(z["step"])


def disk_identity(d):
    return hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()


def run(a):
    cfg=json.loads(CFG.read_bytes());device=tpu_device(a.generation);start_at=cfg["parent_steps"];end=cfg["end_steps"]
    if a.until not in cfg["rungs"] or (a.until-start_at)%a.scan:raise ValueError("Unknown continuation rung")
    old=f"g6-v5e-{a.scale}-{a.core}-{a.seed}"; ancestor=a.parents/old/"step-008192"
    card=json.loads((ancestor/"config.json").read_bytes());receipt=json.loads((ancestor/"verified.json").read_bytes())
    spec=Spec(**card["spec"]); batch=cfg["effective_batch"]
    if card["run"]!=old or receipt["step"]!=start_at or len(set(receipt["disk_uuids"]))!=2:raise ValueError("Unverified parent")
    name=f"g7-{a.generation}-{a.scale}-{a.core}-{a.seed}-{a.arm}";folder=a.out/name;stage=folder/f"step-{a.until:06d}"
    folder.mkdir(exist_ok=True,parents=True)
    if (stage/"complete.json").exists():return
    with np.load(a.data/"prior.npz") as f:cal={k:jax.device_put(f[k]) for k in ("coef","contact")}
    if sha_file(a.data/"prior.npz")!=card["prior_sha256"]:raise ValueError("Physical calibration mismatch")
    base=JointData(a.data,"train",verify=False)
    if base.identity!=card["data"]:raise ValueError("Different replay source")
    data=ContinuationData(base,a.fresh if a.arm=="fresh" else None)
    code={k:sha_file(Path(k)) for k in ("training/g7_continue.py","training/g7_batches.py","training/g6_train.py","models/g5_residual.py","models/g5_shared.py",str(CFG))}
    dc=dict(card,run=name,steps=end,parent=old,parent_optimizer=receipt["sha256"]["resume.npz"],data=data.identity,
            base_data=base.identity,continuation_arm=a.arm,continuation_code=code,lr=cfg["common_peak_lr"],tpu=a.generation)
    identity=disk_identity(dc)
    if (folder/"config.json").exists() and json.loads((folder/"config.json").read_bytes())!=dc:raise ValueError("Continuation recipe changed")
    if not (folder/"config.json").exists():write(folder/"config.json",dc)
    warm=cfg["warmup_continuation_steps"]
    def rate(at):
        x=jnp.maximum(at-start_at,0)
        return cfg["common_peak_lr"]*jnp.minimum(1.,(x+1)/warm)*(.1+.45*(1+jnp.cos(jnp.pi*x/(end-start_at))))
    opt=optax.chain(optax.clip_by_global_norm(1),optax.adamw(rate,b1=.9,b2=.95,weight_decay=.01,mask=lambda p:jax.tree.map(lambda w:w.ndim>=2,p)))
    if (folder/"resume.npz").exists():state=restore(folder/"resume.npz",identity,opt)
    else:
        flag=ancestor/".sha-checked"
        prior=ancestor/"resume.npz"
        if not flag.exists() or flag.read_text()!=receipt["sha256"]["resume.npz"]:
            if sha_file(prior)!=receipt["sha256"]["resume.npz"]:raise ValueError("Corrupt parent transfer")
            flag.write_text(receipt["sha256"]["resume.npz"])
        state=restore(prior,disk_identity(card),opt)
    initial=int(state[-1]); count=sum(v.size for v in state[0].values())
    if not start_at<=initial<=a.until or count!=card["params"]:raise ValueError("Parent state or count mismatch")
    clock=folder/"times.json";dur=json.loads(clock.read_bytes()) if clock.exists() else {"step":initial,"update_seconds":0.,"compile_and_first_seconds":0.}
    if dur["step"]!=initial:
        raise ValueError("Timing snapshot and resumable optimizer are not synchronized")
    fn=compiled_steps(spec,opt,cal,a.micro,batch)
    def sample(s):return stack([data.requests(i,batch) for i in range(s,s+a.scan)])
    print(json.dumps({"starting":name,"device":device.device_kind,"params":count,"from":initial,"to":a.until}),flush=True)
    rates=[];begin=time.monotonic()
    if initial<a.until:
        with ThreadPoolExecutor(max_workers=1) as io:
            future=io.submit(sample,initial)
            for s in range(initial,a.until,a.scan):
                before=time.monotonic();inputs=jax.device_put(future.result())
                if s+a.scan<a.until:future=io.submit(sample,s+a.scan)
                state,m=fn(state,inputs);jax.block_until_ready((state,m));elapsed=time.monotonic()-before
                v=np.asarray(m);done=s+a.scan
                if not np.isfinite(v).all():raise FloatingPointError((name,done,v))
                if s==initial:dur["compile_and_first_seconds"]+=elapsed
                else:rates.append(batch*a.scan/elapsed)
                dur["update_seconds"]+=elapsed;dur["step"]=done
                progress={"run":name,"step":done,"start":initial,"target":a.until,"device":device.device_kind,
                          "micro":a.micro,"batch":batch,"warm_examples_per_second":float(np.median(rates)) if rates else None,
                          "continuation_update_seconds":dur["update_seconds"],"compile_and_first_seconds":dur["compile_and_first_seconds"],
                          "wall_this_invocation":time.monotonic()-begin,"loss":dict(zip(("total","cross","event","action"),v.tolist()))}
                if done%32==0 or s==initial:write(folder/"progress.json",progress)
                if done%1024==0 or done==a.until:
                    checkpoint(folder,state,identity,base.stats);write(clock,dur);print(json.dumps(progress),flush=True)
    stage.mkdir(exist_ok=True,parents=True)
    for k in ("config.json","resume.npz","progress.json"):link_exact(folder/k,stage/k)
    export(stage/"resume.npz",a.data/"prior.npz",stage/"inference.npz")
    hashes={k:sha_file(stage/k) for k in ("resume.npz","inference.npz","config.json","progress.json")}
    write(stage/"complete.json",{"run":name,"step":a.until,"device":device.device_kind,"identity":identity,"sha256":hashes})
    print("Frozen",name,a.until,hashes["inference.npz"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("data","parents","out","fresh"):p.add_argument("--"+k,type=Path,required=k!="fresh")
    p.add_argument("--generation",choices=("v5e","v6e"),required=True);p.add_argument("--scale",choices=("anchor","larger"),required=True)
    p.add_argument("--core",choices=("hybrid","gpt2"),required=True);p.add_argument("--arm",choices=("replay","fresh"),required=True)
    p.add_argument("--seed",type=int,required=True);p.add_argument("--until",type=int,required=True);p.add_argument("--micro",type=int,default=64)
    p.add_argument("--scan",type=int,default=4);a=p.parse_args()
    if a.arm=="fresh" and a.fresh is None:p.error("A verified fresh pack is required")
    run(a)
