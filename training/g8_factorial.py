"""Train a missing cell of the G8 experiment using the exact G7 row stream, loss, optimizer and schedule."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax

from models.g4_shared import Spec
from training.g4_train import checkpoint,sha_file,write
from training.g5_batches import JointData,stack
from training.g6_checkpoint import export
from training.g6_train import compiled_steps,link_exact,tpu_device
from training.g7_batches import ContinuationData
from training.g7_continue import disk_identity,restore

CFG=Path("configs/t8/factorial.json")


class FactorialData:
    def __init__(self,base,fresh,arm,recipe):
        self.base=base
        self.bank=ContinuationData(base,fresh)
        self.arm=arm
        self.families=tuple({**recipe["new_arms"],**recipe["reused_arms"]}[arm])
        self.identity=disk_identity({"replay":base.identity,"fresh":self.bank.identity,"families":list(self.families)})

    def requests(self,step,batch):
        if self.arm=="replay":
            return self.base.requests(step,batch)
        new=self.bank.requests(step,batch)
        if self.arm=="fresh":
            return new
        # Preserve the exact shared RNG stream, including choices made for families absent from this intervention.
        old=self.base.requests(step,batch)
        keep=np.isin(old["family"],self.families)
        for k in ("a","v","x","input_x","n","u","outcome","target_n","event"):
            old[k][keep]=new[k][keep]
        return old


def run(a):
    exp=json.loads(CFG.read_bytes());start,end=exp["parent_steps"],exp["end_steps"]
    if a.until not in exp["rungs"] or a.micro!=exp["microbatch"] or a.scale!=exp["scale"]:
        raise ValueError("Outside the registered schedule")
    dev=tpu_device(a.generation);parent=f"g6-v5e-anchor-{a.core}-{a.seed}"
    origin=a.parents/parent/"step-008192";card=json.loads((origin/"config.json").read_bytes())
    receipt=json.loads((origin/"verified.json").read_bytes())
    if card["run"]!=parent or card["params"]!=exp["parameters"] or receipt["step"]!=start or len(set(receipt["disk_uuids"]))!=2:
        raise ValueError("Wrong physical parent")
    name=f"g8-{a.generation}-anchor-{a.core}-{a.seed}-{a.arm}"
    root=a.out/name;root.mkdir(parents=True,exist_ok=True);stage=root/f"step-{a.until:06d}"
    if (stage/"complete.json").exists():return
    if sha_file(a.data/"prior.npz")!=card["prior_sha256"]:raise ValueError("Physical prior changed")
    with np.load(a.data/"prior.npz") as f:cal={k:jax.device_put(f[k]) for k in ("coef","contact")}
    base=JointData(a.data,"train",verify=False)
    if base.identity!=card["data"]:raise ValueError("Replay dataset changed")
    data=FactorialData(base,a.fresh,a.arm,exp)
    sources=("training/g8_factorial.py","training/g7_batches.py","training/g7_continue.py","training/g6_train.py","models/g5_shared.py","models/g5_residual.py",str(CFG))
    cfg=dict(card,run=name,steps=end,parent=parent,parent_optimizer=receipt["sha256"]["resume.npz"],data=data.identity,base_data=base.identity,
             full_fresh_data=data.bank.identity,fresh_families=list(data.families),continuation_arm=a.arm,lr=exp["common_peak_lr"],
             tpu=a.generation,continuation_code={p:sha_file(Path(p)) for p in sources})
    identity=disk_identity(cfg);meta=root/"config.json"
    if meta.exists() and json.loads(meta.read_bytes())!=cfg:raise ValueError("Immutable run identity changed")
    if not meta.exists():write(meta,cfg)
    def rate(at):
        x=jnp.maximum(at-start,0)
        return exp["common_peak_lr"]*jnp.minimum(1.,(x+1)/exp["warmup_continuation_steps"])*(.1+.45*(1+jnp.cos(jnp.pi*x/(end-start))))
    opt=optax.chain(optax.clip_by_global_norm(1),optax.adamw(rate,b1=.9,b2=.95,weight_decay=.01,mask=lambda p:jax.tree.map(lambda w:w.ndim>=2,p)))
    if (root/"resume.npz").exists():
        state=restore(root/"resume.npz",identity,opt)
    else:
        if (origin/".sha-checked").read_text().strip()!=cfg["parent_optimizer"]:raise ValueError("Unverified original optimizer")
        state=restore(origin/"resume.npz",disk_identity(card),opt)
    initial=int(state[-1]);batch=exp["effective_batch"];scan=exp["scan"]
    if not start<=initial<=a.until or card["batch"]!=batch or sum(v.size for v in state[0].values())!=exp["parameters"]:
        raise ValueError("Unmatched parameters, exposure or step")
    timing_path=root/f"times-{initial:06d}.json"
    dur=json.loads(timing_path.read_bytes()) if timing_path.exists() else {"step":start,"update_seconds":0.,"compile_and_first_seconds":0.}
    if dur["step"]!=initial:raise ValueError("Resumed optimizer has no corresponding timing record")
    learn=compiled_steps(Spec(**card["spec"]),opt,cal,a.micro,batch)
    def sample(s):return stack([data.requests(i,batch) for i in range(s,s+scan)])
    rates=[];begin=time.monotonic()
    if initial<a.until:
        with ThreadPoolExecutor(max_workers=1) as host:
            future=host.submit(sample,initial)
            for step in range(initial,a.until,scan):
                now=time.monotonic();b=jax.device_put(future.result())
                if step+scan<a.until:future=host.submit(sample,step+scan)
                state,values=learn(state,b);jax.block_until_ready((state,values));elapsed=time.monotonic()-now
                completed=step+scan;vals=np.asarray(values)
                if not np.isfinite(vals).all():raise FloatingPointError((name,completed))
                if step==initial:dur["compile_and_first_seconds"]+=elapsed
                else:rates.append(scan*batch/elapsed)
                dur["update_seconds"]+=elapsed;dur["step"]=completed
                progress={"run":name,"step":completed,"target":a.until,"batch":batch,"micro":a.micro,"device":dev.device_kind,
                          "warm_examples_per_second":float(np.median(rates)) if rates else None,"continuation_update_seconds":dur["update_seconds"],
                          "compile_and_first_seconds":dur["compile_and_first_seconds"],"wall_this_invocation":time.monotonic()-begin,
                          "loss":dict(zip(("total","cross","event","action"),vals.tolist()))}
                if completed%32==0:write(root/"progress.json",progress)
                if completed%1024==0 or completed==a.until:
                    # Commit the immutable timing for this step before the optimizer's atomic pointer moves.
                    write(root/f"times-{completed:06d}.json",dur);checkpoint(root,state,identity,base.stats)
                    print(name,completed,progress["warm_examples_per_second"],flush=True)
    stage.mkdir(parents=True,exist_ok=True)
    for leaf in ("config.json","resume.npz","progress.json"):link_exact(root/leaf,stage/leaf)
    export(stage/"resume.npz",a.data/"prior.npz",stage/"inference.npz")
    hashes={k:sha_file(stage/k) for k in ("resume.npz","inference.npz","config.json","progress.json")}
    write(stage/"complete.json",{"run":name,"step":a.until,"device":dev.device_kind,"identity":identity,"sha256":hashes})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("data","out","parents","fresh"):p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--generation",choices=("v5e",),required=True);p.add_argument("--scale",choices=("anchor",),required=True)
    p.add_argument("--core",choices=("hybrid","gpt2"),required=True);p.add_argument("--seed",type=int,choices=(511,512),required=True)
    p.add_argument("--arm",choices=("sim","other","fresh","replay"),required=True);p.add_argument("--until",type=int,required=True)
    p.add_argument("--micro",type=int,default=64);run(p.parse_args())
