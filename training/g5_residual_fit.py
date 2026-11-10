"""G5C matched residual training, with restorable cross-session generations."""
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
from training.g4_train import backend, checkpoint, restore, sha_file, write
from training.g5_batches import CONFIG, JointData, stack
from training.g5_fit import parser
from training.g4_train import archive


def fit(a,data=None):
    dev=backend(a.backend)
    if a.backend=="tpu" and "v6" not in dev.device_kind.lower() and "trillium" not in dev.device_kind.lower():
        raise RuntimeError(f"Expected v6e assignment, received {dev.device_kind}")
    cfg=json.loads(CONFIG.read_bytes())
    spec=Spec(a.core,*cfg["scales"][a.scale],dtype=a.precision)
    steps=cfg["stage_updates"][a.scale] if a.steps is None else a.steps
    batch=cfg["batch"][a.scale]
    micro=a.micro or batch
    if batch%micro or steps%a.scan or a.save_every%a.scan:
        raise ValueError("Batch and save cadence must divide the compiled update block")
    data=data or JointData(a.data,"train")
    priors=a.data/"prior.npz"
    with np.load(priors,allow_pickle=False) as f:
        cal={k:jax.device_put(f[k]) for k in ("coef","contact")}
    run=f"g5-{a.scale}-{a.core}-{a.seed}"+(f"-{a.label}" if a.label else "")
    out=a.output/run
    out.mkdir(parents=True,exist_ok=True)
    paths=["models/g4_shared.py","models/g5_shared.py","models/g5_residual.py","training/g5_priors.py",
           "training/g5_residual_fit.py","training/g5_batches.py",str(CONFIG)]
    specdata=asdict(spec)
    init=initialize(spec,a.seed)
    lr=.0006*(256/spec.width)**.5
    card={"run":run,"spec":specdata,"seed":a.seed,"steps":steps,"batch":batch,"unique_available":list(map(len,data.pools)),
          "data":data.identity,"params":sum(x.size for x in init.values()),"lr":lr,"schedule":"16-step warmup then cosine to 10%",
          "code":{s:sha_file(Path(s)) for s in paths},"data_seed":cfg["seed"]}
    card.update(variant="physical-residual",prior_sha256=sha_file(priors))
    identity=hashlib.sha256(json.dumps(card,sort_keys=True).encode()).hexdigest()
    if (out/"config.json").exists() and json.loads((out/"config.json").read_bytes())!=card:
        raise ValueError("A different training recipe owns this run ID")
    write(out/"config.json",card)
    lr_fn=lambda t:lr*jnp.minimum(1.,(t+1)/16)*(.1+.45*(1+jnp.cos(jnp.pi*t/steps)))
    opt=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(lr_fn,b1=.9,b2=.95,weight_decay=.01,
                                      mask=lambda p:jax.tree.map(lambda a:a.ndim>=2,p)))
    state=(init,opt.init(init),jnp.int32(0))
    del init
    if (out/"resume.npz").exists():
        state=restore(out/"resume.npz",state,identity)
    start=int(state[-1])
    if start==steps and (out/"complete.json").exists():
        return out
    def update(state,b):
        p,ost,at=state
        loss=jax.value_and_grad(lambda w,examples:objective(w,spec,examples,cal),has_aux=True)
        if batch==micro:
            (_,metrics),grads=loss(p,b)
        else:
            blocks=jax.tree.map(lambda x:x.reshape(batch//micro,micro,*x.shape[1:]),b)
            def add(carry,small):
                total,m=carry
                (_,local),grad=loss(p,small)
                return (jax.tree.map(jnp.add,total,grad),m+local),None
            (grads,metrics),_=jax.lax.scan(add,(jax.tree.map(jnp.zeros_like,p),jnp.zeros(4)),blocks)
            grads=jax.tree.map(lambda x:x/(batch//micro),grads)
            metrics=metrics/(batch//micro)
        changes,ost=opt.update(grads,ost,p)
        return (optax.apply_updates(p,changes),ost,at+1),metrics
    def block(state,inputs):
        state,metrics=jax.lax.scan(update,state,inputs)
        return state,metrics.mean(0)
    compiled=jax.jit(block,donate_argnums=(0,))
    def sample(s):
        return stack([data.requests(i,batch) for i in range(s,s+a.scan)])
    rates=[]
    t0=time.monotonic()
    if start<steps:
        with ThreadPoolExecutor(max_workers=1) as io:
            future=io.submit(sample,start)
            for s in range(start,steps,a.scan):
                before=time.monotonic()
                inputs=jax.device_put(future.result())
                if s+a.scan<steps:
                    future=io.submit(sample,s+a.scan)
                state,metrics=compiled(state,inputs)
                metrics=np.asarray(jax.block_until_ready(metrics))
                elapsed=time.monotonic()-before
                if not np.isfinite(metrics).all():
                    raise FloatingPointError((run,s,metrics))
                if s!=start:
                    rates.append(batch*a.scan/elapsed)
                done=s+a.scan
                progress={"run":run,"step":done,"target":steps,"device":dev.device_kind,"batch":batch,"micro":micro,
                          "examples_per_second":batch*a.scan/elapsed,"median_warm_examples_per_second":float(np.median(rates)) if rates else None,
                          "loss":dict(zip(("total","cross","impact","action"),metrics.tolist())),"stream_positions":done*batch*64,
                          "wall_s_since_resume":time.monotonic()-t0}
                write(out/"progress.json",progress)
                if done%a.save_every==0 or done==steps:
                    resume,_=checkpoint(out,state,identity,data.stats)
                    if a.spool:
                        archive(run,done,{"resume.npz":resume,"config.json":out/"config.json"},a.spool)
                    print(json.dumps(progress),flush=True)
    with (out/"inference.tmp").open("wb") as fd:
        weights=jax.device_get(state[0])
        np.savez(fd,**weights,stats=data.stats,**{"prior_"+k:np.asarray(v) for k,v in cal.items()})
    (out/"inference.tmp").replace(out/"inference.npz")
    write(out/"complete.json",{"run":run,"step":steps,"device":dev.device_kind,
            "identity":identity,"sha256":{k:sha_file(out/k) for k in ("resume.npz","inference.npz","config.json","progress.json")}})
    return out


if __name__=="__main__":
    options=parser()
    options.add_argument("--spool",type=Path)
    args=options.parse_args()
    if not args.label or "residual" not in args.label:
        options.error("Use a separate residual study ID")
    fit(args)
