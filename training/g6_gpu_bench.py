"""Measure the identical G6 accumulated optimizer step on the RTX 3090 at the TPU's batch and data."""
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
from models.g5_residual import initialize
from training.g5_batches import JointData,stack
from training.g6_train import CFG,compiled_steps
from training.g4_train import write,sha_file


def run(a):
    dev=jax.devices()[0]
    if dev.platform!="gpu" or "3090" not in dev.device_kind: raise ValueError("Require the registered 3090")
    cfg=json.loads(CFG.read_bytes()); spec=Spec(a.core,*cfg["scales"][a.scale],dtype="bfloat16")
    data=JointData(a.data,"train",verify=False); batch=cfg["batch"]; horizon=cfg["schedule_steps"]
    if sha_file(a.data/"prior.npz")!=cfg["prior_sha256"]: raise ValueError("Calibration changed")
    with np.load(a.data/"prior.npz") as f: cal={k:jax.device_put(f[k]) for k in ("coef","contact")}
    lr=.0006*(256/cfg["learning_rate_reference_width"])**.5
    rate=lambda t:lr*jnp.minimum(1.,(t+1)/16)*(.1+.45*(1+jnp.cos(jnp.pi*t/horizon)))
    opt=optax.chain(optax.clip_by_global_norm(1),optax.adamw(rate,b1=.9,b2=.95,weight_decay=.01,
                    mask=lambda p:jax.tree.map(lambda x:x.ndim>=2,p)))
    p=initialize(spec,511)
    if sum(v.size for v in p.values())!=cfg["expected_parameters"][a.scale]: raise ValueError("Size mismatch")
    state=(p,opt.init(p),jnp.int32(0)); del p
    fn=compiled_steps(spec,opt,cal,a.micro,batch)
    sample=lambda i:stack([data.requests(j,batch) for j in range(i,i+4)])
    times=[]
    with ThreadPoolExecutor(max_workers=1) as io:
        future=io.submit(sample,0)
        for i in (0,4,8):
            start=time.monotonic()
            inputs=jax.device_put(future.result())
            if i<8: future=io.submit(sample,i+4)
            state,m=fn(state,inputs); jax.block_until_ready((state,m))
            elapsed=time.monotonic()-start
            if not np.isfinite(np.asarray(m)).all(): raise ValueError("Nonfinite benchmark")
            times.append(elapsed)
    write(a.output,{"device":dev.device_kind,"jax":jax.__version__,"core":a.core,"scale":a.scale,"params":cfg["expected_parameters"][a.scale],
        "batch":batch,"micro":a.micro,"seed":511,"training_identity":data.identity,"first_block_seconds":times[0],
        "two_warm_block_seconds":times[1:],"median_warm_examples_per_second":float(np.median([batch*4/t for t in times[1:]]))})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--data",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--scale",choices=["anchor","larger"],required=True); p.add_argument("--core",choices=["hybrid","gpt2"],required=True)
    p.add_argument("--micro",type=int,required=True); run(p.parse_args())
