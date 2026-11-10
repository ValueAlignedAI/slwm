"""Isolate available history using frozen matched models and a new 51.2-second cohort."""
import argparse
import json
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from evals.g9_live import initial_memory,live
from models.g5_shared import attention
from models.g9_shared import Spec,extended_attention
from scripts.train.g9_fit import numpy_weights,save_json

SEED=655073


def compatibility():
    sample=jax.random.normal(jax.random.PRNGKey(41),(3,32,128)).astype(jnp.bfloat16)
    valid=jnp.ones(sample.shape[:2],bool)
    for previous in (None,(jnp.zeros((3,64,2,64),jnp.bfloat16),jnp.ones((3,64,2,64),jnp.bfloat16),jnp.zeros((3,64),bool))):
        a=attention(sample,sample,sample,valid,previous)
        b=extended_attention(sample,sample,sample,valid,previous,64)
        if not all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b))):
            raise ValueError("The generalized 64-position kernel differs from the training kernel")


def arm(stage,dest):
    doc=json.loads((stage/"receipt.json").read_text());spec=Spec(**doc["spec"]);tag=doc["tag"];file=dest/(tag+".json")
    if file.is_file():return
    p=jax.tree.map(jnp.asarray,numpy_weights(stage/"weights.npz"))
    result=dict(tag=tag,sha256=doc["inference_sha256"],seed=SEED,arms={})
    for mode in ("carry","full","block32"):
        t=time.perf_counter();arr=live(p,spec,mode,seed=SEED);first=time.perf_counter()-t
        t=time.perf_counter();repeat=live(p,spec,mode,seed=SEED);warm=time.perf_counter()-t
        if not np.array_equal(arr,repeat):raise ValueError("Inconsistent repeated execution")
        cache=initial_memory(spec,1,512 if mode=="full" else 64)
        scored=arr[:,:,0].reshape(8,64,-1)
        result["arms"][mode]=dict(occupancy=arr[:,:,0].mean(0).tolist(),success=scored.any(1).mean(0).tolist(),
               by_goal=scored.mean((1,2)).tolist(),first_wall_s=first,warm_wall_s=warm,
               cache_bytes_per_world=sum(x.size*x.dtype.itemsize for x in jax.tree.leaves(cache)))
    save_json(file,result)
    print(tag,[(k,round(np.mean(v["occupancy"]),4)) for k,v in result["arms"].items()],flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("runs",type=Path);ap.add_argument("dest",type=Path)
    a=ap.parse_args();a.dest.mkdir(exist_ok=True,parents=True);compatibility()
    for seed in (912,911):
        for field in (128,512):
            for core in ("hybrid","gpt2"):
                arm(a.runs/f"confirm-{core}-z{field}-gelu-s{seed}-fresh"/"step-04096",a.dest)
