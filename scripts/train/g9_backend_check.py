"""Repeat terminal control and history on native TPU while CUDA independently reads both-disk generations."""
import argparse
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from evals.g9_live import live,paired_histories
from models.g9_shared import Spec
from scripts.train.g9_fit import numpy_weights,save_json


def main(root,out):
    if jax.default_backend()!="tpu":raise ValueError("This inference crosscheck is defined on native TPU")
    out.mkdir(parents=True,exist_ok=True);collected={}
    for size in ("exposure","confirm"):
        for seed in (911,912):
            for field in (128,512):
                for core in ("hybrid","gpt2"):
                    tag=f"{size}-{core}-z{field}-gelu-s{seed}-fresh";stage=root/tag/"step-04096";dest=out/(tag+".json")
                    meta=json.loads((stage/"receipt.json").read_text());spec=Spec(**meta["spec"])
                    if not dest.exists():
                        with (stage/"weights.npz").open("rb") as f:
                            if hashlib.file_digest(f,"sha256").hexdigest()!=meta["inference_sha256"]:raise ValueError("Native checkpoint mismatch")
                        p=jax.tree.map(jnp.asarray,numpy_weights(stage/"weights.npz"));r=live(p,spec,"carry",seed=180013)
                        history=paired_histories(p,spec,seed=180044)
                        save_json(dest,dict(tag=tag,sha256=meta["inference_sha256"],occupancy=r[:,:,0].mean(0).tolist(),
                              success=r[:,:,0].reshape(8,64,-1).any(1).mean(0).tolist(),paired_history=history["matched"].tolist()))
                    collected[tag]=json.loads(dest.read_text())
                    print(tag,round(np.mean(collected[tag]["occupancy"]),5),flush=True)
    save_json(out/"all.json",collected)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path);p.add_argument("out",type=Path);a=p.parse_args();main(a.root,a.out)
