"""Release two prespecified small equal-exposure parents and measure actual control after quantization."""
import argparse
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from models.g9_portable import pack,load
from models.g9_shared import Spec
from evals.g9_live import live,paired_histories
from scripts.train.g9_fit import numpy_weights,save_json


def main(runs,dest):
    dest.mkdir(exist_ok=True,parents=True)
    results={}
    for core in ("hybrid","gpt2"):
        tag=f"exposure-{core}-z128-gelu-s911-fresh";stage=runs/tag/"step-04096"
        d=json.loads((stage/"receipt.json").read_text());source=stage/"weights.npz"
        with source.open("rb") as f:
            if hashlib.file_digest(f,"sha256").hexdigest()!=d["inference_sha256"]:raise ValueError("Mismatched source weight generation")
        weights=numpy_weights(source);file=dest/(tag+".npz")
        pack(file,weights,d["spec"],d["inference_sha256"]);meta,decoded=load(file)
        spec=Spec(**meta["spec"]);p=jax.tree.map(jnp.asarray,decoded);parent=jax.tree.map(jnp.asarray,weights)
        left=live(parent,spec,"carry");right=live(p,spec,"carry")
        h=paired_histories(p,spec)
        with file.open("rb") as f:sha=hashlib.file_digest(f,"sha256").hexdigest()
        results[tag]=dict(sha256=sha,parent_sha256=d["inference_sha256"],bytes=file.stat().st_size,original_bytes=source.stat().st_size,
               source_mission_occupancy=left[:,:,0].mean(0).tolist(),portable_mission_occupancy=right[:,:,0].mean(0).tolist(),
               portable_sensor_identical_history=float(h["matched"].mean()),backend=jax.default_backend())
        print(tag,sha,file.stat().st_size,float(left[:,:,0].mean()),float(right[:,:,0].mean()),flush=True)
    save_json(dest/"portable.json",results)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("runs",type=Path);ap.add_argument("dest",type=Path)
    a=ap.parse_args();main(a.runs,a.dest)
