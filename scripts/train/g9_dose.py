"""Revisit independent episode count at 4,096 updates using the matched fresh parents."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
from evals.g9_live import live
from models.g9_shared import Spec
from scripts.train.g9_fit import bank_data,one,tag,save_json,numpy_weights

def main(data,primary,out):
    jax.config.update("jax_compilation_cache_dir","/content/g9-xla")
    bank,starts,names=bank_data(data/"starss-train.npz");scores={}
    for seed in (911,912):
        for core in ("hybrid","gpt2"):
            replay=("exposure",core,512,"gelu",seed,True)
            one(replay,bank,starts,names,out)
            for arm,root in ((replay,out),(replay[:-1]+(False,),primary)):
                name=tag(arm);stage=root/name/"step-04096";d=json.loads((stage/"receipt.json").read_text())
                p=jax.tree.map(jnp.asarray,numpy_weights(stage/"weights.npz"));spec=Spec(**d["spec"])
                scores[name]={"sha256":d["inference_sha256"],"optimizer_sha256":d["optimizer_sha256"],"step":4096}
                for mode in ("carry","block32"):
                    values=live(p,spec,mode,seed=857329)
                    scores[name][mode]=dict(occupancy=values[:,:,0].mean(0).tolist(),success=values[:,:,0].reshape(8,64,-1).any(1).mean(0).tolist())
            save_json(out/"paired.json",scores)
    save_json(out/"finished.json",scores)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("data",type=Path);ap.add_argument("primary",type=Path);ap.add_argument("out",type=Path)
    a=ap.parse_args();main(a.data,a.primary,a.out)
