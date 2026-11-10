"""Generate acoustic features for the unchanged, speaker-held speech diagnostic from a terminal G6 pair."""
import argparse
import json
from pathlib import Path

import jax
import numpy as np

from models.g4_shared import Spec
from models.g5_residual import world
from training.g4_train import sha_file
from training.g5_batches import JointData


def predict(a):
    wave={}; models={}
    for core in ("hybrid","gpt2"):
        run=a.root/f"g6-{a.generation}-{a.scale}-{core}-{a.seed}"/"step-008192"
        cfg=json.loads((run/"config.json").read_bytes()); rec=json.loads((run/"verified.json").read_bytes())
        model=rec["sha256"]["inference.npz"]
        if rec["step"]!=8192 or sha_file(run/"inference.npz")!=model:
            raise ValueError("Refusing a nonterminal or changed model")
        with np.load(run/"inference.npz") as f:
            params={k:jax.device_put(f[k]) for k in f if k!="stats" and not k.startswith("prior_")}
            cal={k[6:]:jax.device_put(f[k]) for k in f if k.startswith("prior_")}
            d=JointData(a.data,"dev",stats=f["stats"],verify=False)
        spec=Spec(**cfg["spec"])
        base=d.packs[0]; grid=d.pools[0]
        grid=grid[base["ok"][grid].sum(1)==64]
        picked=np.concatenate([rows[np.argsort(base["uid"][rows])[:8]] for g in np.unique(base["group"][grid])
                               for rows in [grid[base["group"][grid]==g]]])
        ids=np.array(base["uid"][picked])
        if "uid" in wave: np.testing.assert_array_equal(ids,wave["uid"])
        wave["uid"]=ids; wave["codec"]=base["a"][picked].astype(np.float32)/16-12
        @jax.jit
        def infer(p,b):
            shared,_,_=world(p,spec,b,b["source"],cal)
            return shared[:,:,:100].astype(jax.numpy.float32)*3-5
        for view,ix in (("video",1),("text",3)):
            b=d.gather(np.zeros(len(picked),int),picked,np.zeros(len(picked),int))
            b["source"][:,ix]=1
            wave[core+"-"+view]=np.clip(np.asarray(infer(params,b)),-12,4)
        models[core]=model
        del params,cal,d
        jax.clear_caches()
    a.out.mkdir(exist_ok=True,parents=True)
    with (a.out/"speech-mels.partial").open("wb") as f:
        np.savez_compressed(f,**wave)
    (a.out/"speech-mels.partial").replace(a.out/"speech-mels.npz")
    (a.out/"weights.json").write_text(json.dumps(models,sort_keys=True)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--root",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    p.add_argument("--generation",required=True,choices=["v5e","v6e"]); p.add_argument("--scale",required=True,choices=["larger","anchor"])
    p.add_argument("--seed",required=True,type=int); predict(p.parse_args())
