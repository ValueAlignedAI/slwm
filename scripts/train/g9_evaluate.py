"""Read frozen weights and write per-mission, per-room and per-scene paired scores."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from evals.g9_live import live,paired_histories,raw_receipt
from evals.g9_transfer import records,room,robot
from models.g9_shared import Spec,encode
from models.g9_tabletop import ACTIONS,sim_batch
from scripts.train.g9_fit import numpy_weights,save_json


def simulation(params,spec,seed):
    @jax.jit
    def score(p,keep):
        r=sim_batch(jnp.arange(384),seed)
        z,_=encode(p,spec,r,jnp.broadcast_to(keep,(384,4)))
        logits=z@p["read"][:,-ACTIONS:].astype(z.dtype)
        return (logits.argmax(-1)==r["policy"]).mean(1)
    masks=[[1,1,1,1],[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]]
    return {name:np.asarray(score(params,jnp.array(mask,bool))).tolist() for name,mask in zip(("all","no_audio","no_vision","no_physics","no_text"),masks)}


def evaluate(stage,data,dest,role,parts):
    meta=json.loads((stage/"receipt.json").read_text());key=meta["tag"]+"-"+str(meta["step"])
    spec=Spec(**meta["spec"]);seed=180013 if role=="test" else 160007
    dest.mkdir(exist_ok=True,parents=True)
    result=dest/(key+"-"+role+".json")
    doc=json.loads(result.read_text()) if result.exists() else dict(tag=meta["tag"],step=meta["step"],role=role,sha256=meta["inference_sha256"])
    if all(k in doc for k in parts):return
    with (stage/"weights.npz").open("rb") as f:
        if hashlib.file_digest(f,"sha256").hexdigest()!=meta["inference_sha256"]:raise ValueError("Inference digest mismatch")
    p=jax.tree.map(jnp.asarray,numpy_weights(stage/"weights.npz"))
    stamp=time.monotonic()
    for part in parts:
        if part in doc:continue
        if part=="live":
            doc[part]={}
            for mode in ("carry","reset","wrong"):
                arr=live(p,spec,mode,seed=seed)
                doc[part][mode]={k:arr[:,:,j].mean(0).tolist() for j,k in enumerate(("occupancy","hold","tap","missed_hold","oracle_check"))}
                doc[part][mode]["success"]=arr[:,:,0].reshape(8,64,-1).any(1).mean(0).tolist()
                doc[part][mode]["first_mission_actions"]=arr[:,0,5].astype(int).tolist()
        elif part=="history":
            score=paired_histories(p,spec,seed=seed+31)
            if not bool(score["sensor_identical"]) or float(score["opposite"])!=1:raise ValueError("Invalid action-history contrast")
            doc[part]={k:v.tolist() for k,v in score.items()}
        elif part=="sim":
            doc[part]=simulation(p,spec,seed+40)
        elif part=="room":
            doc[part]=room(p,spec,records(data/f"starss-{role}.npz"),records(data/"starss-train.npz"))
        elif part=="contact":
            if role!="test":raise ValueError("External robot panel has only a scene-held evaluation role")
            doc[part]=robot(p,spec,records(data/"contact/roma-cal.npz"),records(data/"contact/roma-held.npz"))
        else:raise ValueError(part)
        doc.setdefault("seconds",{})[part]=round(time.monotonic()-stamp,3);stamp=time.monotonic()
        save_json(result,doc)
        print(key,role,part,doc["seconds"][part],flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("stage",type=Path);ap.add_argument("data",type=Path);ap.add_argument("dest",type=Path)
    ap.add_argument("--role",choices=["dev","test"],required=True);ap.add_argument("--parts",nargs="+",required=True)
    a=ap.parse_args();evaluate(a.stage,a.data,a.dest,a.role,a.parts)
