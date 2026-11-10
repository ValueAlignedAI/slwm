"""Independent fixed-head replication of the four-decision hidden-goal ranking test."""
import argparse
import json
from pathlib import Path
import jax
import numpy as np
from training.g7_policy import base_weights,data,evaluate
from training.g4_train import sha_file,write
from scripts.train.g7_lock_heads import lock


def main(a):
    root=a.root;out=root/"supervised-confirmation";out.mkdir(exist_ok=True)
    tags=[(family,f"{core}-{mode}-{s}") for family in ("policy","policy-prior") for s in (711,712)
          for core,mode in (("hybrid","raw"),("hybrid","frozen"),("gpt2","frozen"),("hybrid","adapter"),("gpt2","adapter"))]
    registered={}
    for family,tag in tags:
        dev=json.loads((root/family/tag/"dev.json").read_bytes())
        h=sha_file(root/family/tag/"weights.npz")
        if h!=dev["weights_sha256"] or dev["steps"]!=1024:raise ValueError("Changed SFT head")
        registered[family+"/"+tag]=h
    lock(out/"lock.json",registered)
    models={}
    for family,tag in tags:
        if tag!=a.tag:continue
        file=out/(family+"-"+tag+".json")
        if file.is_file():continue
        core,mode,s=tag.split("-")
        spec,p,_,_,parent=base_weights(root.parent/"g6",core,int(s))
        with np.load(root/family/tag/"weights.npz") as f:h={k:jax.device_put(v) for k,v in f.items() if k!="parent"}
        prior=None
        if family=="policy-prior":
            with np.load(root/"reacher-prior/prior.npz") as f:prior={k:jax.device_put(v) for k,v in f.items()}
        r=evaluate(p,h,spec,data(a.data/"reacher/test"),mode=="raw",prior)
        write(file,{"family":family,"tag":tag,"fresh_starts":512,"past_cue_goal_pairs":1024,"core_sha256":parent,
            "weights_sha256":registered[family+"/"+tag],"results":r})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--data",type=Path,required=True)
    p.add_argument("--tag",required=True);main(p.parse_args())
