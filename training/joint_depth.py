"""Secondary 6-vs-12 executed-block comparison on the same joint exposure stream."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from training.joint_data import read
from training.joint_evaluate import EVAL_FILES, TASKS, evaluate
from training.joint_results import effects
from training.joint_sources import CONFIG,digest
from training.joint_train import SOURCE,atomic,gpu,train

SPEC = Path("configs/t4/joint_depth.json")
PROTOCOL = Path("docs/research/joint_depth_20261109.md")
EXTRA = [str(SPEC),str(PROTOCOL),"training/joint_depth.py"]


def parent_dir(base):
    return (Path(base["data"])/"depth_parents").resolve()


def prepare(base,spec):
    dest = parent_dir(base)
    original = Path(os.environ["SLWM_ORIGINAL_S5_ROOT"])
    parents = {}
    for key,(name,sha) in base["parents"].items():
        path = dest/name/"best.pt"
        marker = path.with_suffix(".json")
        if not marker.exists():
            src = original/name/"best.pt"
            if digest(src)!=sha:
                raise ValueError("Wrong parent")
            full = torch.load(src,map_location="cpu",mmap=True,weights_only=False)
            layers = spec["layers_zero_based"]
            weights = {}
            for k,v in full["model"].items():
                if k.startswith("blocks."):
                    _,n,suffix = k.split(".",2)
                    if int(n) not in layers:
                        continue
                    k = f"blocks.{layers.index(int(n))}.{suffix}"
                weights[k] = v
            cfg = copy.deepcopy(full["config"])
            cfg["model"]["n_layer"] = len(layers)
            atomic(path,{"model":weights,"run":full["run"],"config":cfg,"total_token_equivalent_step":full["total_token_equivalent_step"],
                 "source_parent_sha256":sha,"retained_layers":layers,"inherited_language_training":True},True)
            atomic(marker,{"derived_parent_sha256":digest(path),"source_parent_sha256":sha,"retained_layers":layers})
        m = read(marker)
        if m["source_parent_sha256"]!=sha or m["retained_layers"]!=spec["layers_zero_based"] or digest(path)!=m["derived_parent_sha256"]:
            raise ValueError("Reduced parent provenance changed")
        parents[key] = [name,m["derived_parent_sha256"]]
    cfg = {**base,"output":str(Path(base["output"])/"depth6"),"parents":parents,
           "profiles":{k:base["profiles"][k] for k in spec["profiles"]}}
    root = Path(cfg["output"])
    atomic(root/"config.json",cfg)
    provenance = {p:digest(Path(p)) for p in [*SOURCE,*EXTRA]}
    provenance["main_training_lock_sha256"] = digest(Path(base["output"])/"training_lock.json")
    provenance["parents"] = parents
    lock = root/"training_lock.json"
    if lock.exists() and read(lock)!=provenance:
        raise RuntimeError("Reduced-depth study changed")
    atomic(lock,provenance)
    return cfg


def summary(base,cfg,step,role):
    r = Path(cfg["output"])
    main = Path(base["output"])
    rows = {}
    for seed in cfg["seeds"]:
        for name in cfg["profiles"]:
            key = f"{name}-{seed}"
            low,full = (read(p/key/f"step-{step}.json") for p in (r,main))
            if low["exposure_sha256"]!=full["exposure_sha256"]:
                raise ValueError("Depth comparison saw unequal data")
            small = r/key/f"eval-{step}"
            large = main/key/f"eval-{step}"
            value = read(small/f"{role}.json")
            rows[key] = {"six_layer_parameters":low["accounting"],"full_parameters":full["accounting"],"exposure_sha256":low["exposure_sha256"],
                "training_seconds_six":low["active_train_seconds"],"training_seconds_full":full["active_train_seconds"],
                "six_layer_scores":{t:value["tasks"][t]["scores"]["ordered-32"]["balanced_accuracy"] for t in TASKS},
                "six_minus_full":{t:effects(small/f"{t}-{role}-predictions.npz",large/f"{t}-{role}-predictions.npz","ordered-32","ordered-32",t) for t in TASKS}}
    atomic(r/f"depth-{step}-{role}.json",{"role":role,"updates":step,"rows":rows})


def follow():
    spec = read(SPEC)
    base = read(CONFIG)
    cfg = prepare(base,spec)
    root, main = Path(cfg["output"]),Path(base["output"])
    env = {**os.environ,"SLWM_S5_ROOT":str(parent_dir(base))}
    models = [(n,s) for s in cfg["seeds"] for n in cfg["profiles"]]
    def job(phase,name,seed,step,role="dev"):
        path = root/f"{name}-{seed}"
        marker = path/(f"step-{step}.json" if phase=="train" else f"eval-{step}/{role}.json")
        if marker.exists():
            return
        args = [sys.executable,"-m","training.joint_depth","--stage",phase,"--profile",name,"--seed",str(seed),"--step",str(step),"--role",role]
        atomic(root/"status.json",{"phase":phase,"name":name,"seed":seed,"step":step,"role":role})
        subprocess.run(args,env=env,check=True)
    for step in spec["milestones"]:
        if step==1000:
            atomic(root/"status.json",{"phase":"waiting_for_primary_confirmation"})
            while not (main/"summary-8000-test.json").exists():
                time.sleep(45)
        for n,s in models:
            job("train",n,s,step)
            if step<8000:
                job("evaluate",n,s,step)
        if step<8000:
            for n,s in models:
                job("evaluate",n,s,0)
            summary(base,cfg,step,"dev")
    lock = {"endpoints":{},"evaluation":{p:digest(Path(p)) for p in EVAL_FILES},
            "depth_sources":{p:digest(Path(p)) for p in EXTRA},"full_endpoint_lock_sha256":digest(main/"confirmation_lock.json")}
    for n,s in models:
        folder = root/f"{n}-{s}"
        sha = digest(folder/"step-8000.pt")
        if sha!=read(folder/"step-8000.json")["delta_sha256"]:
            raise ValueError("Damaged reduced-depth weights")
        lock["endpoints"][folder.name] = sha
    path = root/"confirmation_lock.json"
    if path.exists() and read(path)!=lock:
        raise ValueError("Depth confirmation lock changed")
    atomic(path,lock)
    for n,s in models:
        job("evaluate",n,s,0,"test")
        job("evaluate",n,s,8000,"test")
    summary(base,cfg,8000,"test")
    atomic(root/"status.json",{"phase":"complete"})


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage",choices=("follow","train","evaluate"),required=True)
    p.add_argument("--profile")
    p.add_argument("--seed",type=int)
    p.add_argument("--step",type=int)
    p.add_argument("--role",default="dev")
    a = p.parse_args()
    if a.stage=="follow":
        base = read(CONFIG)
        with (Path(base["data"])/"depth.lock").open("a") as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            follow()
    else:
        root = Path(read(CONFIG)["output"])/"depth6"
        cfg = read(root/"config.json")
        with gpu():
            if a.stage=="train":
                train(cfg,a.profile,a.seed,a.step,digest(root/"training_lock.json"))
            else:
                evaluate(cfg,a.profile,a.seed,a.step,a.role)
