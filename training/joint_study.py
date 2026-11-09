"""Run the registered matrix idempotently and lock all endpoints before final scoring."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from training.joint_data import read
from training.joint_evaluate import EVAL_FILES
from training.joint_g2_anchor import HASHES
from training.joint_results import summarize
from training.joint_sources import CONFIG,digest
from training.joint_train import atomic


def run(cfg):
    root = Path(cfg["output"])
    matrix = [(n,s) for s in cfg["seeds"] for n in cfg["profiles"]]
    def job(module, args, marker):
        if marker.exists():
            return
        status = {"phase":module,"args":args,"started_unix":time.time(),"marker":str(marker)}
        atomic(root/"status.json",status)
        log = root/"study.log"
        with log.open("a") as f:
            code = subprocess.call([sys.executable,"-m","training."+module,*map(str,args)],stdout=f,stderr=subprocess.STDOUT)
        if code or not marker.exists():
            atomic(root/"status.json",{**status,"phase":"failed","exit_code":code})
            raise RuntimeError(f"{module} failed; see {log}")
    def evals(n,s,step,role):
        job("joint_evaluate",["--profile",n,"--seed",s,"--step",step,"--role",role],root/f"{n}-{s}"/f"eval-{step}"/f"{role}.json")
    def controls(role):
        evals("raw",0,0,role)
        # Zero-initialized adapters produce identical parent states across ranks and experts.
        # Each model's own step-zero control is retained for unambiguous artifact pairing.
        for n,s in matrix:
            evals(n,s,0,role)
        for name in HASHES:
            evals(name,0,0,role)
    for step in cfg["milestones"]:
        for n,s in matrix:
            job("joint_train",["--profile",n,"--seed",s,"--target",step],root/f"{n}-{s}"/f"step-{step}.json")
            if step<cfg["milestones"][-1]:
                evals(n,s,step,"dev")
        if step<cfg["milestones"][-1]:
            controls("dev")
            summarize(cfg,step,"dev")
    lock = {"endpoints":{}, "evaluation":{p:digest(Path(p)) for p in EVAL_FILES},
            "training_lock_sha256":digest(root/"training_lock.json"),"g2_anchors":HASHES,"parents":cfg["parents"]}
    for n,s in matrix:
        folder = root/f"{n}-{s}"
        sha = digest(folder/f"step-{step}.pt")
        if sha!=read(folder/f"step-{step}.json")["delta_sha256"]:
            raise ValueError("Cannot lock damaged endpoint")
        lock["endpoints"][folder.name] = sha
    path = root/"confirmation_lock.json"
    if path.exists() and read(path)!=lock:
        raise ValueError("Confirmation lock changed")
    atomic(path,lock)
    controls("test")
    for n,s in matrix:
        evals(n,s,step,"test")
    summarize(cfg,step,"test")
    atomic(root/"status.json",{"phase":"complete","final":f"summary-{step}-test.json","finished_unix":time.time()})


if __name__=="__main__":
    cfg = read(CONFIG)
    root = Path(cfg["output"])
    root.mkdir(parents=True,exist_ok=True)
    with (root/"supervisor.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(cfg)
