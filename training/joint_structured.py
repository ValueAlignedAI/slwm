"""Fixed 2:4 pruning in the actual shared MLP, with equal-exposure joint adaptation."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from training.joint_data import read
from training.joint_evaluate import EVAL_FILES,TASKS,evaluate
from training.joint_results import effects
from training.joint_sources import CONFIG,digest
from training.joint_train import SOURCE,atomic,gpu,seed_all,train

SPEC = Path("configs/t4/joint_structured.json")
EXTRA = [str(SPEC),"training/joint_structured.py","docs/research/joint_structured_20261109.md"]


def prune(weight):
    """[out,in] -> same matrix with at most two nonzeros in each K-axis quartet."""
    groups = weight.detach().reshape(-1,4)
    take = groups.abs().argsort(dim=-1,descending=True,stable=True)[:,:2]
    result = torch.zeros_like(groups).scatter_(1,take,groups.gather(1,take))
    if (result.count_nonzero(dim=1)>2).any():
        raise ValueError("Invalid semi-structured pattern")
    return result.reshape_as(weight)


def prepare(base):
    root = Path(base["output"])/"structured"
    src = Path(os.environ["SLWM_ORIGINAL_S5_ROOT"])
    dest = (Path(base["data"])/"structured_parents").resolve()
    parents,audits = {},{}
    for seed in base["seeds"]:
        key = f"slwm-{seed}"
        name,source_sha = base["parents"][key]
        path = dest/name/"best.pt"
        marker = path.with_suffix(".json")
        if not marker.exists():
            original = src/name/"best.pt"
            if digest(original)!=source_sha:
                raise ValueError("Unexpected dense parent")
            s = torch.load(original,map_location="cpu",mmap=True,weights_only=False)
            weights = dict(s["model"])
            keys = [k for k in weights if k.endswith((".mlp.fc.weight",".mlp.proj.weight"))]
            if len(keys)!=24:
                raise ValueError("Expected both feed-forward matrices in every block")
            count = sum(weights[k].numel() for k in keys)
            for k in keys:
                weights[k] = prune(weights[k])
            survivors = sum(int(weights[k].count_nonzero()) for k in keys)
            atomic(path,{"model":weights,"run":s["run"],"config":s["config"],"total_token_equivalent_step":s["total_token_equivalent_step"],
                         "source_parent_sha256":source_sha,"sparsity":"2:4 MLP weights in all shared blocks"},True)
            atomic(marker,{"source_parent_sha256":source_sha,"checkpoint_sha256":digest(path),"matrices":keys,
                           "masked_matrix_entries":count,"surviving_nonzero_entries":survivors,"structural_slots_removed":count//2})
        m = read(marker)
        if digest(path)!=m["checkpoint_sha256"] or m["source_parent_sha256"]!=source_sha:
            raise ValueError("Structured checkpoint changed")
        parents[key] = [name,m["checkpoint_sha256"]]
        audits[key] = m
    cfg = {**base,"output":str(root),"parents":parents,"profiles":{"signal-dense":base["profiles"]["signal-dense"]}}
    atomic(root/"config.json",cfg)
    lock = {"sources":{p:digest(Path(p)) for p in [*EXTRA,*SOURCE]},
            "main_lock_sha256":digest(Path(base["output"])/"training_lock.json"),"parents":audits}
    p = root/"training_lock.json"
    if p.exists() and read(p)!=lock:
        raise ValueError("Structured protocol changed")
    atomic(p,lock)
    return cfg,dest


def benchmark(cfg,dest):
    from torch.sparse import to_sparse_semi_structured
    root = Path(cfg["output"])
    if (root/"kernel.json").exists():
        return
    seed_all(81)
    name,_ = cfg["parents"]["slwm-81"]
    w = torch.load(dest/name/"best.pt",map_location="cpu",weights_only=False,mmap=True)["model"]["blocks.0.mlp.fc.weight"].cuda().half()
    x = torch.randn(read(SPEC)["kernel_probe_rows"],w.shape[1],device="cuda",dtype=w.dtype)
    # Use the exact same masked weights and inputs for both kernel paths.
    with torch.inference_mode():
        packed = to_sparse_semi_structured(w)
        dense,other = F.linear(x,w).float(),F.linear(x,packed).float()
        relative = float((dense-other).norm()/dense.norm())
        if not np.isfinite(relative) or relative>.01:
            raise ValueError("Packed linear inference differs from the masked dense reference")
        timings = {"dense":[],"packed":[]}
        for rep in range(8):
            names = ("dense","packed") if rep%2==0 else ("packed","dense")
            for key in names:
                weight = w if key=="dense" else packed
                for _ in range(3):
                    F.linear(x,weight)
                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(32):
                    F.linear(x,weight)
                end.record()
                end.synchronize()
                timings[key].append(start.elapsed_time(end)/32)
    atomic(root/"kernel.json",{"gpu":torch.cuda.get_device_name(),"dtype":"fp16","shape":list(w.shape),"rows":len(x),
        "relative_output_l2":relative,"median_ms":{k:float(np.median(v)) for k,v in timings.items()},"per_round_ms":timings,
        "scope":"Single first feed-forward input matrix at one batch size. Joint adaptation uses dense kernels."})


def summary(base,cfg,step,role):
    root,main = Path(cfg["output"]),Path(base["output"])
    rows = {}
    for s in cfg["seeds"]:
        n = f"signal-dense-{s}"
        card = read(root/n/f"step-{step}.json")
        item = read(root/n/f"eval-{step}"/f"{role}.json")
        comparisons = {}
        for model in ("signal-dense","gpt2-dense"):
            other = main/f"{model}-{s}"
            if card["exposure_sha256"]!=read(other/f"step-{step}.json")["exposure_sha256"]:
                raise ValueError("Unequal exposure in structured comparison")
            comparisons[model] = {t:effects(root/n/f"eval-{step}"/f"{t}-{role}-predictions.npz",
                  other/f"eval-{step}"/f"{t}-{role}-predictions.npz","ordered-32","ordered-32",t) for t in TASKS}
        rows[n] = {"balanced_accuracy":{t:item["tasks"][t]["scores"]["ordered-32"]["balanced_accuracy"] for t in TASKS},
                   "comparisons":comparisons,"text":item["diagnostics"]["language"],"seconds":card["active_train_seconds"],
                   "delta_sha256":card["delta_sha256"],"exposure_sha256":card["exposure_sha256"]}
    atomic(root/f"structured-{step}-{role}.json",{"step":step,"role":role,"rows":rows})


def follow():
    base=read(CONFIG)
    cfg,dest=prepare(base)
    root,main=Path(cfg["output"]),Path(base["output"])
    env={**os.environ,"SLWM_S5_ROOT":str(dest)}
    def job(phase,seed,step,role="dev"):
        folder=root/f"signal-dense-{seed}"
        marker=folder/(f"step-{step}.json" if phase=="train" else f"eval-{step}/{role}.json")
        if marker.exists():
            return
        atomic(root/"status.json",{"phase":phase,"seed":seed,"step":step,"role":role})
        subprocess.run([sys.executable,"-m","training.joint_structured","--phase",phase,"--seed",str(seed),"--step",str(step),"--role",role],env=env,check=True)
    with gpu():
        try:
            benchmark(cfg,dest)
        except (RuntimeError,NotImplementedError) as exc:
            atomic(root/"kernel_unavailable.json",{"reason":str(exc),"scope":"Packed kernel unavailable on this software stack"})
    for step in cfg["milestones"]:
        if step>100:
            dependency=main/("summary-1000-dev.json" if step==1000 else "summary-8000-test.json")
            atomic(root/"status.json",{"phase":"waiting_for_full_depth_reference","budget":step})
            while not dependency.exists():
                time.sleep(45)
        for s in cfg["seeds"]:
            job("train",s,step)
            if step<8000:
                job("evaluate",s,step)
        if step<8000:
            for s in cfg["seeds"]:
                job("evaluate",s,0)
            summary(base,cfg,step,"dev")
    lock={"evaluation":{p:digest(Path(p)) for p in EVAL_FILES},"structured_sources":{p:digest(Path(p)) for p in EXTRA},
          "full_lock_sha256":digest(main/"confirmation_lock.json"),"endpoints":{}}
    for s in cfg["seeds"]:
        folder=root/f"signal-dense-{s}"
        sha=digest(folder/"step-8000.pt")
        if sha!=read(folder/"step-8000.json")["delta_sha256"]:
            raise ValueError("Damaged structured endpoint")
        lock["endpoints"][folder.name]=sha
    path=root/"confirmation_lock.json"
    if path.exists() and read(path)!=lock:
        raise ValueError("Structured confirmation changed")
    atomic(path,lock)
    for s in cfg["seeds"]:
        job("evaluate",s,0,"test")
        job("evaluate",s,8000,"test")
    summary(base,cfg,8000,"test")
    atomic(root/"status.json",{"phase":"complete"})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--phase",choices=("follow","train","evaluate"),required=True)
    p.add_argument("--seed",type=int)
    p.add_argument("--step",type=int)
    p.add_argument("--role",default="dev")
    a=p.parse_args()
    base=read(CONFIG)
    seed_all(81)
    if a.phase=="follow":
        with (Path(base["data"])/"structured.lock").open("a") as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            follow()
    else:
        root=Path(base["output"])/"structured"
        cfg=read(root/"config.json")
        with gpu():
            if a.phase=="train":
                train(cfg,"signal-dense",a.seed,a.step,digest(root/"training_lock.json"))
            else:
                evaluate(cfg,"signal-dense",a.seed,a.step,a.role)
