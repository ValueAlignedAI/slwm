"""Prospectively split transfer of G3 representations to paired Push-T forecasting."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from numcodecs import blosc

from training.joint_data import read
from training.joint_evaluate import field_views
from training.joint_sources import CONFIG,digest
from training.joint_train import atomic,gpu,load_parent,seed_all,tensors

SPEC = Path("configs/t4/joint_pusht.json")
VERSION = [SPEC,Path("docs/research/joint_pusht_20261109.md"),Path(__file__),Path("training/joint_evaluate.py"),Path("models/joint_transfer.py")]


class Replay:
    """Read only the requested within-episode rows from official Zarr-v2 chunks."""
    def __init__(self,path):
        self.archive = zipfile.ZipFile(path)
        root = "pusht/pusht_cchi_v7_replay.zarr/"
        self.meta = {}
        for key in ("meta/episode_ends","data/img","data/state","data/action"):
            folder = root+key+"/"
            meta = json.loads(self.archive.read(folder+".zarray"))
            if meta["order"]!="C" or meta["compressor"]["id"]!="blosc" or meta["chunks"][1:]!=meta["shape"][1:]:
                raise ValueError("Unexpected Push-T array schema")
            self.meta[key] = (folder,meta)
        self.ends = self.rows("meta/episode_ends",np.arange(206))
        if self.ends[-1]!=25650 or (np.diff(self.ends)<=0).any():
            raise ValueError("Invalid episode boundaries")

    @lru_cache(maxsize=6)
    def chunk(self,key,idx):
        folder,m = self.meta[key]
        part = ".".join([str(idx)]+["0"]*(len(m["shape"])-1))
        data = np.frombuffer(blosc.decompress(self.archive.read(folder+part)),m["dtype"]).reshape(m["chunks"])
        return data

    def rows(self,key,idx):
        _,m = self.meta[key]
        blocks = idx//m["chunks"][0]
        out = np.empty((len(idx),*m["shape"][1:]),dtype=m["dtype"])
        for b in np.unique(blocks):
            match = blocks==b
            out[match] = self.chunk(key,int(b))[idx[match]%m["chunks"][0]]
        return out

    def arrays(self,episodes,cfg):
        h,q = cfg["history"],cfg["horizon"]
        images,signals,plans,targets,groups = [],[],[],[],[]
        for e in episodes:
            lo = int(self.ends[e-1]) if e else 0
            hi = int(self.ends[e])
            anchors = np.unique(np.linspace(lo+h,hi-q-1,cfg["windows"],dtype=int))
            if hi-lo<h+q+1:
                continue
            for end in anchors:
                ix = np.arange(end-h+1,end+1)
                past = self.rows("data/state",ix)[:,:2]
                prev = self.rows("data/action",ix-1)
                rgb = self.rows("data/img",ix).reshape(h,8,12,8,12,3).mean((2,4))
                # Official arrays store RGB float values in the 0..255 image scale.
                if not np.isfinite(rgb).all() or rgb.max()>255.01 or rgb.min()<-.01 or rgb.max()<2:
                    raise ValueError("Unexpected image range")
                images.append((rgb.transpose(0,3,1,2).reshape(h,192)/127.5-1).astype(np.float32))
                signals.append(np.column_stack([past/256-1,(prev-past)/256]).astype(np.float32))
                plan = self.rows("data/action",np.arange(end,end+q))
                plans.append(((plan-past[-1])/256).flatten())
                pos = self.rows("data/state",np.array([end,end+q]))[:,2:4]
                targets.append(pos[1]-pos[0])
                groups.append(e)
        return {k:np.asarray(v) for k,v in zip(("image","signal","plan","target","episode"),(images,signals,plans,targets,groups))}


def splits(cfg):
    order = sorted(range(206),key=lambda i:hashlib.sha256(f"{cfg['split_salt']}:{i}".encode()).digest())
    offset,parts = 0,{}
    for role,n in cfg["episodes"].items():
        parts[role] = order[offset:offset+n]
        offset += n
    if offset!=206:
        raise ValueError("Incomplete partition")
    return parts


def representations(model,data,mode):
    sig = data["signal"]
    if model is None:
        pieces = [data["image"].reshape(len(sig),-1)] if mode!="numeric" else []
        if mode!="image":
            pieces.append(sig.reshape(len(sig),-1))
        return np.concatenate(pieces,1)
    result = []
    for i in range(0,len(sig),96):
        sub = sig[i:i+96]
        batch = {}
        if mode!="image":
            n = np.zeros((*sub.shape[:2],64),np.float32)
            present = np.zeros(n.shape,bool)
            channels = [33,34,43,44]
            n[:,:,channels] = sub
            present[:,:,channels] = True
            batch.update(numeric=n,present=present)
        if mode!="numeric":
            batch["image"] = data["image"][i:i+len(sub)]
        with torch.autocast("cuda",dtype=torch.bfloat16):
            z,_,_ = model.field(tensors(batch,"cuda"))
        lens = torch.full((len(z),),z.shape[1],device=z.device)
        result.append(field_views(z.float(),lens)["ordered"].cpu().numpy())
    return np.concatenate(result)


def kernel(fit,held):
    fit,held = fit.astype(np.float64),held.astype(np.float64)
    mu,sd = fit.mean(0),fit.std(0).clip(.02)
    a,b = (fit-mu)/sd,(held-mu)/sd
    a /= np.linalg.norm(a,axis=1,keepdims=True).clip(1e-8)
    b /= np.linalg.norm(b,axis=1,keepdims=True).clip(1e-8)
    return a@a.T,b@a.T


def errors(gram,cross,fit,held,cfg):
    k = gram.copy()
    k.flat[::len(k)+1] += cfg["ridge"]
    y = fit["target"]
    mu = y.mean(0)
    pred = cross@np.linalg.solve(k,y-mu)+mu
    return np.linalg.norm(pred-held["target"],axis=1)


def by_episode(err,held):
    ep = held["episode"]
    return np.array([err[ep==g].mean() for g in np.unique(ep)])


def gain(base,candidate,cfg):
    diff = base-candidate
    rng = np.random.default_rng(91317)
    draws = rng.choice(diff,(cfg["bootstrap"],len(diff))).mean(1)
    ci = np.quantile(draws,[.025,.975])
    return {"mean_gain":float(diff.mean()),"relative_gain":float(diff.mean()/base.mean()),
            "paired_95_interval":ci.tolist(),"passes":bool(diff.mean()/base.mean()>=.05 and ci[0]>0)}


def run(cfg,spec,role):
    seed_all(81)
    root = Path(cfg["output"])
    step = spec["budgets"][role]
    out = root/"pusht"/role
    if (out/"results.json").exists():
        return
    if role=="scale" and not (root/"confirmation_lock.json").exists():
        raise RuntimeError("Full G3 endpoint lock required")
    archive = Path(cfg["data"])/"sources/pusht.zip"
    if digest(archive)!=spec["archive_sha256"]:
        raise ValueError("Unexpected replay archive")
    models = [(n,s) for s in cfg["seeds"] for n in cfg["profiles"]]
    lock = {"sources":{p.name:digest(p) for p in VERSION},"sha256":spec["archive_sha256"],"splits":splits(spec),"endpoints":{}}
    for name,seed in models:
        folder = root/f"{name}-{seed}"
        sha = digest(folder/f"step-{step}.pt")
        if sha!=read(folder/f"step-{step}.json")["delta_sha256"]:
            raise ValueError("Damaged endpoint")
        lock["endpoints"][folder.name] = sha
    path = out/"lock.json"
    if path.exists() and read(path)!=lock:
        raise RuntimeError("Replay protocol or weights changed after the lock")
    atomic(path,lock)
    replay = Replay(archive)
    fit,held = [replay.arrays(lock["splits"][r],spec) for r in ("fit",role)]
    kp,vp = kernel(fit["plan"],held["plan"])
    ep = held["episode"]
    rng = np.random.default_rng(6193)
    # Derange complete held episodes, matching each trajectory's window ordinal.
    group_order = rng.permutation(np.unique(ep))
    idx = np.arange(len(ep))
    for i,g in enumerate(group_order):
        other = np.flatnonzero(ep==group_order[(i+1)%len(group_order)])
        at = np.flatnonzero(ep==g)
        idx[at] = other[np.arange(len(at))%len(other)]
    outcomes = {"zero":by_episode(np.linalg.norm(held["target"],axis=1),held)}
    for name,seed in [("raw",0),*models]:
        model = None
        if name!="raw":
            folder = root/f"{name}-{seed}"
            model,parent = load_parent(cfg,name,seed)
            delta = torch.load(folder/f"step-{step}.pt",map_location="cpu",weights_only=True)
            if parent!=delta["parent_sha256"]:
                raise ValueError("Wrong language parent")
            model.apply_delta(delta["delta"])
            model.cuda().eval()
            del delta
        with torch.inference_mode():
            for mode in ("joint","image","numeric"):
                a,b = (representations(model,d,mode) for d in (fit,held))
                k,v = kernel(a,b)
                variants = {"plan":((k+kp)/2,(v+vp)/2)}
                if mode=="joint":
                    variants.update(no_plan=(k,v),shuffled=((k+kp)/2,(v+vp[idx])/2))
                for variant,(gram,cross) in variants.items():
                    key = f"{name}-{seed}/{mode}/{variant}"
                    outcomes[key] = by_episode(errors(gram,cross,fit,held,spec),held)
        del model
    comparisons,gates = {},{}
    for name,seed in models:
        key = f"{name}-{seed}"
        refs = ["zero","raw-0/joint/plan",key+"/image/plan",key+"/numeric/plan"]
        if name.startswith("signal"):
            refs.append(f"gpt2-{'wide' if name=='signal-wide' else 'dense'}-{seed}/joint/plan")
            if name=="signal-sparse":
                refs.append(f"gpt2-wide-{seed}/joint/plan")
        comparisons[key] = {ref:gain(outcomes[ref],outcomes[key+"/joint/plan"],spec) for ref in refs}
        gates[key] = all(v["passes"] for v in comparisons[key].values())
    atomic(out/"results.json",{"role":role,"step":step,"fit_episodes":len(set(fit["episode"])),"held_episodes":len(set(ep)),
        "fit_windows":len(fit["target"]),"held_windows":len(ep),"held_episode_order":np.unique(ep).tolist(),
        "mean_episode_error":{k:float(v.mean()) for k,v in outcomes.items()},"per_episode":{k:v.tolist() for k,v in outcomes.items()},
        "comparisons":comparisons,"passes":gates,"lock_sha256":digest(path)})
    print(json.dumps({k:round(float(v.mean()),2) for k,v in outcomes.items() if k=="zero" or "/joint/plan" in k}),flush=True)


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--role",required=True,choices=["screen","scale"])
    a = p.parse_args()
    with gpu():
        run(read(CONFIG),read(SPEC),a.role)
