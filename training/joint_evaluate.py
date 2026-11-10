"""Frozen few-shot transfer, input-only controls and joint-language retention."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from training.joint_benchmarks import benchmark
from training.joint_data import CORPUS, JointData, read
from training.joint_sources import CONFIG, digest
from training.joint_g2_anchor import OldState
from training.joint_train import atomic, gpu, load_parent, seed_all, tensors
from training.scale_text_train import UniqueWindows

TASKS = ("fsdd","fashion","banking")
EVAL_FILES = ["training/joint_evaluate.py","training/joint_benchmarks.py","training/joint_g2_anchor.py","training/joint_study.py","training/joint_results.py",
              "configs/t4/joint_transfer.json","docs/research/joint_transfer_20261109.md"]


def field_views(state, lengths):
    """Causal [B,T,D] -> four ordered averages [B,4D] and last valid [B,D]."""
    device = state.device
    time = torch.arange(state.shape[1],device=device)[None,:]
    bins = []
    for part in range(4):
        a = lengths*part//4
        b = torch.maximum(lengths*(part+1)//4,a+1)
        mask = (time>=a[:,None]) & (time<b[:,None])
        bins.append((state*mask[...,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None])
    return {"ordered":torch.cat(bins,1),"last":state[torch.arange(len(state),device=device),lengths-1]}


def raw_features(arrays):
    if "text" in arrays:
        out = np.zeros((len(arrays["text"]),2048),np.float32)
        for i,(row,n) in enumerate(zip(arrays["text"],arrays["lengths"])):
            np.add.at(out[i],row[:n]%2048,1)
        return np.log1p(out)
    key = "audio" if "audio" in arrays else "image"
    return arrays[key].reshape(len(arrays[key]),-1).astype(np.float32)


def features(model, arrays):
    if model is None:
        return {"raw":raw_features(arrays)}
    if isinstance(model,OldState):
        return model.representations(arrays)
    buffers = {"ordered":[],"last":[]}
    for i in range(0,len(arrays["labels"]),128):
        batch = tensors({k:v[i:i+128] for k,v in arrays.items() if k in ("text","image","audio","lengths")},"cuda")
        with torch.autocast("cuda",dtype=torch.bfloat16):
            z,_,_ = model.field(batch,batch.get("lengths"))
        lengths = batch.get("lengths",torch.full((len(z),),z.shape[1],device=z.device))
        for k, val in field_views(z.float(),lengths).items():
            buffers[k].append(val.cpu().numpy())
    return {k:np.concatenate(v) for k,v in buffers.items()}


def ridge(x,y,v,classes,penalty):
    """Fit only on labelled support with one fixed ridge penalty and intercept."""
    x,v = x.astype(np.float32),v.astype(np.float32)
    mu, std = x.mean(0),x.std(0).clip(.1)
    x,v = (x-mu)/std,(v-mu)/std
    # Use unit-normalized kernels so the penalty has a common interpretation.
    x /= np.linalg.norm(x,axis=1,keepdims=True).clip(1e-6)
    v /= np.linalg.norm(v,axis=1,keepdims=True).clip(1e-6)
    labels = np.eye(classes,dtype=np.float32)[y]
    base = labels.mean(0)
    kernel = x @ x.T
    kernel.flat[::len(x)+1] += penalty
    coefficients = np.linalg.solve(kernel,labels-base)
    return ((v@x.T)@coefficients+base).argmax(1), (mu,std,x,coefficients,base)


def score(pred, labels):
    return float(np.mean([np.mean(pred[labels==k]==k) for k in np.unique(labels)]))


def write_npz(path,values):
    path.parent.mkdir(exist_ok=True,parents=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp,**values)
    tmp.replace(path)


def task_scores(cfg, model, out, role, task):
    fit, held = (benchmark(cfg,task,x) for x in ("fit",role))
    x, v = features(model,fit),features(model,held)
    classes = len(np.unique(fit["labels"]))
    predictions = {"labels":held["labels"],"groups":held["groups"],"ids":held["ids"]}
    scores = {}
    for n in cfg["few_shot"]:
        idx = np.concatenate([np.flatnonzero(fit["labels"]==k)[:n] for k in range(classes)])
        if len(idx)!=n*classes:
            raise ValueError("Incomplete few-shot support")
        for name in x:
            pred,fitted = ridge(x[name][idx],fit["labels"][idx],v[name],classes,cfg["ridge"])
            key = f"{name}-{n}"
            predictions[key] = pred
            scores[key] = {"balanced_accuracy":score(pred,held["labels"]),"support":len(idx)}
            if task=="fsdd":
                scores[key]["by_speaker"] = {s:score(pred[held["groups"]==s],held["labels"][held["groups"]==s]) for s in np.unique(held["groups"])}
            if role=="test" and n==32 and name=="ordered":
                # Lossless dual coefficients allow applying the same frozen head.
                write_npz(out/f"{task}-32shot-head.npz",dict(zip(("mean","std","support","coefficients","intercept"),fitted)))
    write_npz(out/f"{task}-{role}-predictions.npz",predictions)
    return {"examples":len(held["labels"]),"classes":classes,"scores":scores}


def retention(cfg,model):
    source = UniqueWindows(CORPUS/"validation.tokens.bin",cfg["text_length"],6097)
    values = []
    for i in range(8):
        x,y = source.batch(1600+i*16,16,"cuda")
        with torch.autocast("cuda",dtype=torch.bfloat16):
            z,_,_ = model.field({"text":x})
            loss = F.cross_entropy(F.linear(z,model.core.token_embedding.weight).flatten(0,1),y.flatten())
        values.append(float(loss))
    return {"nll":float(np.mean(values)),"tokens":8*16*cfg["text_length"],"context":cfg["text_length"]}


def retrieval(cfg,model):
    source = JointData(cfg,83,"selection")
    buffers = {f:[[],[],[],[]] for f in ("coco","clotho")}
    for step in range(100):
        family, occurrence = source.position(step)
        if family not in buffers or occurrence>=4:
            continue
        _,batch,ids = source.get(step)
        # strings are represented by exact observed token sequences to count duplicates
        caption_ids = [tuple(row[:n]) for row,n in zip(batch["caption"],batch["lengths"])]
        b = tensors(batch,"cuda")
        with torch.autocast("cuda",dtype=torch.bfloat16):
            z,_,_ = model.field(b)
            text,_,_ = model.field({"text":b["caption"]},b["lengths"])
            from models.joint_transfer import pool
            l,r = model.align(z.mean(1)),model.align(pool(text,b["lengths"]))
        a,c,ident,keys = buffers[family]
        a.append(F.normalize(l.float(),dim=-1).cpu())
        c.append(F.normalize(r.float(),dim=-1).cpu())
        ident.extend(ids)
        keys.extend(caption_ids)
    result = {}
    for f,(a,b,ids,keys) in buffers.items():
        match = np.array([[i==j or x==y for j,y in zip(ids,keys)] for i,x in zip(ids,keys)])
        sim = (torch.cat(a)@torch.cat(b).T).numpy()
        k = np.arange(len(ids))
        perm = np.random.default_rng(2647).permutation(len(k))
        result[f] = {"candidates":len(k),"r1":float(match[k,sim.argmax(1)].mean()),
                     "shuffled_r1":float(match[k,perm[sim.argmax(1)]].mean()),"chance":float(match.mean())}
    return result


def evaluate(cfg,name,seed,step,role):
    seed_all(seed)
    root = Path(cfg["output"])
    version = {f:digest(Path(f)) for f in EVAL_FILES}
    lock = root/"confirmation_lock.json"
    if role=="test" and (not lock.exists() or read(lock)["evaluation"]!=version):
        raise RuntimeError("Final evaluation source lock required")
    folder = root/f"{name}-{seed}"
    out = folder/f"eval-{step}"
    marker = out/f"{role}.json"
    if marker.exists():
        saved = read(marker)
        if saved["evaluation_sources"]!=version:
            raise ValueError("Existing evaluation used another recipe")
        return
    if name.startswith("G2-"):
        model = OldState(name)
        parent = model.sha
    else:
        model, parent = (None,None) if name=="raw" else load_parent(cfg,name,seed)
    weight = parent
    if step:
        checkpoint = folder/f"step-{step}.pt"
        weight = digest(checkpoint)
        if weight!=read(folder/f"step-{step}.json")["delta_sha256"]:
            raise ValueError("Endpoint checksum mismatch")
        if role=="test" and read(lock)["endpoints"][folder.name]!=weight:
            raise ValueError("Confirmation endpoint mismatch")
        delta = torch.load(checkpoint,map_location="cpu",weights_only=True)
        if delta["parent_sha256"]!=parent:
            raise ValueError("Wrong inference parent")
        model.apply_delta(delta["delta"])
    if model is not None:
        model.cuda().eval()
    with torch.inference_mode():
        results = {t:task_scores(cfg,model,out,role,t) for t in TASKS}
        diagnostics = {"language":retention(cfg,model),"retrieval":retrieval(cfg,model)} if model is not None and not isinstance(model,OldState) else {}
    atomic(marker,{"profile":name,"seed":seed,"step":step,"role":role,"weights_sha256":weight,
                   "evaluation_sources":version,"tasks":results,"diagnostics":diagnostics})
    print(json.dumps({t:v["scores"].get("ordered-32",v["scores"].get("raw-32")) for t,v in results.items()}),flush=True)


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--profile",required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--step",type=int,required=True)
    p.add_argument("--role",choices=["dev","test"],default="dev")
    a=p.parse_args()
    with gpu():
        evaluate(read(CONFIG),a.profile,a.seed,a.step,a.role)
