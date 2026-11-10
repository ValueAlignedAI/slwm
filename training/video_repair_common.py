"""Verified paired arrays, support-aware gaps and consistent controls."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from training.restoration_train import load_codec
from training.t2_prepare_cremad import sha256
from models.slwm_video_repair import gap_masks


def load_data(cfg,split,device="cuda"):
    root=Path(cfg["corpus"]); path=root/(split+"_manifest.json"); card=json.loads(path.read_text())
    if sha256(root/(split+".json"))!=card["metadata_sha256"]: raise ValueError("Metadata changed")
    data={}
    for name,item in card["arrays"].items():
        if sha256(Path(item["path"]))!=item["sha256"]: raise ValueError("Data changed")
        value=np.load(item["path"],allow_pickle=False)
        data[name]=value if name=="lengths" else torch.from_numpy(value).to(device)
    data["rows"]=json.loads((root/(split+".json")).read_text()); data["manifest_sha256"]=sha256(path)
    wrong=[]
    for i,row in enumerate(data["rows"]):
        pool=[j for j,r in enumerate(data["rows"]) if r["actor"]==row["actor"] and r["sentence"]!=row["sentence"]]
        if not pool: raise ValueError("Different-utterance control absent")
        wrong.append(pool[int(hashlib.sha256(row["sample_id"].encode()).hexdigest()[:8],16)%len(pool)])
    data["wrong"]=np.array(wrong); return data


def crop(data,ids,starts,window=32):
    device=data["mel100"].device; i=torch.as_tensor(ids,device=device)[:,None]; t=torch.as_tensor(starts,device=device)[:,None]+torch.arange(window,device=device)[None]
    return {k:data[k][i,t].float() if k!="mouth" else data[k][i,t] for k in ("mel100","mouth","video")}


def get_codec(cfg,device="cuda"):
    return load_codec(json.loads(Path(cfg["restoration_config"]).read_text()),device)


def make_masks(batch,gap,device,starts=None):
    length=torch.full((batch,),gap,device=device,dtype=torch.long)
    begin=torch.full_like(length,(32-gap)//2) if starts is None else torch.as_tensor(starts,device=device)
    return gap_masks(begin,length),begin,length


def hole_errors(pred,target,mask):
    error=(pred.float()-target.float()).square().mean(-1)
    return (error*mask).sum(1)/mask.sum(1).clamp_min(1)
