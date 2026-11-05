"""Memory-bounded verified data, physical controls, and semantic losses."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from training.research_sources import file_sha
from training.restoration_train import load_codec
from models.slwm_video_repair import gap_masks


def load_data(cfg,split,device="cuda"):
    legacy=split.startswith("crema_"); libri=split.startswith("libri_")
    if libri:
        parent=json.loads(Path(cfg["restoration_config"]).read_text()); root=Path(parent["new_corpus"]); card=json.loads((root/(split+"_manifest.json")).read_text())
        for name,digest in card["files"].items():
            if file_sha(root/name)!=digest: raise ValueError("Legacy data changed")
        paths={k:{"path":str(root/(split+"_"+k+".npy"))} for k in ("mel100","lengths")}
    else:
        root=Path(cfg["legacy_video"] if legacy else cfg["corpus"]); card=json.loads((root/(split+"_manifest.json")).read_text()); paths=card["arrays"]
        for item in paths.values():
            if file_sha(item["path"])!=item["sha256"]: raise ValueError("Paired data changed")
        if file_sha(root/(split+".json"))!=card["metadata_sha256"]: raise ValueError("Metadata changed")
    rows=json.loads((root/(split+".json")).read_text()); result={"rows":rows,"lengths":np.load(paths["lengths"]["path"]),"manifest_sha256":file_sha(root/(split+"_manifest.json")),"split":split}
    result["mel100"]=torch.as_tensor(np.load(paths["mel100"]["path"],allow_pickle=False),device=device)
    result["mouth"]=np.load(paths["mouth"]["path"],mmap_mode="r") if "mouth" in paths else None
    result["labels"]=torch.as_tensor(np.load(paths["labels"]["path"]),device=device).long() if "labels" in paths else None
    pools={actor:[i for i,r in enumerate(rows) if r["actor"]==actor] for actor in sorted({r["actor"] for r in rows})}; wrong=[]
    for row in rows:
        pool=[i for i in pools[row["actor"]] if rows[i].get("sentence",rows[i]["transcript"])!=row.get("sentence",row["transcript"])]
        wrong.append(pool[int(hashlib.sha256(row["sample_id"].encode()).hexdigest()[:8],16)%len(pool)] if pool else 0)
    result["wrong"]=np.asarray(wrong); return result


def crop(data,ids,starts):
    device=data["mel100"].device; t=np.asarray(starts)[:,None]+np.arange(32)[None]; i=np.asarray(ids)[:,None]; ti=torch.as_tensor(t,device=device); ii=torch.as_tensor(i,device=device)
    y=data["mel100"][ii,ti].float(); mouth=torch.as_tensor(np.asarray(data["mouth"][i,t]).copy(),device=device) if data["mouth"] is not None else torch.zeros(len(ids),32,64,96,device=device,dtype=torch.uint8)
    labels=data["labels"][ii,ti] if data["labels"] is not None else torch.full((len(ids),32),-100,device=device,dtype=torch.long)
    return {"mel100":y,"mouth":mouth,"labels":labels}


def get_codec(cfg,device="cuda"):
    return load_codec(json.loads(Path(cfg["restoration_config"]).read_text()),device)


def make_masks(batch,gap,device,starts=None):
    lengths=torch.full((batch,),gap,device=device,dtype=torch.long); begin=torch.full_like(lengths,(32-gap)//2) if starts is None else torch.as_tensor(starts,device=device)
    return gap_masks(begin,lengths)


def unit_loss(logits,labels):
    valid=labels>=0
    if not valid.any(): return logits.sum()*0
    weight=torch.where(labels[valid]==0,.1,1.)
    return (F.cross_entropy(logits[valid].float(),labels[valid],reduction="none")*weight).sum()/weight.sum()


def hole_mse(pred,target,mask):
    error=(pred.float()-target.float()).square().mean(-1)
    return (error*mask).sum(1)/mask.sum(1).clamp_min(1)


def repair_loss(pred,target,masks):
    return hole_mse(pred,target,masks["target"]).mean()+.25*hole_mse(pred,target,masks["unknown"]).mean()+.1*F.l1_loss(pred[:,1:]-pred[:,:-1],target[:,1:]-target[:,:-1])


def control_mouth(data,ids,starts,batch,mode,gap=8):
    if mode=="wrong":
        other=data["wrong"][ids]; return crop(data,other,(data["lengths"][other]-32)//2)["mouth"]
    if mode.startswith("shift"):
        offset=int(mode.removeprefix("shift")); moved=np.asarray(starts)+offset
        if (moved<0).any() or (moved+32>data["lengths"][ids]).any(): raise ValueError("Shift must use real neighboring frames")
        return crop(data,ids,moved)["mouth"]
    mouth=batch["mouth"].clone()
    if mode=="static": mouth=mouth[:,(32-gap)//2-1:(32-gap)//2].expand_as(mouth).clone()
    elif mode=="blank": mouth.zero_()
    return mouth
