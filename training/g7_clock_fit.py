"""One-sided versus sign-balanced physical timing supervision with fixed common encoder clocks."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from training.g7_alignment import Readout,meta,gpu,save
from training.g4_data import digest
from training.g7_sync_audit import compare
from scripts.train.g7_lock_heads import lock


class Matcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.edges=nn.ModuleList([nn.Sequential(nn.LayerNorm(128),nn.Linear(128,72),nn.GELU()) for _ in range(2)])
        self.time=nn.ModuleList([nn.Conv1d(72,48,3,padding=1) for _ in range(2)])
    def project(self,x,m,side):
        z=self.edges[side](x);z=self.time[side]((z*m[...,None]).transpose(1,2)).transpose(1,2)
        z=z-(z*m[...,None]).sum(1,keepdim=True)/m.sum(1,keepdim=True).clamp_min(1)[...,None]
        return F.normalize(z,dim=-1)
    def forward(self,z,m):
        n,k,t,d=z.shape;m=m.float();q=self.project(z[:,0],m,0)
        keep=m[:,None].expand(n,k-1,t).reshape(-1,t)
        v=self.project(z[:,1:].reshape(-1,t,d),keep,1).reshape(n,k-1,t,48)
        return 18*((q[:,None]*v).sum(-1)*m[:,None]).sum(-1)/m.sum(-1,keepdim=True).clamp_min(1)


def datasets(root,tag,role):
    result={}
    for family in ("crema","events"):
        path=root/"physical-time/cache"/tag/family/role
        with np.load(path/"rows.npz") as f:rows={k:np.array(v) for k,v in f.items()}
        result[family]=(np.load(path/"field.npy",mmap_mode="r"),rows,meta(path/"card.json"))
    return result


def freeze(out):
    bits={}
    for seed in (711,712):
        for core in ("raw","hybrid","gpt2"):
            for recipe in ("older-only","both-signs"):
                tag=f"{core}-{seed}/{recipe}";f=out/tag
                d=meta(f/"complete.json");h=digest(f/"weights.pt")
                if h!=d["weights"] or d["updates"]!=1536:raise ValueError("Changed timing head")
                bits[tag]=h
    lock(out/"confirm-lock.json",bits);return bits


def run(a):
    tag=f"{a.core}-{a.seed}";dest=a.out/tag/a.recipe;dest.mkdir(exist_ok=True,parents=True)
    if a.phase=="lock":freeze(a.out);return
    torch.set_num_threads(2);torch.manual_seed(a.seed);model=Matcher().cuda()
    if a.phase=="train":
        if (dest/"complete.json").exists():return
        sets=datasets(a.root,tag,"train");opt=torch.optim.AdamW(model.parameters(),lr=.00065,weight_decay=.01)
        ck=dest/"resume.pt";first=0
        if ck.exists():
            last=torch.load(ck,map_location="cuda",weights_only=True);model.load_state_dict(last["net"]);opt.load_state_dict(last["opt"]);first=last["step"]
        for step in range(first,1536):
            rng=np.random.default_rng([712112,step]);zs=[];ms=[]
            for f,(v,row,_) in sets.items():
                ix=rng.integers(len(v),size=32);samples=np.array(v[ix],np.float32);sign=rng.integers(2,size=32)
                cols=np.where(sign==0,2,3) if a.recipe=="both-signs" else np.full(32,2)
                zs.append(np.stack([samples[:,0],samples[:,1],samples[np.arange(32),cols],samples[:,4]],1));ms.append(row["mask"][ix])
            logits=model(gpu(np.concatenate(zs)),gpu(np.concatenate(ms)));loss=F.cross_entropy(logits,torch.zeros(64,dtype=torch.long,device="cuda"))
            if not torch.isfinite(loss):raise FloatingPointError(step)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            if (step+1)%256==0:
                torch.save({"net":model.state_dict(),"opt":opt.state_dict(),"step":step+1},dest/"resume.part");(dest/"resume.part").replace(ck)
        torch.save(model.state_dict(),dest/"weights.part");(dest/"weights.part").replace(dest/"weights.pt")
        save(dest/"complete.json",{"updates":1536,"weights":digest(dest/"weights.pt"),"optimizer":digest(ck),"matched_train_rows":98304,
            "training_sets":{f:d for f,(_,_,d) in sets.items()},"parameters":sum(p.numel() for p in model.parameters())});return
    if (dest/"confirm.json").exists():return
    hashes=freeze(a.out)
    if digest(dest/"weights.pt")!=hashes[tag+"/"+a.recipe]:raise ValueError("Test lock changed")
    model.load_state_dict(torch.load(dest/"weights.pt",weights_only=True,map_location="cuda"));model.eval();sets=datasets(a.root,tag,"confirm");scores={}
    old=None
    if a.core!="raw":
        head=a.out.parent/"alignment"/tag/"units-sync/weights.pt";record=meta(a.out.parent/"alignment/test-lock.json")["models"][tag+"/units-sync"]
        if digest(head)!=record["sha256"]:raise ValueError("Original timing head changed")
        old=Readout().cuda().eval();old.load_state_dict(torch.load(head,map_location="cuda",weights_only=True)["weights"])
    with torch.inference_mode():
        for family,(v,row,card) in sets.items():
            logits=[];former=[]
            for j in range(0,len(v),64):
                z=gpu(np.array(v[j:j+64],np.float32));m=gpu(row["mask"][j:j+64]);logits.extend(model(z,m).cpu().numpy())
                if old is not None:former.extend(torch.stack([old.match(z[:,0],z[:,k],m) for k in (1,2,3,4)],1).cpu().numpy())
            scores[family]={"trained":compare(np.asarray(logits),row["group"]),"raw_flux_motion":compare(row["raw"],row["group"]),
                            "original_on_physical_inputs":compare(np.array(former),row["group"]) if former else None,"card":card}
            np.savez_compressed(dest/(family+"-predictions.npz"),scores=np.array(logits),group=row["group"],uid=row["uid"])
    save(dest/"confirm.json",{"tag":tag,"recipe":a.recipe,"weights":hashes[tag+"/"+a.recipe],"scores":scores})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=("train","lock","confirm"));p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--core",required=True);p.add_argument("--seed",type=int,required=True)
    p.add_argument("--recipe",required=True,choices=("older-only","both-signs"));run(p.parse_args())
