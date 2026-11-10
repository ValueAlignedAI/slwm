"""Matched conditional regression into continuous acoustic coefficients followed by the frozen quantizer."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from training.g8_acoustic_codec import OLD,DATA
from training.g8_acoustic_head import Head
from training.g8_rate_codec import OUT_RATE,load_book,coefficients
from training.g8_models import read
from training.g7_units import put
from scripts.train.g7_lock_heads import sha

HEADS=OUT_RATE.parent/"rate-heads"


class ContinuousHead(Head):
    def __init__(self):
        super().__init__();self.output=nn.Linear(128,128)
    def forward(self,z):
        import torch.nn.functional as F
        x=self.norm(z);temporal=self.temporal(F.pad(x.transpose(1,2),(4,0))).transpose(1,2)
        return self.output(x+self.mix(temporal)).reshape(*x.shape[:2],2,64)


def labels():
    book=load_book()
    for family in ("libri","crema"):
        src=OLD/"speech"/family/"train";dst=DATA/"coefficients"/family;dst.mkdir(parents=True,exist_ok=True)
        if (dst/"card.json").exists():continue
        fine=np.load(src/"fine.npy",mmap_mode="r")
        out=np.lib.format.open_memmap(dst/"train.partial.npy",mode="w+",dtype=np.float16,shape=(len(fine),64,2,64))
        for i in range(0,len(fine),96):
            x=fine[i:i+96];out[i:i+len(x)]=(coefficients(x,book)/book["std"]).reshape(len(x),64,2,64).astype(np.float16)
        out.flush();(dst/"train.partial.npy").replace(dst/"train.npy")
        put(dst/"card.json",{"source":sha(src/"card.json"),"codec":sha(OUT_RATE/"weights.npz"),"target_sha256":sha(dst/"train.npy")})


def fit(tag):
    folder=HEADS/tag;folder.mkdir(parents=True,exist_ok=True)
    if (folder/"complete.json").exists():return
    torch.set_num_threads(2);seed=int(tag.split("-")[1]);torch.manual_seed(seed)
    model=ContinuousHead().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.00065,weight_decay=.01)
    sets=[]
    for f in ("crema","libri"):
        root=OLD/"align-features"/tag/f/"train";c=read(root/"card.json");idx=np.load(root/"idx.npy")
        z={k:np.load(root/f"z{k}-{'mask' if k==0 else 'clean'}.npy",mmap_mode="r") for k in c["views"]}
        labs=np.load(DATA/"coefficients"/f/"train.npy",mmap_mode="r");length=np.load(OLD/"speech"/f/"train/length.npy",mmap_mode="r")
        sets.append((idx,z,labs,length,c["parent_sha256"]))
    parent=sets[0][-1];assert parent==sets[1][-1];pin=sha(OUT_RATE/"weights.npz")
    ck=folder/"optimizer.pt";first=0
    if ck.exists():
        d=torch.load(ck,map_location="cuda",weights_only=True)
        if (d["parent"],d["codec"])!=(parent,pin):raise ValueError("Wrong continuous-head checkpoint")
        model.load_state_dict(d["model"]);opt.load_state_dict(d["opt"]);first=d["step"]
    b=load_book();weight=torch.tensor(b["std"]/b["std"].mean(),device="cuda").sqrt()
    for step in range(first,2048):
        rng=np.random.default_rng([81813,step]);parts=[]
        for idx,z,labs,length,_ in sets:
            ix=rng.integers(len(idx),size=32);mode=np.resize(list(z),32);rng.shuffle(mode)
            feat=np.empty((32,64,128),np.float32)
            for k,v in z.items():feat[mode==k]=v[ix[mode==k]].astype(np.float32)
            rows=idx[ix];parts.append((feat,labs[rows].astype(np.float32),np.arange(64)[None]<length[rows,None]))
        x,y,m=(torch.tensor(np.concatenate(v),device="cuda") for v in zip(*parts))
        error=model(x)-y;err=(error.square()*weight).mean((2,3))
        loss=(err*m).sum()/m.sum();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if not torch.isfinite(loss):raise FloatingPointError(step)
        if (step+1)%256==0:
            torch.save({"step":step+1,"model":model.state_dict(),"opt":opt.state_dict(),"parent":parent,"codec":pin},folder/"resume.part");(folder/"resume.part").replace(ck)
    torch.save(model.state_dict(),folder/"weights.part");(folder/"weights.part").replace(folder/"weights.pt")
    put(folder/"complete.json",{"parent":parent,"codec":pin,"weights":sha(folder/"weights.pt"),"optimizer":sha(ck),"updates":2048,
        "parameters":sum(t.numel() for t in model.parameters()),"loss":"Masked valid-frame MSE of whitened PCA coefficients weighted by square root of relative component SD."})
    print("Frozen continuous head",tag,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase");a=p.parse_args()
    if a.phase=="labels":labels()
    else:fit(a.phase)
