"""Predict acoustic codes using frozen shared states with single-modality input and no target acoustics."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from training.g8_acoustic_codec import DATA,OLD,OUT
from training.g8_models import read
from training.g4_data import digest
from training.g7_units import put

HEADS=OUT.parent/"acoustic-heads"


class Head(nn.Module):
    def __init__(self):
        super().__init__();self.norm=nn.LayerNorm(128);self.temporal=nn.Conv1d(128,192,5)
        self.mix=nn.Sequential(nn.GELU(),nn.Linear(192,128),nn.LayerNorm(128))
        self.output=nn.Linear(128,2*8*64)
    def forward(self,z):
        x=self.norm(z);temporal=self.temporal(F.pad(x.transpose(1,2),(4,0))).transpose(1,2)
        return self.output(x+self.mix(temporal)).reshape(*x.shape[:2],2,8,64)


class Train:
    def __init__(self,tag,family):
        folder=OLD/"align-features"/tag/family/"train";card=read(folder/"card.json")
        self.idx=np.load(folder/"idx.npy");self.parent=card["parent_sha256"]
        self.modes=card["views"];self.fields={m:np.load(folder/f"z{m}-{'mask' if m==0 else 'clean'}.npy",mmap_mode="r") for m in self.modes}
        self.y=np.load(DATA/"codes"/family/"train.npy",mmap_mode="r")
        self.length=np.load(OLD/"speech"/family/"train/length.npy",mmap_mode="r")
        if read(DATA/"codes"/family/"card.json")["codec"]!=digest(OUT/"codec.npz"):raise ValueError("Mismatched target tokenizer")
    def sample(self,rng,n):
        rows=rng.integers(len(self.idx),size=n);modes=np.resize(self.modes,n);rng.shuffle(modes)
        z=np.empty((n,64,128),np.float32)
        for m in self.modes:
            ix=np.flatnonzero(modes==m);z[ix]=self.fields[m][rows[ix]].astype(np.float32)
        physical=self.idx[rows];mask=np.arange(64)[None]<self.length[physical,None]
        return z,self.y[physical],mask


def fit(tag):
    out=HEADS/tag;out.mkdir(parents=True,exist_ok=True)
    if (out/"complete.json").exists():return
    torch.set_num_threads(2);seed=int(tag.split("-")[1]);torch.manual_seed(seed)
    sets=[Train(tag,f) for f in ("crema","libri")];assert sets[0].parent==sets[1].parent
    model=Head().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.00065,weight_decay=.01)
    start=0;ck=out/"resume.pt";codec=digest(OUT/"codec.npz")
    if ck.exists():
        d=torch.load(ck,map_location="cuda",weights_only=True)
        if d["parent"]!=sets[0].parent or d["codec"]!=codec:raise ValueError("Acoustic readout lock changed")
        model.load_state_dict(d["model"]);opt.load_state_dict(d["opt"]);start=d["step"]
    for step in range(start,2048):
        rng=np.random.default_rng([81813,step]);batch=[s.sample(rng,32) for s in sets]
        z,y,m=(torch.as_tensor(np.concatenate(v),device="cuda") for v in zip(*batch))
        output=model(z);ce=F.cross_entropy(output.reshape(-1,64),y.long().reshape(-1),reduction="none").reshape(64,64,2,8).mean((2,3))
        loss=(ce*m).sum()/m.sum().clamp_min(1)
        if not torch.isfinite(loss):raise FloatingPointError(step)
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        if (step+1)%256==0:
            torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"step":step+1,"parent":sets[0].parent,"codec":codec},out/"resume.part");(out/"resume.part").replace(ck)
    torch.save(model.state_dict(),out/"weights.part");(out/"weights.part").replace(out/"weights.pt")
    put(out/"complete.json",{"updates":2048,"patterns_per_update":64,"parent":sets[0].parent,"codec":codec,"weights":digest(out/"weights.pt"),
         "optimizer":digest(ck),"params":sum(w.numel() for w in model.parameters()),"training_cache_rows":{"crema":len(sets[0].idx),"libri":len(sets[1].idx)}})
    print("Frozen candidate",tag,digest(out/"weights.pt"),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("tag");fit(p.parse_args().tag)
