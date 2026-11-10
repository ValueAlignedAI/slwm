"""A training-only acoustic codebook and a separately conditioned probabilistic unit-to-mel renderer."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from training.g4_data import digest


def fields(folder,names):
    return {k:np.load(folder/(k+".npy"),mmap_mode="r",allow_pickle=False) for k in names}


def put(path,obj):
    part=path.with_suffix(".new");part.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n");part.replace(path)


def codec(root):
    out=root/"units";out.mkdir(exist_ok=True); model=out/"codebook.npz"
    if not model.is_file():
        rng=np.random.default_rng(711);samples=[];pins={}
        for family in ("crema","libri"):
            folder=root/"speech"/family/"train";a=fields(folder,("h","length","group"))
            groups=np.unique(a["group"]); rows={int(g):np.flatnonzero(a["group"]==g) for g in groups}
            chosen=rng.choice(groups,65536);ix=np.array([int(rng.choice(rows[int(g)])) for g in chosen])
            clock=(rng.random(len(ix))*a["length"][ix]).astype(int); slot=rng.integers(2,size=len(ix))
            samples.append(a["h"][ix,clock,slot].astype(np.float32));pins[family]=digest(folder/"card.json")
        v=np.concatenate(samples);mean=v.mean(0);scale=np.maximum(v.std(0),1e-3);v=(v-mean)/scale
        centers=v[rng.choice(len(v),64,replace=False)].copy()
        for epoch in range(30):
            sums=np.zeros_like(centers);counts=np.zeros(64)
            for block in np.array_split(v,16):
                ix=(np.sum(centers**2,1)-2*block@centers.T).argmin(1)
                np.add.at(sums,ix,block);counts+=np.bincount(ix,minlength=64)
            centers=np.where(counts[:,None]>0,sums/np.maximum(counts[:,None],1),v[rng.integers(len(v),size=64)]).astype(np.float32)
        with model.with_suffix(".part").open("wb") as f:np.savez(f,mean=mean,scale=scale,centers=centers)
        model.with_suffix(".part").replace(model)
        put(out/"card.json",{"fitted_only_on":"train people","fitting_frames":len(v),"K":64,"views":"frozen wav2vec2 layer 6, projected to 32 dimensions",
             "teacher_revision":"22aad52d435eb6dbaf354bdad9b0da84ce7d6156","training_source_cards":pins,"sha256":digest(model)})
    with np.load(model) as f:mu,std,centers=(np.array(f[k]) for k in ("mean","scale","centers"))
    for family in ("libri","crema"):
        for role in ("train","dev","test"):
            src=root/"speech"/family/role; dest=out/family/role;dest.mkdir(parents=True,exist_ok=True)
            if (dest/"card.json").is_file():continue
            v=fields(src,("h","length"));n=len(v["h"])
            units=np.lib.format.open_memmap(dest/"units.npy",mode="w+",dtype=np.uint8,shape=(n,64,2));counts=np.zeros(64,np.int64)
            for j in range(0,n,256):
                x=(v["h"][j:j+256].astype(np.float32)-mu)/std
                y=(np.sum(centers**2,1)-2*x.reshape(-1,32)@centers.T).argmin(1).reshape(-1,64,2)
                valid=np.arange(64)[None]<v["length"][j:j+256,None]
                counts+=np.bincount(y[valid].ravel(),minlength=64); units[j:j+len(y)]=y
            units.flush();put(dest/"card.json",{"rows":n,"source":digest(src/"card.json"),"sha256":digest(dest/"units.npy"),"codebook":digest(model),"counts":counts.tolist()})
            print("Acoustic units",family,role,n,flush=True)


def style(waves,lengths):
    m=(np.arange(64)[None,:]<lengths[:,None]).astype(np.float32)
    bands=(waves.astype(np.float32)/16-12).reshape(len(waves),64,20,5).mean(-1)
    return np.concatenate([(bands*m[:,:,None]).sum(1)/np.maximum(1,m.sum(1,keepdims=True)),
                           np.sqrt(np.maximum(0,((bands**2)*m[:,:,None]).sum(1)/np.maximum(1,m.sum(1,keepdims=True))-
                                                    ((bands*m[:,:,None]).sum(1)/np.maximum(1,m.sum(1,keepdims=True)))**2))],1).astype(np.float32)


def renderer():
    import torch
    from torch import nn
    class Render(nn.Module):
        def __init__(self):
            super().__init__();self.emb=nn.Embedding(64,64);self.style=nn.Linear(40,24)
            self.rnn=nn.GRU(88,96,num_layers=2,batch_first=True,bidirectional=True)
            self.phase=nn.Parameter(torch.zeros(2,192));self.out=nn.Sequential(nn.Linear(192,192),nn.GELU(),nn.Linear(192,200))
        def forward(self,units,voice):
            x=self.emb(units); s=torch.tanh(self.style(voice))[:,None].expand(-1,x.shape[1],-1)
            z,_=self.rnn(torch.cat([x,s],-1)); h=(z[:,:,None]+self.phase).flatten(1,2)
            mu,lv=self.out(h).chunk(2,dim=-1)
            return mu,lv.clamp(-4,4)
    return Render()


def train_renderer(root,out,steps):
    import torch
    torch.set_num_threads(2);torch.manual_seed(711)
    if not torch.cuda.is_available():raise RuntimeError("Expected CUDA")
    out.mkdir(exist_ok=True,parents=True)
    sets={};poolstyles=[]; transfers={}
    for family in ("libri","crema"):
        a=fields(root/"speech"/family/"train",("a","fine","length","group"));a["units"]=np.load(root/"units"/family/"train/units.npy",mmap_mode="r")
        styles=np.concatenate([style(a["a"][j:j+256],a["length"][j:j+256]) for j in range(0,len(a["a"]),256)])
        poolstyles.append(styles.mean(0));ref=np.zeros(len(styles),int)
        for g in np.unique(a["group"]):
            ids=np.flatnonzero(a["group"]==g);ref[ids]=np.roll(ids,1)
        a["style"]=styles;a["ref"]=ref;sets[family]=a
    default=np.mean(poolstyles,0).astype(np.float32);net=renderer().cuda();opt=torch.optim.AdamW(net.parameters(),lr=.001)
    pending=out/"resume.pt"
    begin=0
    if pending.exists():
        ck=torch.load(pending,map_location="cuda",weights_only=True);net.load_state_dict(ck["net"]);opt.load_state_dict(ck["opt"]);begin=ck["step"]
    def gpu(x):return torch.from_numpy(np.asarray(x)).cuda()
    for step in range(begin,steps):
        rng=np.random.default_rng([711,step]); samples=[]
        for family,count in (("libri",48),("crema",16)):
            a=sets[family];ix=rng.integers(len(a["length"]),size=count)
            voice=a["style"][a["ref"][ix]].copy();voice[rng.random(count)<.5]=default
            samples.append((a["units"][ix].reshape(count,128),a["fine"][ix].reshape(count,256,100).astype(np.float32)/16-12,
                            a["length"][ix].astype(np.int32),voice))
        units,y,n,s=(gpu(np.concatenate(t)) for t in zip(*samples))
        mu,lv=net(units.long(),s); m=(torch.arange(256,device="cuda")[None]<4*n[:,None]).float()
        # Normalized physical log-mel scale, decoded at inference using the exact inverse.
        target=(y+5)/3
        err=mu-target
        loss=(((err.square()*torch.exp(-lv)+lv)*.5+.25*err.abs()).mean(-1)*m).sum()/m.sum()
        loss=loss+.12*(torch.diff(err,dim=1).abs().mean(-1)*m[:,1:]).sum()/m[:,1:].sum()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        if not torch.isfinite(loss):raise FloatingPointError(step)
        if (step+1)%200==0 or step+1==steps:
            ck={"net":net.state_dict(),"opt":opt.state_dict(),"step":step+1,"default_style":gpu(default).cpu(),"codebook":digest(root/"units/codebook.npz")}
            torch.save(ck,out/"resume.part");(out/"resume.part").replace(pending)
            print("Renderer",step+1,float(loss.detach()),flush=True)
    torch.save({"net":net.state_dict(),"default_style":gpu(default).cpu(),"codebook":digest(root/"units/codebook.npz")},out/"renderer.part")
    (out/"renderer.part").replace(out/"renderer.pt")
    put(out/"complete.json",{"updates":steps,"parameters":sum(p.numel() for p in net.parameters()),"weights_sha256":digest(out/"renderer.pt"),
          "optimizer_sha256":digest(pending),"style":"Other training utterance of same person or pooled neutral style. Neutral style is the primary evaluation."})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=("units","renderer"));p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path);p.add_argument("--steps",type=int,default=2000);a=p.parse_args()
    if a.phase=="units":codec(a.root)
    else:train_renderer(a.root,a.out,a.steps)
