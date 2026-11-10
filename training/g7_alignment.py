"""Matched reconstruction and masked-unit/synchrony fine-tuning of a common frozen G6 field."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def meta(path):return json.loads(path.read_bytes())
def save(path,obj):
    tmp=path.with_suffix(".new");tmp.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+"\n");tmp.replace(path)


class Views:
    def __init__(self,root,old,tag,family,role):
        self.family=family;self.speech=family in ("crema","libri")
        folder=(root/"speech"/family/role if self.speech else old/"impact"/role if family=="oldimpact" else root/"fresh/events"/role)
        self.src={k:np.load(folder/(k+".npy"),mmap_mode="r") for k in meta(folder/"card.json")["sha256"]}
        cache=root/"align-features"/tag/family/role; self.card=meta(cache/"card.json");self.ix=np.load(cache/"idx.npy")
        self.z={(m,masked):np.load(cache/f"z{m}-{'mask' if masked else 'clean'}.npy",mmap_mode="r") for m in self.card["views"] for masked in (False,True)}
        self.units=np.load(root/"units"/family/role/"units.npy",mmap_mode="r") if self.speech else None
        self.length=self.src["length"][self.ix].astype(int) if self.speech else np.full(len(self.ix),64,int)
        self.groups=self.src["group"][self.ix]
        self.ids=self.src["uid"][self.ix] if "uid" in self.src else self.ix.astype(np.uint64)
        label=self.src["phrase"][self.ix] if self.speech else self.src["recording"][self.ix]
        self.wrong=np.full(len(self.ix),-1,int)
        self.ref=np.full(len(self.ix),-1,int)
        for g in np.unique(self.groups):
            local=np.flatnonzero(self.groups==g); self.ref[local]=np.roll(local,1)
            # A different target label prevents accidentally treating an equivalent utterance as a negative.
            by={lv:local[label[local]!=lv] for lv in np.unique(label[local])}
            for i in local:
                possible=by[label[i]]
                if len(possible):self.wrong[i]=possible[int(self.ids[i])%len(possible)]
        self.positive=np.flatnonzero(self.src["event"][self.ix].any(1)) if not self.speech else np.array([],int)
    def pick(self,rng,n):
        if self.speech or not len(self.positive):return rng.integers(len(self.ix),size=n)
        return np.r_[rng.choice(self.positive,n//2),rng.integers(len(self.ix),size=n-n//2)]
    def get(self,rows,mode,masked,condition="correct",style_condition="neutral"):
        real=self.ix[rows]; n=len(rows); valid=np.arange(64)[None]<self.length[rows,None]
        wrong=self.wrong[rows]; donor=np.where(wrong>=0,wrong,rows) if condition=="wrong" else rows
        z=np.asarray(self.z[mode,masked][donor],np.float32)
        if condition=="shifted":z=np.concatenate((np.repeat(z[:,:1],8,1),z[:,:-8]),1)
        if condition=="zero_field":z.fill(0)
        size=np.maximum(1,np.rint(self.length[rows]*.25).astype(int)); start=(self.ids[rows]%np.maximum(1,self.length[rows]-size+1).astype(np.uint64)).astype(int)
        blank=(np.arange(64)[None]>=start[:,None])&(np.arange(64)[None]<(start+size)[:,None])
        # Match the target mask used for the input cache. Timed units are evaluated on erased spans for audio.
        use=valid&(blank if mode==0 and masked else True)
        voice=np.zeros((n,40),np.float32); texture=np.zeros((n,16),np.float32)
        if style_condition=="enrolled":
            from training.g7_units import style
            ref=self.ref[rows]; safe=ref!=rows
            if self.speech:
                voice[safe]=style(self.src["a"][self.ix[ref[safe]]],self.length[ref[safe]])/6
            if "v" in self.src:
                texture[safe]=(self.src["v"][self.ix[ref[safe]],0].reshape(-1,4,4,4,4).mean((2,4)).reshape(-1,16)/127.5-1)
        arr={"z":z,"a":(self.src["a"][real].astype(np.float32)/16-7)/3,
             "v":self.src["v"][real].astype(np.float32)/127.5-1 if "v" in self.src else np.zeros((n,64,256),np.float32),
             "unit":np.array(self.units[real],np.int32) if self.speech else np.zeros((n,64,2),np.int32),
             "has_v":np.full(n,"v" in self.src),"speech":np.full(n,self.speech),"use":use,
             "valid":valid,"voice":voice,"texture":texture,"event":self.src["event"][real] if not self.speech else np.zeros((n,64),np.uint8),
             "wrong_available":wrong>=0}
        return arr
    def sync(self,rows):
        wrong=self.wrong[rows];use=wrong>=0; rows=rows[use]; wrong=wrong[use]
        a=np.array(self.z[0,False][rows],np.float32);v=np.array(self.z[1,False][rows],np.float32)
        other=np.array(self.z[1,False][wrong],np.float32)
        # Center eight-tick-interior queries avoid wraparound or silence padding as a class cue.
        mm=np.arange(8,56)[None]<np.minimum(self.length[rows,None]-8,self.length[wrong,None]-8)
        return a[:,8:56],v[:,8:56],np.concatenate([v[:,:48]],axis=1),other[:,8:56],mm


class Readout(nn.Module):
    def __init__(self):
        super().__init__();self.norm=nn.LayerNorm(128);self.temporal=nn.Conv1d(128,128,5)
        self.mix=nn.Sequential(nn.Linear(128,192),nn.GELU(),nn.Linear(192,128))
        self.unit=nn.Linear(128,128);self.boundary=nn.Linear(128,1);self.contact=nn.Linear(128,1)
        self.voice=nn.Linear(40,24);self.texture=nn.Linear(16,16)
        self.audio=nn.Linear(152,200);self.video=nn.Linear(144,512)
        self.audio_match=nn.Linear(128,48);self.video_match=nn.Linear(128,48)
    def field(self,z):
        h=self.norm(z);q=self.temporal(F.pad(h.transpose(1,2),(4,0))).transpose(1,2)
        return h+.35*self.mix(q)
    def forward(self,z,voice,texture):
        h=self.field(z);n,t=h.shape[:2]
        am,av=self.audio(torch.cat([h,self.voice(voice)[:,None].expand(-1,t,-1)],-1)).chunk(2,-1)
        vm,vv=self.video(torch.cat([h,self.texture(texture)[:,None].expand(-1,t,-1)],-1)).chunk(2,-1)
        return self.unit(h).reshape(n,t,2,64),self.boundary(h)[...,0],self.contact(h)[...,0],am,av.clamp(-4,3),vm,vv.clamp(-5,2)
    def match(self,a,v,m):
        aq=F.normalize(self.audio_match(self.field(a)),dim=-1);vq=F.normalize(self.video_match(self.field(v)),dim=-1)
        return ((aq*vq).sum(-1)*m).sum(1)/m.sum(1).clamp_min(1)*15


def gpu(x):return torch.as_tensor(np.asarray(x),device="cuda")
def combine(parts):return {k:gpu(np.concatenate([b[k] for b in parts])) for k in parts[0]}
def weighted(x,mask):return (x*mask).sum()/mask.sum().clamp_min(1)


def loss(model,b,matched,variant):
    logits,bnd,ev,am,av,vm,vv=model(b["z"],b["voice"],b["texture"])
    m=b["use"].bool(); valid=b["valid"].bool();speech=b["speech"][:,None].bool()
    audio=.5*((am-b["a"]).square()*(-av).exp()+av).mean(-1)+.14*(am-b["a"]).abs().mean(-1)
    video=.5*((vm-b["v"]).square()*(-vv).exp()+vv).mean(-1)
    event=F.binary_cross_entropy_with_logits(ev,b["event"].float(),pos_weight=torch.tensor(13.,device="cuda"),reduction="none")
    unit=F.cross_entropy(logits.flatten(0,2),b["unit"].long().flatten(),reduction="none").reshape(*m.shape,2).mean(-1)
    boundary=torch.zeros_like(b["event"]);boundary[:,1:]=(b["unit"][:,1:,0]!=b["unit"][:,:-1,1]).to(boundary.dtype)
    bt=F.binary_cross_entropy_with_logits(bnd,boundary.float(),reduction="none")
    total=weighted(audio,m)+.5*weighted(video,m&b["has_v"][:,None].bool())+.4*weighted(event,valid&~speech)
    if variant=="units-sync":
        a,v,shift,other,mm=matched
        sync=torch.stack([model.match(a,x,mm) for x in (v,shift,other)],1)
        total=total+.85*weighted(unit,m&speech)+.12*weighted(bt,valid&speech)+.35*F.cross_entropy(sync,torch.zeros(len(sync),dtype=torch.long,device="cuda"))
    return total


def ap(score,y):
    ix=np.argsort(-score,kind="stable");s=score[ix];l=y[ix]
    ends=np.r_[s[1:]!=s[:-1],True];cum=np.cumsum(l)[ends];rank=np.arange(1,len(s)+1)[ends]
    return float((np.diff(cum,prepend=0)*cum/rank).sum()/max(1,l.sum()))


def evaluate(model,sets,role,variant):
    scores={}
    with torch.inference_mode():
        for family,d in sets.items():
            rs={}
            for mode in d.card["views"]:
                for cond in ("correct","wrong","shifted","zero_field"):
                    hit=total=0;aud=coverage=vis=0.; ev=[];labels=[];sy=[]
                    for j in range(0,len(d.ix),64):
                        ix=np.arange(j,min(j+64,len(d.ix)))
                        if cond=="wrong":ix=ix[d.wrong[ix]>=0]
                        if not len(ix):continue
                        b=combine([d.get(ix,mode,mode==0,cond)])
                        u,_,on,am,lv,vm,_=model(b["z"],b["voice"],b["texture"]); mask=b["use"].bool()
                        if cond=="shifted":mask[:,:8]=False
                        n=float(mask.sum());total+=n
                        if d.speech:hit+=float(((u.argmax(-1)==b["unit"])*mask[...,None]).sum())/2
                        aud+=float((((am-b["a"])**2).mean(-1)*mask).sum())
                        coverage+=float((((am-b["a"]).abs()<1.645*(.5*lv).exp()).float().mean(-1)*mask).sum())
                        if "v" in d.src:vis+=float((((vm-b["v"])**2).mean(-1)*mask).sum())
                        if not d.speech:
                            ev.append(on[mask].cpu().numpy());labels.append(b["event"][mask].cpu().numpy())
                    if total:
                        rs[f"{mode}-{cond}"]={"ticks":int(total),"unit_top1":hit/total if d.speech and variant=="units-sync" else None,
                            "audio_mse":aud/total,"video_mse":vis/total if "v" in d.src else None,
                            "audio_90pct_coverage":coverage/total,"event_ap":ap(np.concatenate(ev),np.concatenate(labels)) if ev else None}
            if 0 in d.card["views"] and 1 in d.card["views"] and variant=="units-sync":
                winners=[]
                for j in range(0,len(d.ix),64):
                    seq=d.sync(np.arange(j,min(j+64,len(d.ix))))
                    if not len(seq[0]):continue
                    a,v,shift,other,m=map(gpu,seq); logits=torch.stack([model.match(a,x,m) for x in (v,shift,other)],1)
                    valid=m.any(1); tied=logits==logits.max(1,keepdim=True).values
                    winners.extend((tied[:,0].float()/tied.sum(1))[valid].cpu().tolist())
                rs["synchrony_top1"]={"correct_vs_shift_vs_same_group_wrong":float(np.mean(winners)) if winners else None,"eligible":len(winners),"chance":1/3}
            scores[family]=rs
    return scores


def fit(a):
    from training.g4_data import digest
    tag=f"{a.core}-{a.seed}"; out=a.out/tag/a.variant;out.mkdir(exist_ok=True,parents=True)
    if (out/"dev.json").is_file():return
    sets={f:Views(a.root,a.old,tag,f,"train") for f in ("libri","crema","events","oldimpact")}
    torch.set_num_threads(2);torch.manual_seed(a.seed)
    model=Readout().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=.01)
    parent=sets["libri"].card["parent_sha256"];ck=out/"resume.pt";start=0
    if ck.exists():
        x=torch.load(ck,map_location="cuda",weights_only=True)
        if x["parent"]!=parent:raise ValueError("Alignment parent changed")
        model.load_state_dict(x["weights"]);opt.load_state_dict(x["opt"]);start=x["step"]
    for step in range(start,a.steps):
        rng=np.random.default_rng([7781,step]);parts=[];syn=[]
        for family,n in (("libri",32),("crema",32),("events",16),("oldimpact",16)):
            d=sets[family];rows=d.pick(rng,n)
            kinds=np.resize(np.array(d.card["views"]),len(rows));rng.shuffle(kinds)
            half=rng.random(n)<.5
            for mode in d.card["views"]:
                for masked in (False,True):
                    use=(kinds==mode)&((half|(mode==0))==masked)
                    if use.any():parts.append(d.get(rows[use],mode,masked,style_condition="enrolled" if step%2 else "neutral"))
            if 1 in d.card["views"]:syn.append(d.sync(rows[:12]))
        matched=[gpu(np.concatenate(values)) for values in zip(*syn)]
        objective=loss(model,combine(parts),matched,a.variant)
        if not torch.isfinite(objective):raise FloatingPointError(step)
        opt.zero_grad();objective.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if (step+1)%256==0 or step+1==a.steps:
            torch.save({"weights":model.state_dict(),"opt":opt.state_dict(),"step":step+1,"parent":parent},out/"resume.part")
            (out/"resume.part").replace(ck);print(tag,a.variant,step+1,float(objective.detach()),flush=True)
    torch.save({"weights":model.state_dict(),"parent":parent,"variant":a.variant},out/"weights.part")
    (out/"weights.part").replace(out/"weights.pt");model.eval()
    dev={f:Views(a.root,a.old,tag,f,"dev") for f in sets}
    save(out/"dev.json",{"tag":tag,"variant":a.variant,"updates":a.steps,"weights_sha256":digest(out/"weights.pt"),"resume_sha256":digest(ck),
         "core_sha256":parent,"codebook_sha256":digest(a.root/"units/codebook.npz"),"trainable_parameters":sum(p.numel() for p in model.parameters()),
         "training_examples":{k:len(d.ix) for k,d in sets.items()},"scores":evaluate(model,dev,"dev",a.variant)})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--old",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--core",required=True,choices=("hybrid","gpt2"));p.add_argument("--seed",type=int,required=True);p.add_argument("--variant",choices=("reconstruct","units-sync"),required=True)
    p.add_argument("--steps",type=int,default=2048);fit(p.parse_args())
