"""Develop evidence shrinkage and independent utterance transfer before a fresh-family confirmation."""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import jax
import jax.numpy as jnp
import numpy as np
from scripts.train.g12_fit import predict,Spec,read_params
from scripts.train.g9_storage import disks,sha,write

ROOT=Path("/mnt/slwm-data/SLWM/g13");OLD=ROOT.parent/"g12";OLDER=ROOT.parent/"g11/data"
CORES=("gpt2","gated","oscillator","associative");SEEDS=(1201,1202,1203)
STOP=set("a the it and in you can to on of is and then just so now that this with your there okay yeah".split())
ALIASES={"remove":"withdraw","unplug":"withdraw","takeout":"withdraw","put":"place","set":"place","take":"grab","pick":"grab","grasp":"grab",
         "tighten":"screw","loosen":"unscrew","twist":"rotate","shut":"close","plug":"insert","pushin":"insert","click":"press","raise":"lift"}

def tokens(x):
    words=re.findall(r"[a-z]+",x.lower())
    extra=["pair_"+u+"_"+v for u,v in zip(words,words[1:])]
    return set([ALIASES.get(w,w) for w in words if w not in STOP]+extra)


def refs(rows,dir_new):
    memory={};new=ROOT/"data/holo"
    result=[]
    for r in rows:
        n=r["recording"]
        if n not in memory:
            path=(new if dir_new else OLDER)/"speech-small"/(n+".json")
            if not path.exists():path=OLD/"data/speech-small"/(n+".json")
            memory[n]=json.loads(path.read_bytes())["utterances"]
        db=memory[n];result.append((db[str(r["cue"])]["hypothesis"],db[str(r["wrong"])]["hypothesis"]))
    return result


def read_role(fold,role):
    root=ROOT/"data/holo/banks" if role=="test" else OLD/"data/banks"
    file=root/(fold+"-"+role+".npz");meta=json.loads(file.with_suffix(".json").read_bytes())
    with np.load(file) as b:row={k:b[k] for k in b.files}
    use=np.flatnonzero(row["cued"]);row={k:v[use] for k,v in row.items()};rows=[meta["rows"][i] for i in use]
    return meta["vocabulary"],row,rows


def macro(values,groups):
    return float(np.mean([np.asarray(values)[groups==g].mean() for g in np.unique(groups)]))

def nll(p,y):
    a=np.take_along_axis(p,np.maximum(y,0)[:,:,None],2)[:,:,0]
    return -np.log(np.clip(np.where(y>=0,a,1e-7),1e-7,1)).mean(1)


def language_map(material,dim):
    base=np.full((3,dim),.03);total=Counter();counts={}
    rec=Counter(n for n,s,y in material)
    for n,s,y in material:
        weight=1/rec[n]
        for h,c in enumerate(y):
            if c>=0:base[h,c]+=weight
        for token in tokens(s):
            a=counts.setdefault(token,np.zeros((3,dim)))
            for h,c in enumerate(y):
                if c>=0:a[h,c]+=weight
            total[token]+=weight
    prior=base/base.sum(-1,keepdims=True)
    stats={t:(np.log((a+1.5*prior)/(a.sum(-1,keepdims=True)+1.5))-np.log(prior),min(1.,total[t]/2)) for t,a in counts.items() if total[t]>=.12}
    return prior,stats


def lexical(texts,reference,stats):
    out=[]
    for text in texts:
        active=[stats[t] for t in sorted(tokens(text)) if t in stats];res=sum((v*w for v,w in active),start=np.zeros_like(reference))
        res=res/max(1,sum(w for _,w in active)**.65)
        p=reference*np.exp(np.clip(res,-6,6));out.append(p/p.sum(-1,keepdims=True))
    return np.stack(out)


def combine(parent,prior,lex,w,b):
    base=(1-w)*prior[None]+w*parent
    p=base*(np.maximum(lex,1e-8)/prior[None])**b
    return p/p.sum(-1,keepdims=True)

def main(action):
    dest=ROOT/"work/holo";dest.mkdir(exist_ok=True,parents=True)
    extra=json.loads((ROOT/"data/holo/extra-language.json").read_bytes())
    roles=json.loads((OLD/"data/groups.json").read_bytes())
    families={"A":{"coffee","computer"},"B":{"printer","camera"}}
    recipes={};summary={};outputs={}
    for fold in ("A","B"):
        vocab,tr,meta=read_role(fold,"train");idx={v:i for i,v in enumerate(vocab)}
        tr_words=refs(meta,False)
        original=[(r["recording"],s[0],v) for r,s,v in zip(meta,tr_words,tr["y"])]
        addition=[(n,v["text"],[idx.get(a,-1) for a in v["verbs"]]) for n,rec in extra.items() if rec["family"] not in families[fold] for v in rec["examples"]]
        prior,baselex=language_map(original,len(vocab));_,added=language_map(original+addition,len(vocab))
        vocab,dev,dm=read_role(fold,"dev");ds=refs(dm,False);dy=dev["y"]
        dlex={t:lexical([s[0] for s in ds],prior,x) for t,x in (("small",baselex),("expanded",added))}
        recipes[fold]={};summary[fold]=dict(extra_cases=len(addition),extra_recordings=len(set(x[0] for x in addition)),train_cases=len(original))
        if action=="test":
            locks=json.loads((dest/"locked.json").read_bytes());_,test,tm=read_role(fold,"test");ss=refs(tm,True);yt=test["y"]
            tl={variant:{mode:lexical([s[0] if mode=="correct" else s[1] if mode=="wrong" else "" for s in ss],prior,stat)
               for mode in ("correct","wrong","none")} for variant,stat in (("small",baselex),("expanded",added))}
            prefix=fold+"_";outputs[prefix+"y"]=yt;outputs[prefix+"groups"]=np.asarray([r["recording"] for r in tm]);outputs[prefix+"families"]=np.asarray([r["family"] for r in tm])
            outputs[prefix+"prior"]=np.broadcast_to(prior,(len(yt),*prior.shape)).astype("f4");summary[fold]["fresh_cases"]=len(yt)
        for seed in SEEDS:
            for core in CORES:
                tag=f"{fold}-{core}-{seed}";candidates=[OLD/"early-eval"/f"{fold}-{core}-{arm}-{seed}" for arm in ("full","dropout")]
                def score(folder):
                    d=json.loads((folder/"receipt.json").read_bytes())
                    return next(m["selection"] for m in d["curve"] if m["step"]==d["selected"])
                parent=min(candidates,key=score);pr=json.loads((parent/"receipt.json").read_bytes())
                with np.load(parent/"dev.npz") as z:dprob=z["correct_verb"]
                selected={}
                for arm in ("shrink","small","expanded"):
                    choices=[]
                    for a in (0.,.05,.15,.3,.5,.75,1.):
                        for b in ((0.,) if arm=="shrink" else (0.,.15,.35,.6,1.)):
                            p=combine(dprob,prior,dlex.get(arm,dlex["small"]),a,b)
                            choices.append((macro(nll(p,dy),dev["group"]),a,b))
                    loss,a,b=min(choices);selected[arm]=dict(parent_weight=a,language_power=b,development_nll=loss)
                recipes[fold][tag]=dict(parent=parent.name,weights_sha=pr["weights_sha256"],selected=selected)
                if action!="test":continue
                if recipes[fold][tag]!=locks["recipes"][fold][tag]:raise ValueError("Natural-speech dev choice drift")
                folder=disks()[0].parent.parent/"g12/early"/parent.name/f"step-{pr['selected']:05d}";file=folder/"weights.npz"
                if sha(file)!=pr["weights_sha256"]:raise ValueError("Unregistered natural speech parent")
                pp=read_params(file);spec=Spec(**pr["config"]["spec"])
                for mode in ("correct","wrong","none"):
                    p=predict(spec,core,pp,test,mode)[0]
                    outputs[tag+"_parent_"+mode]=p
                    for arm,cfg in selected.items():
                        lex=tl.get(arm,tl["small"])[mode]
                        outputs[tag+"_"+arm+"_"+mode]=combine(p,prior,lex,cfg["parent_weight"],cfg["language_power"]).astype("f4")
        print("Natural speech",action,fold,flush=True)
    if action=="develop":write(dest/"locked.json",dict(recipes=recipes,data=summary))
    else:
        np.savez_compressed(dest/"predictions.npz",**outputs);write(dest/"complete.json",summary)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=("develop","test"));a=p.parse_args();main(a.phase)
