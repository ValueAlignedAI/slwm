"""Build role-separated causal banks. Effect availability never enters predictor features."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
import requests
from data.g10_language import text_features
from g11_events import G10,FOLDS,rank

def write(path,obj):
    tmp=path.with_suffix(".new");tmp.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+"\n");tmp.replace(path)

def digest(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def file_for(root,name):
    old=G10/"shards"/(name+".npz")
    return old if old.is_file() else root/"fresh/shards"/(name+".npz")

def ontology(root):
    path=root/"fine_grained_actions_map.txt"
    if not path.exists():
        r=requests.get("https://raw.githubusercontent.com/holoAssist/holoassist.github.io/main/label_files/fine_grained_actions_map.txt",timeout=40);r.raise_for_status();path.write_bytes(r.content)
    known=set()
    for line in path.read_text().splitlines():
        parts=line.strip().split(maxsplit=1)
        if len(parts)==2 and "-" in parts[1]:known.add(parts[1].split("-",1)[1].strip().casefold())
    return sorted(known)

def build_base(root):
    cat=json.loads((root/"annotations.json").read_text());groups=json.loads((root/"groups.json").read_text());pub=ontology(root)
    dest=root/"banks";dest.mkdir(exist_ok=True)
    if not (root/"fresh/complete.json").is_file() or not (root/"speech/complete").is_file():raise ValueError("Causal AV/speech banks are not ready")
    for fold,roles in groups["folds"].items():
        train=[e for n in roles["train"] for e in cat[n]["events"]+cat[n]["background"]]
        labels={k:sorted({e[k] for e in train}) for k in ("verb","noun")};labels["open_noun"]=pub;labels["fold"]=fold
        write(dest/(fold+"-vocabulary.json"),labels)
        vocab={k:{v:i for i,v in enumerate(labels[k])} for k in ("verb","noun","open_noun")}
        for role,names in roles.items():
            path=dest/f"{fold}-{role}.npz";meta=dest/f"{fold}-{role}.json"
            if path.exists() and meta.exists():continue
            arrays={k:[] for k in ("a","v","n","ac","wc","aw","ww","ah","wh","verb","noun","open_noun","prev_verb","prev_noun","cued","group","conflict")}
            rows=[]
            for gi,name in enumerate(names):
                r=cat[name]
                with np.load(file_for(root,name),allow_pickle=False) as sh:
                    audio=sh["a"];vis=sh["v"];phys=sh["n"]
                speechmask=np.zeros(len(audio),bool)
                for low,high in r["conversation"]:
                    speechmask[max(0,int(np.floor(low*10))):min(len(audio),int(np.ceil(high*10)))]=True
                if r["events"]:
                    with np.load(root/"speech"/(name+".npz")) as file:
                        si={int(k):i for i,k in enumerate(file["ids"])};h,asr,human=file["acoustic"],file["asr"],file["oracle"]
                for ev in r["events"]+(r["background"] if role=="train" else []):
                    cut=ev["cut"];ix=cut-31+np.arange(32)
                    if cut>=len(audio) or ix[0]<0:continue
                    cued=ev["cue"]>=0
                    aa=audio[ix].copy();aa[speechmask[ix]]=0
                    arrays["a"].append(aa);arrays["v"].append(vis[ix]);arrays["n"].append(phys[ix].astype(np.float16))
                    for nm,key,dat in (("ac","cue",h if cued else None),("wc","wrong",h if cued else None),
                             ("aw","cue",asr if cued else None),("ww","wrong",asr if cued else None),("ah","cue",human if cued else None),("wh","wrong",human if cued else None)):
                        blank=np.zeros((16,128),np.float16) if nm in ("ac","wc") else np.zeros(128,np.int8)
                        arrays[nm].append(dat[si[ev[key]]] if cued else blank)
                    for k,value in (("verb",vocab["verb"].get(ev["verb"],-1)),("noun",vocab["noun"].get(ev["noun"],-1)),("open_noun",vocab["open_noun"].get(ev["noun"],-1)),
                             ("prev_verb",vocab["verb"].get(ev.get("last_verb"),-1)),("prev_noun",vocab["noun"].get(ev.get("last_noun"),-1)),
                             ("cued",cued),("group",gi),("conflict",ev.get("conflict",False))):arrays[k].append(value)
                    rows.append(dict(recording=name,action_id=ev["id"],cut_tick=cut,verb=ev["verb"],noun=ev["noun"],family=r["family"],task=r["task"],
                                     cue=ev["cue"],wrong=ev["wrong"],physical=ev["physical"],isolated=ev.get("isolated",False)))
            block={k:np.asarray(v) for k,v in arrays.items()}
            for k in ("verb","noun","open_noun","prev_verb","prev_noun","group"):block[k]=block[k].astype(np.int32)
            sem=np.stack([text_features(n.replace("_"," ")) for n in labels["open_noun"]]).astype(np.float32)/127
            tmp=path.with_suffix(".tmp.npz");np.savez_compressed(tmp,**block,names=np.array(names),semantic=sem);tmp.replace(path)
            write(meta,dict(fold=fold,role=role,sha256=digest(path),records=names,rows=rows,examples=len(rows),cued=int(block["cued"].sum()),source_hash=digest(root/"groups.json")))
            print(fold,role,len(rows),int(block["cued"].sum()),path.stat().st_size,flush=True)
    (dest/"base-ready").touch()

def build_effect(root):
    dest=root/"banks";physical=root/"geometry";strong=root/"speech-small"
    for p in (dest/"base-ready",physical/"complete",strong/"complete"):
        if not p.exists():raise ValueError("Effect or secondary speech is not ready: "+str(p))
    docs={};features={};ss={}
    for meta in sorted(dest.glob("[AB]-*.json")):
        if "vocabulary" in meta.name:continue
        d=json.loads(meta.read_text());rows=d["rows"];n=len(rows);geo=np.zeros((n,201),np.float32);y=np.zeros((n,3),np.float32)
        valid=np.zeros(n,bool);strict=np.zeros(n,bool);motion=np.zeros(n,np.float32);neutral=np.zeros(n,np.float32);background=np.full(n,np.nan,np.float32)
        small=np.zeros((n,128),np.int8);wrong=np.zeros_like(small)
        for i,r in enumerate(rows):
            name=r["recording"];key=r["action_id"]
            if name not in docs:
                info=json.loads((physical/(name+".json")).read_text());docs[name]={q["id"]:q for q in info["valid"]}
                with np.load(physical/(name+".npz")) as file:features[name]={int(k):(f,float(t)) for k,f,t in zip(file["prefix_ids"],file["prefix"],file["prefix_times"])}
            if key in features[name]:
                f,t=features[name][key]
                if t>(r["cut_tick"]+1)/10+.0001:raise ValueError("Future geometry entered the prefix")
                geo[i,:200]=f;geo[i,200]=1
            if key in docs[name]:
                v=docs[name][key];valid[i]=True;strict[i]=bool(v["isolated"] and v["anchor_source"]=="hand_adjacent")
                y[i]=[v["nearer"],v["farther"],v["magnitude_mm"]/100]
                neutral[i]=v["neutral"];background[i]=np.nan if v["background_changed"] is None else v["background_changed"]
                motion[i]=v["pre_motion"] if v["pre_motion"] is not None else 0.
            if r["cue"]>=0:
                if name not in ss:
                    with np.load(strong/(name+".npz")) as file:ss[name]={int(k):v for k,v in zip(file["ids"],file["asr"])}
                small[i]=ss[name][r["cue"]];wrong[i]=ss[name][r["wrong"]]
        target=dest/(meta.stem+"-effects.npz");tmp=target.with_suffix(".new.npz")
        np.savez_compressed(tmp,geometry=geo,effect=y,valid=valid,strict=strict,pre_motion=motion,neutral=neutral,background=background,small_words=small,wrong_small=wrong)
        tmp.replace(target);write(dest/(meta.stem+"-effects-receipt.json"),dict(sha256=digest(target),base_sha256=d["sha256"],examples=n,valid=int(valid.sum()),strict=int(strict.sum())))
        print(meta.stem,"measurable",int(valid.sum()),"strict",int(strict.sum()),flush=True)
    (dest/"effects-ready").touch()

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("stage",choices=("base","effects"));a=p.parse_args()
    build_base(a.root) if a.stage=="base" else build_effect(a.root)
