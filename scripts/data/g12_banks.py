"""Pack strictly causal sensor features separately from ordered future targets and blinded state labels."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from data.g10_language import text_features
from g12_events import ROOT, G11, write
from g11_events import KEYWORDS
from g12_media import source


def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def word_slots(sentence):
    tokens=re.findall(r"[a-z0-9']+",sentence.casefold())
    slots=np.zeros((4,128+len(KEYWORDS)),np.float16)
    if not tokens:return slots
    # Word order remains visible. The last semantic channels are fixed public action-word cues at the edge.
    chunks=np.array_split(np.arange(len(tokens)),min(4,len(tokens)))
    for i,ix in enumerate(chunks):
        s=" ".join(tokens[j] for j in ix)
        slots[i,:128]=text_features(s).astype(np.float32)/127
        slots[i,128:]=[float(any(re.search(r"\b"+re.escape(x)+r"\b",s) for x in aliases)) for aliases in KEYWORDS.values()]
    return slots


def packed_record(root,name,record,catalog,vocab):
    with np.load(source(root,name,"shards",".npz"),allow_pickle=False) as sh:
        audio,imu,rgb=sh["a"],sh["n"],sh["v"]
    with np.load(root/"vision"/(name+".npz"),allow_pickle=False) as v:
        times,visual=v["time"],v["x"]
    mask=np.zeros(len(audio),bool)
    for l,r in catalog["conversation"]:
        mask[max(0,int(np.floor(l*10))):min(len(mask),int(np.ceil(r*10)))]=True
    audio=audio.copy();audio[mask]=0
    speech={}
    if any(r["cue"]>=0 for r in record["rows"]):
        path=G11/"speech-small"/(name+".npz")
        if not path.is_file():path=root/"speech-small"/(name+".npz")
        trans=json.loads(path.with_suffix(".json").read_bytes())["utterances"]
        reference={v["id"]:v["text"] for v in catalog["speech"]}
        with np.load(path) as file:
            for ident,x in zip(file["ids"],file["acoustic"]):
                k=int(ident)
                speech[k]=(x.reshape(4,4,128).mean(1),word_slots(trans[str(k)]["hypothesis"]),word_slots(reference[k]))
    blank=(np.zeros((4,128),np.float16),np.zeros((4,128+len(KEYWORDS)),np.float16),np.zeros((4,128+len(KEYWORDS)),np.float16))
    rows=[]
    for r in record["rows"]:
        tick=r["tick"];cut=(tick+1)/10
        if tick>=min(len(audio),len(imu),len(rgb)):continue
        grid=cut-np.arange(7,-1,-1)
        # Strictly use timestamps ending by each historical read, including the source A/V clock tolerance.
        vi=np.searchsorted(times,grid-.11,side="right")-1
        xvis=visual[np.maximum(vi,0)].astype(np.float32)
        xvis[vi<0]=0
        xvis=xvis/(np.sqrt(np.mean(xvis*xvis,-1,keepdims=True))+.1)
        ix=tick-10*np.arange(7,-1,-1)[:,None]+np.arange(-3,1)
        ok=ix>=0;safe=np.maximum(ix,0)
        a=(audio[safe].astype(np.float32)*ok[:,:,None]).mean(1)/255
        n=(imu[safe,:7].astype(np.float32)*ok[:,:,None]).mean(1)
        raw=(rgb[safe[:,-1]].reshape(8,4,4,4,4,3).mean((2,4)).reshape(8,48)/255-.5)*ok[:,-1,None]
        scene=np.concatenate([xvis,a,n,raw,(vi>=0)[:,None],ok[:,-1,None]],axis=1).astype(np.float16)
        ac,words,human=speech.get(r["cue"],blank);wc,ww,wh=speech.get(r["wrong"],blank)
        y=[vocab.get(v,-1) for v in r["verbs"]]
        rows.append((dict(scene=scene,ac=ac,wc=wc,words=words,wrong=ww,human=human,wrong_human=wh,
                          y=y,prev=vocab.get(r["previous"],-1),cued=r["cue"]>=0,conflict=r["conflict"]),
                     dict(recording=name,id=r["id"],family=record["family"],cut_tick=tick,
                          cue=r["cue"],wrong=r["wrong"],verbs=r["verbs"],previous=r["previous"],
                          target_onsets=r["starts"],target_ends=r["ends"])))
    return rows


def build(root):
    groups=json.loads((root/"groups.json").read_bytes());records=json.loads((root/"records.json").read_bytes())
    catalog=json.loads((G11/"annotations.json").read_bytes())
    catalog.update(json.loads((root/"annotations.json").read_bytes()))
    for fold,roles in groups.items():
        vocab=sorted({v for n in roles["train"] for r in records[n]["rows"] for v in r["verbs"]})
        ids={v:i for i,v in enumerate(vocab)}
        for role,names in roles.items():
            directory=root/"banks";base=directory/f"{fold}-{role}.npz";meta=base.with_suffix(".json")
            if base.is_file() and meta.is_file():continue
            data=[];index=[]
            for n in names:
                for block,desc in packed_record(root,n,records[n],catalog[n],ids):
                    if role!="train" and not block["cued"]:continue
                    block["group"]=names.index(n)
                    data.append(block);index.append(desc)
            arr={k:np.stack([row[k] for row in data]) for k in data[0]}
            for k in ("y","prev","group"):arr[k]=arr[k].astype(np.int32)
            tmp=base.with_suffix(".new.npz");np.savez_compressed(tmp,**arr);tmp.replace(base)
            write(meta,dict(names=names,vocabulary=vocab,rows=index,sha256=sha(base),cued=int(arr["cued"].sum()),
                            dimensions={k:list(v.shape[1:]) for k,v in arr.items()}))
            print(fold,role,len(index),int(arr["cued"].sum()),base.stat().st_size,flush=True)
    (root/"banks/ready").touch()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=ROOT)
    build(p.parse_args().root)
