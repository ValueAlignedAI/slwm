"""Frozen 1/4/8-shot transfer, modality swaps and speaker-conditioned retrieval; no backbone updates."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from models.g4_shared import centroid,encode
from training.g4_eval import check_lock,weights
from training.g4_train import backend,read,sha_file,write


@lru_cache(maxsize=16)
def enc_fn(spec):
    @jax.jit
    def apply(p,stats,batch,kind):
        source=jnp.full((len(batch["ok"]),),kind,jnp.int32)
        z,_=encode(p,spec,batch,source,stats)
        valid=batch["ok"].reshape(len(z),4,16,1).astype(jnp.float32)
        parts=(z.astype(jnp.float32).reshape(len(z),4,16,-1)*valid).sum(2)/jnp.maximum(valid.sum(2),1)
        aligned=(centroid(z,batch)@p["align"]).astype(jnp.float32)
        aligned=aligned*jax.lax.rsqrt((aligned*aligned).sum(1,keepdims=True)+1e-8)
        return parts.reshape(len(z),-1),aligned
    return apply


def representations(spec,p,stats,mode,values,batch_size):
    f=enc_fn(spec); arrays=[]; aligned=[]
    for start in range(0,len(values["ok"]),batch_size):
        n=min(batch_size,len(values["ok"])-start)
        def fill(v):
            result=np.zeros((batch_size,*v.shape[1:]),dtype=v.dtype)
            result[:n]=v[start:start+n]; return result
        batch={k:fill(v) for k,v in values.items()}
        z,a=f(p,stats,batch,np.int32(mode))
        arrays.append(np.array(z[:n])); aligned.append(np.array(a[:n]))
    return np.concatenate(arrays),np.concatenate(aligned)


def ridge(train,labels,val,penalty=.1):
    mean=train.mean(0)
    scale=np.maximum(train.std(0),.05)
    z=(train-mean)/scale; x=(val-mean)/scale
    targets=np.eye(int(labels.max())+1)[labels]
    center=targets.mean(0); k=z@z.T/z.shape[-1]
    fit=np.linalg.solve(k+np.eye(len(z))*penalty,targets-center)
    return ((x@z.T/z.shape[-1])@fit+center).argmax(1)


def balanced(y,guess):
    return float(np.mean([(y[c==y]==guess[c==y]).mean() for c in np.unique(y)]))


def support(labels,shots):
    return np.concatenate([np.flatnonzero(labels==c)[:shots] for c in np.unique(labels)])


def probe_batch(data,kind):
    n=len(data)
    result={"a":np.zeros((n,64,100),np.uint8),"v":np.zeros((n,64,256),np.uint8),
            "n":np.zeros((n,64,17),np.float16),"x":np.zeros((n,64),np.uint16),"ok":np.ones((n,64),np.uint8),
            "family":np.full(n,7,np.uint8)}
    result[("a","v")[kind]]=data
    return result


def probe(root,name,role,spec,p,stats,batch_size):
    reps={}; labels={}; raw={}
    for part in ("fit",role):
        with np.load(root/f"{name}-{part}.npz") as d:
            values=d["data"]; kind=int(d["kind"]); labels[part]=d["labels"]
        reps[part],_=representations(spec,p,stats,kind,probe_batch(values,kind),batch_size)
        raw[part]=values.reshape(len(values),4,16,-1).mean(2).astype(np.float32).reshape(len(values),-1)/32
    scores={}; predictions={}
    for shots in (1,4,8):
        ix=support(labels["fit"],shots)
        for title,x in (("shared",reps),("raw",raw)):
            guess=ridge(x["fit"][ix],labels["fit"][ix],x[role])
            scores[f"{title}-{shots}"]=balanced(labels[role],guess)
            predictions[f"{title}-{shots}"]=guess
    return {"samples":len(labels[role]),"classes":len(np.unique(labels[role])),"balanced_accuracy":scores},predictions


def grid_views(root,role,spec,p,stats,batch_size):
    reps={}; emb={}; labels={}; group=None; cap=None
    for split in ("train",role):
        d={k:np.load(root/split/f"{k}.npy",mmap_mode="r") for k in ("a","v","n","x","ok","family","group","uid","cap")}
        ix=np.flatnonzero(d["family"]==0)
        ix=ix[np.argsort(d["uid"][ix])]
        words=[bytes(int(x)-3 for x in d["x"][j] if x>=3).decode(errors="replace").split()[0].lower() for j in ix]
        classes=("bin","lay","place","set")
        label=np.array([classes.index(w) if w in classes else -1 for w in words])
        if np.any(label<0): raise ValueError("Unexpected GRID command vocabulary")
        if split=="train":
            sub=support(label,32); ix,label=ix[sub],label[sub]
        else: group=np.asarray(d["group"][ix]); cap=np.asarray(d["cap"][ix])
        labels[split]=label
        values={k:np.array(d[k][ix]) for k in ("a","v","n","x","ok","family")}
        for kind,key in ((0,"audio"),(1,"visual"),(3,"text")):
            z,projected=representations(spec,p,stats,kind,values,batch_size)
            reps[split,key]=z
            if split==role: emb[key]=projected
    scores={}; retrieval={}
    for src in emb:
        for dst in emb:
            fit=support(labels["train"],8)
            gs=ridge(reps["train",src][fit],labels["train"][fit],reps[role,dst])
            scores[f"{src}-to-{dst}"]=balanced(labels[role],gs)
            if src==dst: continue
            parts=[]
            for grp in np.unique(group):
                idx=np.flatnonzero(group==grp)
                if len(idx)<2: continue
                matches=(emb[src][idx]@emb[dst][idx].T).argmax(1)
                parts.append([len(idx),np.mean(matches==np.arange(len(idx))),np.mean(cap[idx[matches]]==cap[idx]),1/len(idx)])
            retrieval[f"{src}-to-{dst}"]={"groups":len(parts),"mean_pool_size":float(np.mean(parts,0)[0]),
                "exact_top1":float(np.mean(parts,0)[1]),"caption_top1":float(np.mean(parts,0)[2]),"chance_exact":float(np.mean(parts,0)[3])}
    return {"unseen_people":len(np.unique(group)),"support_per_command":8,"command_balanced_accuracy":scores,"within_person_retrieval":retrieval}


def transfer(run,data,probes,role,batch_size,device,lock=None):
    backend(device)
    cfg,complete,spec,p,stats=weights(run)
    if role=="test":
        check_lock(lock,run,complete)
        if read(lock)["transfer_sha256"]!=sha_file(Path(__file__)): raise ValueError("Transfer rules changed")
    results={}; records={}
    start=time.monotonic()
    for name in ("cifar10","esc50"):
        results[name],pred=probe(probes,name,role,spec,p,stats,batch_size)
        records.update({name+"-"+k:v for k,v in pred.items()})
    results["grid"]=grid_views(data,role,spec,p,stats,batch_size)
    write(run/f"transfer-{role}.json",{"run":cfg["name"],"role":role,"weight_sha256":complete["weight_file_sha256"],
        "elapsed_s":time.monotonic()-start,"trained_backbone_parameters":0,"tasks":results})
    np.savez(run/f"transfer-{role}.npz",**records)


if __name__=="__main__":
    q=argparse.ArgumentParser(description=__doc__)
    for x in ("run","data","probes"): q.add_argument("--"+x,type=Path,required=True)
    q.add_argument("--role",default="dev",choices=["dev","test"]); q.add_argument("--batch",type=int,default=128)
    q.add_argument("--backend",default="tpu"); q.add_argument("--lock",type=Path)
    a=q.parse_args(); transfer(a.run,a.data,a.probes,a.role,a.batch,a.backend,a.lock)
