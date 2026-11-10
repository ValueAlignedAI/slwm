"""Four-way cross-modal matching with speaker and exact clip length held constant."""
import argparse
from pathlib import Path
import numpy as np
import torch
from training.g4_data import digest
from training.g7_alignment import Views,Readout,meta,gpu,save


def summaries(field,lengths):
    # Fixed phase bins remove the amount of padding as an answer cue.
    features=[]
    for row,n in zip(field,lengths):
        block=np.array_split(row[:n],8);features.append(np.concatenate([b.mean(0) for b in block]))
    x=np.stack(features).astype(np.float32)
    return x/np.maximum(1e-8,np.linalg.norm(x,axis=-1,keepdims=True))


def choices(d,other_recording):
    groups=d.groups;codes=d.src["phrase"][d.ix];ids=d.ids;rows=[];options=[]
    for i in range(len(ids)):
        matched=(groups==groups[i])&(d.length==d.length[i])
        wrong=np.flatnonzero(matched&(codes!=codes[i]))
        if len(wrong)<3:continue
        correct=i
        if other_recording:
            same=np.flatnonzero(matched&(codes==codes[i])&(ids!=ids[i]))
            if not len(same):continue
            correct=same[int(ids[i])%len(same)]
        rng=np.random.default_rng([int(ids[i]),771131]);foils=rng.choice(wrong,3,replace=False)
        rows.append(i);options.append([correct,*foils.tolist()])
    return np.array(rows,int),np.asarray(options,int).reshape(-1,4)


def metric(a,b,rows,opts,groups):
    if not len(rows):return {"eligible":0}
    scores=np.einsum("rd,rkd->rk",a[rows],b[opts]);ties=scores==scores.max(1,keepdims=True)
    win=ties[:,0]/ties.sum(1)
    unique,ix=np.unique(groups[rows],return_inverse=True);n=np.bincount(ix)
    per=np.bincount(ix,weights=win);sample=np.random.default_rng(7711).integers(len(n),size=(2000,len(n)))
    return {"accuracy":float(win.mean()),"eligible":len(rows),"people":len(n),"speaker_ci95":np.quantile(per[sample].sum(1)/n[sample].sum(1),[.025,.975]).tolist()}


def run(a):
    lock=meta(a.ck/"alignment/test-lock.json");out=a.ck/"retrieval";out.mkdir(exist_ok=True)
    for key,v in lock["models"].items():
        if not key.endswith("/units-sync"):continue
        tag=key.split("/")[0]
        path=out/(tag+".json")
        if path.exists():continue
        weights=a.ck/"alignment"/key/"weights.pt"
        if digest(weights)!=v["sha256"]:raise ValueError("Model changed")
        head=Readout().cuda().eval();head.load_state_dict(torch.load(weights,map_location="cuda",weights_only=True)["weights"])
        result={}
        with torch.inference_mode():
            for family in ("crema","libri"):
                d=Views(a.root,a.old,tag,family,"test");fields={};posteriors={}
                for view in d.card["views"]:
                    z=d.z[view,False];states=[];codes=[]
                    for i in range(0,len(z),32):
                        obs=gpu(np.array(z[i:i+32],np.float32))
                        u=head.unit(head.field(obs)).reshape(len(obs),64,2,64).softmax(-1).mean(2)
                        codes.extend(u.cpu().numpy());states.extend(obs.cpu().numpy())
                    fields[view]=summaries(states,d.length);posteriors[view]=summaries(codes,d.length)
                result[family]={}
                for alias,another in (("same_clip",False),("distinct_recording_same_words",True)):
                    rows,opts=choices(d,another)
                    for src in d.card["views"]:
                        for target in d.card["views"]:
                            if src==target:continue
                            result[family][f"{alias}-{src}-to-{target}"]={"frozen_shared_field":metric(fields[src],fields[target],rows,opts,d.groups),
                                "trained_unit_distributions":metric(posteriors[src],posteriors[target],rows,opts,d.groups)}
        save(path,{"scores":result,"chance":.25,"head_sha256":v["sha256"],
            "controls":"All four candidates have the same actor and valid duration. Three have different words. Ties receive fractional credit."})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("root","old","ck"):p.add_argument("--"+k,type=Path,required=True)
    run(p.parse_args())
