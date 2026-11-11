"""Compare learned and semantically anchored shared fields on whole-inventory future component retrieval."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import re
import time
import numpy as np
import jax
import jax.numpy as jnp
from models.g13_component import initial,forward,loss
from scripts.train.g13_optim import ROOT,roots,complete,save,load,update
from scripts.train.g9_storage import write,sha
from training.g13_physics import group_mean
from evals.g13_stats import Groups

CORES=("gpt2","gated","oscillator","associative");SEEDS=(1301,1302,1303);STEPS=(32,96,256,512)
ROOT_DATA=ROOT/"data/components/component-banks";WORK=ROOT/"work/components"
FEATURES=("scene","ac","wc","text","alt","oracle","alt_oracle")
def read(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}
def normalize(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)
def softmax(z):
    p=np.exp(z-z.max(-1,keepdims=True));return p/p.sum(-1,keepdims=True)
def task(prob,y):
    order=np.argsort(-prob,axis=1,kind="stable")[:,:5]
    correct=(order[:,0,None]==y).any(1)
    mass=np.zeros(len(y))
    for h in range(3):
        ok=y[:,h]>=0
        if h:ok&=np.all(y[:,:h]!=y[:,h,None],axis=1)
        mass[ok]+=prob[np.flatnonzero(ok),y[ok,h]]
    return dict(hit=correct.astype(float),first=(order[:,0]==y[:,0]).astype(float),
                top5=(order[:,:,None]==y[:,None,:]).any((1,2)).astype(float),nll=-np.log(np.maximum(mass,1e-7)))


def count_prior(rows,n):
    y=rows["y"];cu=rows["cued"];g=rows["group"][cu];_,inv,ct=np.unique(g,return_inverse=True,return_counts=True)
    counts=np.full(n,.15)
    for h in range(3):
        v=y[cu,h];ok=v>=0
        np.add.at(counts,v[ok],(1./ct[inv])[ok])
    return (counts/counts.sum()).astype("f4")

def draws(rows,seed):
    rng=np.random.default_rng(seed)
    def part(mask,n):
        j=np.flatnonzero(mask);_,inv,c=np.unique(rows["group"][j],return_inverse=True,return_counts=True);w=1./c[inv]
        return rng.choice(j,(512,n),p=w/w.sum())
    return np.concatenate((part(rows["cued"],96),part(~rows["cued"],32)),1).astype("i4")

@lru_cache(maxsize=12)
def train_block(core,arm):
    @jax.jit
    def run(p,m,v,bank,idx,step,nouns,basis,prior):
        def body(state,data):
            pp,mm,vv=state;ix,t=data;r={k:x[ix] for k,x in bank.items()}
            l,g=jax.value_and_grad(loss)(pp,core,arm,r,nouns,basis,prior)
            return update(pp,mm,vv,g,t,512,.00065),l
        return jax.lax.scan(body,(p,m,v),(idx,step+1+jnp.arange(len(idx))))
    return run

@lru_cache(maxsize=64)
def inference(core,arm,mode):
    @jax.jit
    def go(p,r,v,b,prior):
        score,gate=forward(p,core,arm,r,v,b,prior,mode);return jax.nn.softmax(score),jax.nn.sigmoid(gate)
    return go

def predict(p,r,core,arm,nouns,basis,prior,mode):
    parts=[];ck=[];n=len(r["y"])
    for i in range(0,n,128):
        ix=np.minimum(np.arange(i,i+128),n-1)
        v,g=inference(core,arm,mode)(p,{k:jnp.asarray(r[k][ix]) for k in FEATURES},nouns,basis,prior)
        parts.append(np.asarray(v)[:min(128,n-i)]);ck.append(np.asarray(g)[:min(128,n-i)])
    return np.concatenate(parts),np.concatenate(ck)

def lexical(strings,labels):
    vocab=[set(re.findall(r"[a-z0-9]+",s.replace("_"," ").lower())) for s in labels]
    counts={}
    for v in vocab:
        for token in v:counts[token]=counts.get(token,0)+1
    idf={k:np.log(1+len(vocab)/(v+1)) for k,v in counts.items()}
    matrix=[]
    for sentence in strings:
        w=set(re.findall(r"[a-z0-9]+",sentence.lower()))
        matrix.append([sum(idf[t] for t in v&w)/max(1,sum(idf[t] for t in v))**.70 for v in vocab])
    return np.array(matrix,np.float32)

def baselines(data,meta,labels,nouns,prior,mode):
    key={"correct":"text","wrong":"alt","human":"oracle"}.get(mode,None)
    text=data[key].astype("f4") if key else np.zeros((len(data["y"]),384),"f4")
    words=[x["spoken" if mode=="correct" else "foil"] if mode in ("correct","wrong") else "" for x in meta["rows"]]
    prem=np.log(prior[None,:])
    return dict(prior=np.broadcast_to(prior,(len(words),len(prior))),semantic=softmax(12*text@nouns.T+.2*prem),
                word_match=softmax(2.5*lexical(words,labels)+.2*prem))

def main(stage):
    WORK.mkdir(exist_ok=True);dirs=roots("components");inv=json.loads((ROOT_DATA/"inventory.json").read_bytes())
    labels=inv["nouns"];proto=read(ROOT_DATA/"prototypes.npz");nouns=jnp.asarray(proto["semantic"]);basis=jnp.asarray(proto["basis"])
    record={};all_results={};public={};baseref={};coverage={}
    for fold in "AB":
        tr=read(ROOT_DATA/f"{fold}-train.npz");dv=read(ROOT_DATA/f"{fold}-dev.npz");prior=count_prior(tr,len(labels))
        dt={k:jnp.asarray(tr[k]) for k in FEATURES+("y",)}
        if stage=="test":
            ted=read(ROOT_DATA/f"{fold}-test.npz");tm=json.loads((ROOT_DATA/f"{fold}-test.json").read_bytes())
            family=np.array([x["family"] for x in tm["rows"]]);bs=Groups(ted["group"],family)
            seen=set(tr["y"].ravel())-{-1};y=ted["y"];novel=(y>=0).any(1)&np.array([not any(v in seen for v in row if v>=0) for row in y])
            coverage[fold]=dict(test_cases=len(y),source_recordings=len(set(x["recording"] for x in tm["rows"])),known=(y>=0).any(1).sum().item(),
                     all_new_targets=int(novel.sum()),train_cued=int(tr["cued"].sum()),dev_cued=len(dv["y"]),candidate_nouns=len(labels))
            ref={m:{n:task(x,y) for n,x in baselines(ted,tm,labels,proto["full"],prior,m).items()} for m in ("correct","wrong","none")}
            baseref[fold]={m:{n:{k:bs.score(a) for k,a in score.items()} for n,score in src.items()} for m,src in ref.items()}
            baseref[fold]["semantic_vs_wrong"]={k:bs.score(ref["wrong"]["semantic"][k]-a if k=="nll" else a-ref["wrong"]["semantic"][k]) for k,a in ref["correct"]["semantic"].items()}
            public[fold+"_y"]=y;public[fold+"_groups"]=ted["group"];public[fold+"_families"]=family;public[fold+"_novel"]=novel;public[fold+"_row_ids"]=np.array([str(x["recording"])+"/"+str(x["id"]) for x in tm["rows"]])
            saved=json.loads((WORK/"lock.json").read_bytes())
        for core in CORES:
            for arm in ("learned","semantic"):
                for seed in SEEDS:
                    tag=f"{fold}-{core}-{arm}-{seed}";cfg=dict(fold=fold,core=core,arm=arm,seed=seed,steps=list(STEPS),batch=128,
                         train_sha=json.loads((ROOT_DATA/f"{fold}-train.json").read_bytes())["sha256"],prototype_sha=sha(ROOT_DATA/"prototypes.npz"))
                    if stage=="develop":
                        p=initial(seed);m=jax.tree.map(jnp.zeros_like,p);v=jax.tree.map(jnp.zeros_like,p);idx=draws(tr,seed);start=0;curve=[]
                        for stop in STEPS:
                            if complete(dirs,tag,stop):
                                p,m,v,curve=load(dirs,tag,stop,cfg);start=stop;continue
                            tic=time.monotonic()
                            for i in range(start,stop,16):
                                (p,m,v),ls=train_block(core,arm)(p,m,v,dt,jnp.asarray(idx[i:i+16]),jnp.int32(i),nouns,basis,jnp.asarray(prior))
                                if not np.isfinite(np.asarray(ls)).all():raise ValueError("Invalid component update")
                            duration=time.monotonic()-tic;pp,certainty=predict(p,dv,core,arm,nouns,basis,prior,"correct")
                            met=task(pp,dv["y"]);curve.append(dict(step=stop,seconds=duration,**{k:group_mean(a,dv["group"]) for k,a in met.items()},
                                 threshold=float(np.quantile(.5*(pp.max(1)+certainty),.6))))
                            save(dirs,tag,stop,p,m,v,cfg,curve);start=stop
                        best=min(curve,key=lambda q:(q["nll"],q["step"]))
                        record[tag]=dict(best=best,curve=curve,sha=sha(dirs[0]/tag/f"step-{best['step']:05d}"/"weights.npz"),parameters=sum(x.size for x in p.values()))
                    else:
                        entry=saved[tag];pfile=dirs[0]/tag/f"step-{entry['best']['step']:05d}"/"weights.npz"
                        if sha(pfile)!=entry["sha"]:raise ValueError("Changed component model")
                        p={k:jnp.asarray(x) for k,x in read(pfile).items()};ms={}
                        for mode in (("correct","wrong","none","human","wrong_human","words","acoustic","no_scene") if seed==1301 else ("correct","wrong","none")):
                            pp,gate=predict(p,ted,core,arm,nouns,basis,prior,mode);met=task(pp,y);ms[mode]=met
                            public[tag+"_"+mode+"_top"]=np.argsort(-pp,axis=1)[:,:5].astype("i2")
                            if mode=="correct":committed=.5*(pp.max(1)+gate)>=entry["best"]["threshold"]
                        def gain(a,b,k,mask=None):
                            v=b-a if k=="nll" else a-b
                            return bs.score(v if mask is None else np.where(mask,v,np.nan))
                        correct=ms["correct"]
                        all_results[tag]=dict(modes={m:{k:bs.score(v) for k,v in y0.items()} for m,y0 in ms.items()},selected=entry["best"],
                          vs_wrong={k:gain(a,ms["wrong"][k],k) for k,a in correct.items()},vs_none={k:gain(a,ms["none"][k],k) for k,a in correct.items()},
                          vs_raw_semantic={k:gain(a,ref["correct"]["semantic"][k],k) for k,a in correct.items()},
                          novel_hit=bs.score(np.where(novel,correct["hit"],np.nan)),novel_vs_wrong= gain(correct["hit"],ms["wrong"]["hit"],"hit",novel),
                          gate_coverage=bs.score(committed),conditional_hit=bs.score(np.where(committed,correct["hit"],np.nan)))
        print("Component",stage,fold,flush=True)
    if stage=="develop":write(WORK/"lock.json",record)
    else:
        dest=ROOT/"release/metrics"
        write(dest/"component-results.json",dict(results=all_results,baselines=baseref,coverage=coverage,nouns=labels))
        np.savez_compressed(dest/"component-predictions.npz",**public)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("develop","test"));a=p.parse_args()
    jax.config.update("jax_compilation_cache_dir",str(ROOT/"cache"));main(a.stage)
