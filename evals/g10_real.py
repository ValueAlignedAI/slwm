"""Held recording retrieval, conditional reconstruction and strictly pre-action outcome readouts."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from models.g9_shared import decode
from models.g10_alignment import encode_record
from evals.g9_invariance import center,matching
from evals.g10_control import load_stage
from scripts.train.g9_fit import save_json

SOURCES=("audio","vision","geometry","words")
MODES={**{name:tuple(int(j==i) for j in range(4)) for i,name in enumerate(SOURCES)},
       **{"hide_"+name:tuple(int(j!=i) for j in range(4)) for i,name in enumerate(SOURCES)},
       "all":(1,1,1,1)}
N=32


class Bank:
    def __init__(self,file,holo):
        with np.load(file,allow_pickle=False) as f:
            self.data={k:f[k] for k in ("a","v","n","w")}
            self.data["x"]=f["x"] if "x" in f else np.zeros((len(self.data["a"]),128),np.int8)
            self.names=list(map(str,f["names"]));self.lengths=f["lengths"].astype(int)
        self.origins=np.cumsum(np.r_[0,self.lengths[:-1]]);self.holo=holo

    def samples(self,starts,group):
        ix=starts[:,None]+np.arange(N)
        previous=np.maximum(ix-1,self.origins[group,None])
        r={k:v[ix] for k,v in self.data.items()}
        motion=np.abs(self.data["v"][ix].astype(np.int16)-self.data["v"][previous].astype(np.int16))
        r["m"]=np.clip(motion.reshape(*ix.shape,256,3).mean(-1,dtype=np.float32)/64,0,2)
        r["u"]=np.zeros(ix.shape,np.int32)
        return r

    def selection(self):
        starts=np.concatenate([o+np.linspace(0,n-N-1,8,dtype=int) for o,n in zip(self.origins,self.lengths)])
        return starts,np.repeat(np.arange(len(self.names)),8)


def tags(words):
    return (words[...,None]==np.array([1,2,3,*range(32,45)])).any(-2).astype(np.float32)


def target(row,k,holo):
    if k=="audio":return row["a"].astype(np.float32)/127.5-1
    if k=="vision":return row["v"].astype(np.float32)/127.5-1
    if k=="geometry":return row["n"]
    return row["x"].astype(np.float32)/127 if holo else tags(row["w"])


def priors(bank):
    sums={};ns={}
    for start in range(0,len(bank.data["a"]),4096):
        r={k:v[start:start+4096] for k,v in bank.data.items()}
        for k in SOURCES:
            vals=target(r,k,bank.holo)
            if k=="words" and bank.holo:vals=vals[(r["x"]!=0).any(-1)]
            if len(vals):
                sums[k]=sums.get(k,0)+vals.sum(0,dtype=np.float64);ns[k]=ns.get(k,0)+len(vals)
    return {k:(v/ns[k]).astype(np.float32) for k,v in sums.items()}


@lru_cache(maxsize=24)
def predictor(spec,just_features):
    @jax.jit
    def run(p,r,mask):
        z=encode_record(p,spec,r,jnp.broadcast_to(mask,(r["u"].shape[0],4)))
        if just_features:return z[:,-1].astype(jnp.float32)
        pred=decode(p,z,r["u"])
        out={name:pred[key] for name,key in zip(SOURCES,("a","v","n","w"))}
        out["natural"]=(z@p["utterance_read"].astype(z.dtype)).astype(jnp.float32)
        return z[:,-8:].astype(jnp.float32).mean(1),out
    return run


def block_error(pred,truth,holo,k):
    if k=="words" and not holo:return (np.logaddexp(0,pred)-truth*pred).mean(-1)
    return ((pred-truth)**2).mean(-1)


def panel(p,spec,bank,prior):
    start,group=bank.selection();features={k:[] for k in SOURCES};losses={};raw={k:[] for k in SOURCES}
    # Batch whole eight-way candidate sets together so the timing shuffle never crosses a recording.
    for begin in range(0,len(start),64):
        ids=slice(begin,begin+64);row=bank.samples(start[ids],group[ids])
        def activity(arr):return arr[:,-8:].reshape(-1,8,8)
        raw["audio"].append(activity(row["a"].astype(np.float32).mean(-1)))
        raw["vision"].append(activity(row["m"].mean(-1)))
        raw["geometry"].append(activity(np.linalg.norm(row["n"][:,:,:3],axis=-1) if bank.holo else row["n"][:,:,3]))
        word_activity=np.linalg.norm(np.diff(row["x"].astype(np.float32),axis=1,prepend=row["x"][:,:1]),axis=-1) if bank.holo else (row["w"][:,:,0]==24)
        raw["words"].append(activity(word_activity))
        for mode in (*SOURCES,*("hide_"+k for k in SOURCES)):
            field,output=predictor(spec,False)(p,row,np.array(MODES[mode],bool))
            if mode in SOURCES:features[mode].append(np.asarray(field).reshape(-1,8,spec.field))
            targets=(mode[5:],) if mode.startswith("hide_") else ("vision",) if mode=="audio" else ("audio",) if mode=="vision" else ("audio","vision","geometry") if mode=="words" else ()
            for k in targets:
                pred=np.asarray(output["natural" if k=="words" and bank.holo else k])
                truth=target(row,k,bank.holo)
                shape=(-1,8,N,pred.shape[-1]);pp=pred.reshape(shape);tt=truth.reshape(shape)
                weight=(np.any(tt!=0,axis=-1) if k=="words" and bank.holo else np.ones(tt.shape[:3],bool))
                if k=="words" and not bank.holo:
                    prob=np.clip(prior[k],1e-6,1-1e-6);prior_pred=np.log(prob)-np.log1p(-prob)
                else:prior_pred=prior[k]
                result=losses.setdefault(mode+"/"+k,dict(aligned=[],shifted=[],prior=[],valid=[]))
                den=weight.sum((1,2));result["valid"].extend(den.astype(int).tolist())
                for label,arr in (("aligned",pp),("shifted",np.roll(pp,4,axis=1)),("prior",prior_pred)):
                    err=block_error(arr,tt,bank.holo,k)
                    means=(err*weight).sum((1,2))/np.maximum(1,den)
                    result[label].extend([float(v) if d else None for v,d in zip(means,den)])
    fs={k:center(np.concatenate(v).astype(np.float64)) for k,v in features.items()}
    activity={k:center(np.concatenate(v).astype(np.float64)) for k,v in raw.items()}
    scores={}
    for k in SOURCES:
        for j in SOURCES:
            if k!=j:
                scores[k+"_to_"+j]={"direct":matching(fs[k],fs[j]),"raw_activity":matching(activity[k],activity[j])}
    return dict(recordings=bank.names,chance=.125,retrieval=scores,losses=losses,
                self_retrieval={k:matching(v,v) for k,v in fs.items()})


def outcomes(bank,path,role):
    starts=[];labels=[];groups=[];tasks=[]
    for i,name in enumerate(bank.names):
        with np.load(path/"shards"/(name+".npz")) as db:ev=db["events"]
        note=json.loads((path/"shards"/(name+".json")).read_text());tasks.append(note["task"])
        if role=="train" and len(ev)>96:
            seed=int(hashlib.sha256(("G10/risk/"+name).encode()).hexdigest()[:8],16)
            ix=np.sort(np.random.default_rng(seed).choice(len(ev),96,replace=False));ev=ev[ix]
        if len(ev):
            starts.extend((bank.origins[i]+ev[:,0]-N+1).tolist());labels.extend(ev[:,1].tolist());groups.extend([i]*len(ev))
    return np.asarray(starts,int),np.asarray(labels,int),np.asarray(groups,int),np.asarray(tasks)


def raw_features(row):
    n=len(row["a"])
    image=row["v"].astype(np.float32).reshape(n,N,4,4,4,4,3).mean((3,5)).reshape(n,N,48)/127.5-1
    sensor=np.concatenate([row["a"].astype(np.float32)/127.5-1,image,row["n"]],-1)
    words=row["x"].astype(np.float32)/127
    return np.concatenate([sensor[:,:8].mean(1),sensor[:,-8:].mean(1),sensor.std(1),words[:,-1],words.mean(1)],-1)


def outcome_features(p,s,bank,events,with_raw):
    starts,y,groups,tasks=events;out={k:[] for k in ("all","no_words","words_only")}
    raw=[]
    for i in range(0,len(starts),128):
        row=bank.samples(starts[i:i+128],groups[i:i+128])
        for k,mask in (("all",(1,1,1,1)),("no_words",(1,1,1,0)),("words_only",(0,0,0,1))):
            out[k].append(np.asarray(predictor(s,True)(p,row,np.array(mask,bool))))
        if with_raw:raw.append(raw_features(row))
    out={k:np.concatenate(v) for k,v in out.items()}
    if with_raw:out["raw"]=np.concatenate(raw)
    return out


def logistic(x,y):
    mu=x.mean(0);sd=np.maximum(x.std(0),.04)
    x=np.asarray((x-mu)/sd,np.float64);yf=y.astype(float)
    def obj(theta):
        w,b=theta[:-1],theta[-1];logit=x@w+b;delta=expit(logit)-yf
        return np.mean(np.logaddexp(0,logit)-yf*logit)+.005*(w@w),np.r_[x.T@delta/len(x)+.01*w,delta.mean()]
    init=np.r_[np.zeros(x.shape[1]),np.log((.5+y.sum())/(.5+len(y)-y.sum()))]
    res=minimize(obj,init,jac=True,method="L-BFGS-B",options={"maxiter":100,"ftol":1e-10})
    return dict(mean=mu,scale=sd,weight=res.x[:-1],intercept=np.asarray(res.x[-1]))


def predict_head(h,x):
    return expit((x-h["mean"])/h["scale"]@h["weight"]+h["intercept"])


def risk(p,s,banks,events,cache,out,tag):
    names=("train","familiar","novel")
    all_features={}
    for role in names:
        feats=outcome_features(p,s,banks[role],events[role],role not in cache)
        if "raw" in feats:cache[role]=feats.pop("raw")
        feats["raw"]=cache[role];all_features[role]=feats
    y=events["train"][1];labels={};heads={}
    for k,x in all_features["train"].items():heads[k]=logistic(x,y)
    arrays={k+"_"+a:v for k,h in heads.items() for a,v in h.items()}
    file=out/(tag+"-risk-heads.npz");np.savez_compressed(file,**arrays)
    for role in ("familiar","novel"):
        _,truth,group,tasks=events[role];preds={k:predict_head(heads[k],x).astype(np.float32) for k,x in all_features[role].items()}
        train_tasks=events["train"][3][events["train"][2]]
        lookup={task:float((y[train_tasks==task].sum()+1)/(np.sum(train_tasks==task)+2)) for task in set(train_tasks)}
        preds["prevalence"]=np.full(len(truth),y.mean(),np.float32)
        preds["task_prior"]=np.array([lookup.get(t,float(y.mean())) for t in tasks[group]],np.float32)
        np.savez_compressed(out/(tag+"-"+role+"-risk.npz"),y=truth,group=group,names=np.array(banks[role].names),**preds)
        labels[role]=dict(examples=len(truth),mistakes=int(truth.sum()),recordings=int(len(set(group))))
    return dict(training_examples=len(y),training_mistakes=int(y.sum()),held=labels,head_sha256=hashlib.sha256(file.read_bytes()).hexdigest())


def main(root,data,starss,out):
    out.mkdir(exist_ok=True,parents=True)
    banks={k:Bank(data/("holo-"+k+".npz"),True) for k in ("train","familiar","novel")}
    star_train=Bank(starss/"starss-train.npz",False);bank_prior=priors(star_train);del star_train
    panels={**{k:banks[k] for k in ("familiar","novel")},"sony_test":Bank(starss/"starss-test.npz",False),"sony_extra":Bank(starss/"starss-invariance.npz",False)}
    holo_prior=priors(banks["train"]);ev={k:outcomes(v,data,k) for k,v in banks.items()};raw_cache={}
    wanted=[f"align-{c}-{mode}-s{seed}" for seed in (1001,1002,1003) for mode in ("paired","reconstruct","shifted","rooms") for c in ("hybrid","gpt2")]
    while wanted:
        worked=False
        for tag in list(wanted):
            job=root/tag/"step-01536";file=out/(tag+".json")
            if file.exists():wanted.remove(tag);continue
            if not (job/"verified.json").exists():continue
            print("reading",tag,flush=True);d,p,s=load_stage(job)
            doc=dict(tag=tag,sha256=d["inference_sha256"],backend=jax.default_backend(),panels={})
            for key,bank in panels.items():doc["panels"][key]=panel(p,s,bank,holo_prior if bank.holo else bank_prior)
            doc["risk"]=risk(p,s,banks,ev,raw_cache,out,tag)
            save_json(file,doc);wanted.remove(tag);worked=True
            print(tag,"novel AV",round(np.mean(doc["panels"]["novel"]["retrieval"]["audio_to_vision"]["direct"]),4),flush=True)
        if wanted and not worked:time.sleep(10)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("runs",type=Path);ap.add_argument("holo",type=Path);ap.add_argument("starss",type=Path);ap.add_argument("out",type=Path)
    ap.add_argument("--cache");a=ap.parse_args()
    if a.cache:jax.config.update("jax_compilation_cache_dir",a.cache)
    main(a.runs,a.holo,a.starss,a.out)
