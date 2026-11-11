"""Paired recording-level inference and clustered ranking uncertainty for future human action errors."""
import argparse
import json
from pathlib import Path
import numpy as np
from g10_analysis import SEEDS,effect,read,custody
from g9_storage import write

MODES=("paired","reconstruct","shifted","rooms")
PANELS=("novel","familiar","sony_test","sony_extra")


def nonmissing(values):
    return np.asarray([np.nan if x is None else x for x in values],np.float64)


def intervals(arr):
    x=np.asarray(arr,np.float64)
    return [float(v) for v in np.nanquantile(x,[.025,.975])]


def weighted_ranking(y,pred,groups,draws):
    # Scores are sorted once. Entire recordings receive a bootstrap multiplicity.
    order=np.argsort(-pred,kind="stable");pred=pred[order];ys=y[order];gs=groups[order]
    ends=np.r_[np.flatnonzero(np.diff(pred)!=0),len(y)-1]
    residual=(pred-ys)**2
    brier=np.bincount(gs,weights=residual,minlength=draws.shape[1]);freq=np.bincount(gs,minlength=draws.shape[1])
    ret={k:[] for k in ("auroc","ap","brier")}
    for batch in np.array_split(draws,max(1,(len(draws)+31)//32)):
        w=batch[:,gs]
        tp=np.cumsum(w*ys,axis=1)[:,ends];fp=np.cumsum(w*(1-ys),axis=1)[:,ends]
        positives=tp[:,-1:];negatives=fp[:,-1:]
        tpr=tp/np.maximum(1,positives);fpr=fp/np.maximum(1,negatives)
        prec=tp/np.maximum(1,tp+fp)
        ap=np.sum(np.diff(tpr,axis=1,prepend=0)*prec,1)
        roc=np.sum(.5*(tpr+np.concatenate([np.zeros((len(w),1)),tpr[:,:-1]],1))*np.diff(fpr,axis=1,prepend=0),1)
        roc[(positives[:,0]==0)|(negatives[:,0]==0)]=np.nan;ap[positives[:,0]==0]=np.nan
        ret["auroc"].extend(roc);ret["ap"].extend(ap);ret["brier"].extend((batch@brier)/np.maximum(1,batch@freq))
    return {k:np.asarray(v) for k,v in ret.items()}


def risk_scores(file,cache):
    with np.load(file,allow_pickle=False) as db:
        y=db["y"].astype(int);g=db["group"].astype(int);names=db["names"];ids=np.unique(g)
        index=np.searchsorted(ids,g);n=len(ids);rng=np.random.default_rng(71932)
        weights=np.r_[np.ones((1,n),int),rng.multinomial(n,np.full(n,1/n),size=1000)]
        values={};metrics={}
        for key in ("raw","prevalence","task_prior","all","no_words","words_only"):
            pred=db[key].astype(np.float64)
            if key in ("raw","prevalence","task_prior"):
                digest=(key,hash((y.tobytes(),g.tobytes(),pred.tobytes())))
                if digest not in cache:cache[digest]=weighted_ranking(y,pred,index,weights)
                scores=cache[digest]
            else:scores=weighted_ranking(y,pred,index,weights)
            values[key]=scores
            metrics[key]={m:dict(mean=float(v[0]),ci95=intervals(v[1:])) for m,v in scores.items()}
        gains={}
        for ref in ("raw","prevalence","task_prior","no_words","words_only"):
            gains[ref]={m:dict(mean=float((v-values[ref][m])[0]),ci95=intervals((v-values[ref][m])[1:])) for m,v in values["all"].items()}
        return dict(recordings=n,events=len(y),mistakes=int(y.sum()),prevalence=float(y.mean()),metrics=metrics,all_minus=gains)


def observed(roles,doc):
    out={}
    for role in roles:
        r=doc["panels"][role]
        ret={}
        for direction,values in r["retrieval"].items():
            v=values["direct"];raw=values["raw_activity"]
            ret[direction]=dict(direct=float(np.mean(v)),raw=float(np.mean(raw)),minus_chance=effect(v,np.full(len(v),.125)),minus_raw=effect(v,raw))
        conditional={}
        for key,v in r["losses"].items():
            xs={k:nonmissing(v[k]) for k in ("aligned","shifted","prior")}
            conditional[key]={k:float(np.nanmean(x)) for k,x in xs.items()}
            conditional[key].update(shifted_minus_aligned=effect(xs["shifted"],xs["aligned"]),prior_minus_aligned=effect(xs["prior"],xs["aligned"]))
        out[role]=dict(recordings=len(r["recordings"]),retrieval=ret,conditional=conditional,
                       self_retrieval={k:float(np.mean(v)) for k,v in r["self_retrieval"].items()})
    return out


def analyze(evaldir,dest):
    dest.mkdir(exist_ok=True,parents=True);rows={};sources={};art={};cache={};unit_arrays={}
    for core in ("hybrid","gpt2"):
        for seed in SEEDS:
            for mode in MODES:
                tag=f"align-{core}-{mode}-s{seed}";doc=read(evaldir/"real"/(tag+".json"));source=custody(tag,1536);art[tag+"/1536"]=source
                if source["inference_sha256"]!=doc["sha256"]:raise ValueError("Real evaluation generation mismatch")
                live=read(evaldir/"control"/(tag+"-1536.json"))
                if live["sha256"]!=doc["sha256"]:raise ValueError("Post-training control identity mismatch")
                out=observed(PANELS,doc);sources[tag]=(doc,live)
                risk={role:risk_scores(evaldir/"real"/(tag+"-"+role+"-risk.npz"),cache) for role in ("familiar","novel")}
                rows[tag]=dict(sha256=doc["sha256"],panels=out,risk=risk,readout=doc["risk"],control={k:float(np.mean(live[k]["occupancy"])) for k in ("full","changes","block32","wrong")})
                for role in PANELS:
                    panel=doc["panels"][role]
                    for name,v in panel["retrieval"].items():unit_arrays[tag+"|"+role+"|"+name]=np.asarray(v["direct"],"f4")
                    for name,v in panel["losses"].items():
                        for cond in ("aligned","shifted","prior"):unit_arrays[tag+"|"+role+"|"+name+"|"+cond]=nonmissing(v[cond]).astype("f4")
    contrasts={};gates={}
    for core in ("hybrid","gpt2"):
        gates[core]={}
        for seed in SEEDS:
            form=lambda mode:f"align-{core}-{mode}-s{seed}"
            paired,live=sources[form("paired")];relative={}
            for mode in ("reconstruct","shifted","rooms"):
                alt,altlive=sources[form(mode)]
                relative[mode]={role:effect(paired["panels"][role]["retrieval"]["audio_to_vision"]["direct"],alt["panels"][role]["retrieval"]["audio_to_vision"]["direct"]) for role in PANELS}
                relative[mode]["control"]=effect(live["full"]["occupancy"],altlive["full"]["occupancy"])
            parent=read(evaldir/"control"/f"context-{core}-variable-s{seed}-5120.json")
            relative["parent_control"]=effect(live["full"]["occupancy"],parent["full"]["occupancy"])
            contrasts[f"{core}-s{seed}"]=relative
            gates[core][str(seed)]={role:bool(relative["reconstruct"][role]["ci95"][0]>0 and relative["shifted"][role]["ci95"][0]>0 and rows[form("paired")]["panels"][role]["retrieval"]["audio_to_vision"]["minus_chance"]["ci95"][0]>0) for role in PANELS}
    doc=dict(fits=24,seeds=SEEDS,rows=rows,paired_contrasts=contrasts,gates=gates,decision_bootstrap="1000 whole-recording resamples; tied scores grouped",
             signal_bootstrap="4000 whole-recording resamples within each seed",threshold="paired 95% lower bounds above both controls and chance")
    write(dest/"grounding.json",doc);write(dest/"grounding-artifacts.json",art);np.savez_compressed(dest/"grounding-paired.npz",**unit_arrays)
    return doc


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("eval",type=Path);ap.add_argument("dest",type=Path);a=ap.parse_args()
    print(analyze(a.eval,a.dest)["gates"])
