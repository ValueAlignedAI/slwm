"""Directional timing and wrong-source stress tests of the locked alignment heads and a raw-motion control."""
import argparse
from pathlib import Path
import numpy as np
import torch
from training.g4_data import digest
from training.g7_alignment import Views,Readout,meta,gpu,save


def compare(score,groups):
    correct=score[:,0,None]
    wins=(correct>score[:,1:]).astype(float)+.5*(correct==score[:,1:])
    top=score==score.max(1,keepdims=True)
    values=np.column_stack((wins,top[:,0]/top.sum(1)))
    unique,ix=np.unique(groups,return_inverse=True);counts=np.bincount(ix)
    group_sum=np.stack([np.bincount(ix,weights=v) for v in values.T],axis=1)
    draw=np.random.default_rng(77311).integers(len(unique),size=(2000,len(unique)))
    boot=group_sum[draw].sum(1)/counts[draw].sum(1)[:,None]
    return {"eligible":len(groups),"groups":len(unique),"conditions":{k:{"accuracy":float(values[:,i].mean()),
            "group_ci95":np.quantile(boot[:,i],[.025,.975]).tolist()} for i,k in enumerate(("correct_vs_older","correct_vs_newer","correct_vs_wrong","four_way"))}}


def raw(a,v,m):
    # Broadband upward spectral flux and absolute pixel motion; score their within-window correlation.
    n=m.sum(1,keepdims=True).clip(1);aa=a-(a*m).sum(1,keepdims=True)/n;vv=v-(v*m).sum(1,keepdims=True)/n
    return (aa*vv*m).sum(1)/np.sqrt(((aa**2*m).sum(1)*(vv**2*m).sum(1)).clip(1e-8))


def main(a):
    lock=meta(a.ck/"alignment/test-lock.json");out=a.ck/"timing-audit";out.mkdir(exist_ok=True)
    for name,registered in lock["models"].items():
        if not name.endswith("/units-sync"):continue
        tag=name.split("/")[0];file=out/(tag+".json")
        if file.exists():continue
        weight=a.ck/"alignment"/name/"weights.pt"
        if digest(weight)!=registered["sha256"]:raise ValueError("Timing head differs from pre-test lock")
        model=Readout().cuda().eval();model.load_state_dict(torch.load(weight,map_location="cuda",weights_only=True)["weights"])
        results={}
        with torch.inference_mode():
            for family in ("crema","events","oldimpact"):
                d=Views(a.root,a.old,tag,family,"test");learned=[];baseline=[];groups=[]
                for j in range(0,len(d.ix),64):
                    row=np.arange(j,min(len(d.ix),j+64));row=row[d.wrong[row]>=0];wrong=d.wrong[row]
                    row=row[np.minimum(d.length[row],d.length[wrong])>16];wrong=d.wrong[row]
                    if not len(row):continue
                    valid=(np.arange(8,56)[None]+8<np.minimum(d.length[row],d.length[wrong])[:,None])
                    aa=np.array(d.z[0,False][row],np.float32);vv=np.array(d.z[1,False][row],np.float32);other=np.array(d.z[1,False][wrong],np.float32)
                    fields=(vv[:,8:56],vv[:,:48],vv[:,16:],other[:,8:56])
                    sa,sm=gpu(aa[:,8:56]),gpu(valid)
                    learned.extend(torch.stack([model.match(sa,gpu(v),sm) for v in fields],1).cpu().numpy())
                    a0=d.src["a"][d.ix[row]].astype(np.float32);v0=d.src["v"][d.ix[row]].astype(np.float32);w0=d.src["v"][d.ix[wrong]].astype(np.float32)
                    flux=np.maximum(0,np.diff(a0,axis=1,prepend=a0[:,:1])).mean(-1)
                    def motion(v):return np.abs(np.diff(v,axis=1,prepend=v[:,:1])).mean(-1)
                    v,w=motion(v0),motion(w0);signals=(v[:,8:56],v[:,:48],v[:,16:],w[:,8:56])
                    baseline.extend(np.stack([raw(flux[:,8:56],x,valid) for x in signals],axis=1))
                    groups.extend(d.groups[row])
                results[family]={"common_head":compare(np.asarray(learned),groups),"raw_flux_motion":compare(np.asarray(baseline),groups)}
        save(file,{"head_sha256":registered["sha256"],"signed_shift_seconds":8*1024/24000,
             "design":"Supplementary directional stress test after the original timing scores. Common valid interior and tie credit.", "results":results})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("root","ck","old"):p.add_argument("--"+k,type=Path,required=True)
    main(p.parse_args())
