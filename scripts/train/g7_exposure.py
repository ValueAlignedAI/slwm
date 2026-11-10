"""Replay the exact sampling schedule to count changed rows and independent identities over all updates."""
import argparse
import json
from pathlib import Path
import numpy as np
from training.g5_batches import JointData,FAMILIES
from training.g7_batches import ContinuationData
from training.g7_units import put


def run(root,old,out):
    base=JointData(old,"train",verify=False); bank=ContinuationData(base,root/"fresh")
    exposures=np.zeros(9,np.int64); changed=np.zeros(9,np.int64)
    visits={f:np.zeros(len(pool),np.int64) for f,pool in bank.pools.items()};orders={}
    patterns={0:[1,1,0,1],1:[1,0,0,1],5:[0,0,0,1],7:[1,1,0,0],8:[0,1,1,0]}
    names=("audio","video","numeric","text")
    modality={k:np.zeros(4,np.int64) for k in ("present_rows","conditioning_rows","conditioning_ticks","cross_or_language_target_rows")}
    event_rows=dynamics_rows=0
    for step in range(8192,16384):
        b=base.requests(step,256);rng=np.random.default_rng([71110,step,113])
        exposures+=np.bincount(b["family"],minlength=9)
        has=b["has"].astype(bool);src=b["source"].astype(bool);fam=b["family"]
        modality["present_rows"]+=has.sum(0);modality["conditioning_rows"]+=src.sum(0)
        usable=(b["sense"].astype(bool)&b["ok"].astype(bool)).sum(1)
        modality["conditioning_ticks"]+=(src*usable[:,None]).sum(0)
        targets=has&~src;targets[:,3]|=fam==5;targets&=(fam!=8)[:,None]
        modality["cross_or_language_target_rows"]+=targets.sum(0)
        event_rows+=int(((fam==7)&src[:,1]&~src[:,0]).sum());dynamics_rows+=int((fam==8).sum())
        for f,pool in bank.pools.items():
            members=np.flatnonzero(b["family"]==f);half=len(members)//2; chosen=rng.permutation(members)[:half]
            ok=(b["has"][chosen]==patterns[f]).all(1)&(b["ok"][chosen].sum(1)==(32 if f==8 else 64))
            ordinal=(step-8192)*half+np.arange(half)[ok]
            changed[f]+=len(ordinal)
            for epoch in np.unique(ordinal//len(pool)):
                key=(f,int(epoch))
                if key not in orders:
                    orders[key]=np.random.default_rng([77113,f,int(epoch)]).permutation(len(pool))
                ids=orders[key][ordinal[ordinal//len(pool)==epoch]%len(pool)]
                np.add.at(visits[f],ids,1)
    result={"updates":[8192,16384],"rows_per_arm":int(exposures.sum()),"actual_changed":int(changed.sum()),
            "changed_fraction":float(changed.sum()/exposures.sum()),"counts":{}}
    result["modalities"]={k:{name:int(x) for name,x in zip(names,v)} for k,v in modality.items()}
    result["specialized_supervision_rows"]={"action_conditioned_future":dynamics_rows,"video_only_event_timing":event_rows}
    result["modality_note"]="Several modalities can be present, conditioning, or targets on one row. Target counts precede per-row objective weighting. Text-only rows include next-byte prediction."
    for f,name in enumerate(FAMILIES):
        v=visits.get(f,np.zeros(0,int))
        result["counts"][name]={"presentations":int(exposures[f]),"fresh_presentations":int(changed[f]),"eligible_pool":len(v),
             "unique_fresh_rows_seen":int((v>0).sum()),"maximum_repeats":int(v.max()) if len(v) else 0}
    if exposures.sum()!=8192*256 or sum(v.sum() for v in visits.values())!=changed.sum():raise ValueError("Incomplete accounting")
    put(out,result);print(json.dumps(result),flush=True)


if __name__=="__main__":
    q=argparse.ArgumentParser()
    for k in ("root","old","out"):q.add_argument("--"+k,type=Path,required=True)
    a=q.parse_args();run(a.root,a.old,a.out)
