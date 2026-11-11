"""Independently check metric registration, physical coverage and the information in speech."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
from g11_geometry import make_rays,cloud,grid,delta
from g11_events import KEYWORDS
from scripts.train.g11_analysis import estimate


def fixture():
    intr=np.r_[np.array([[250,0,255.5],[0,250,255.5],[0,0,1]]).ravel(),np.zeros(8),512,512]
    ray=make_rays(intr);anchor=np.array([.50,0,0]);basis=np.array([[0,0,1],[-1,0,0],[0,-1,0]])
    def surface(camera_x,lateral,change):
        pose=np.eye(4);pose[:3,3]=[camera_x,lateral,0]
        u=-(ray[:,:,1]*(.5-camera_x)/ray[:,:,0]+lateral)
        x=.5+change*(u>0);raw=np.zeros((512,512),np.uint16);raw[::2,::2]=np.rint(1000*(x-camera_x)/ray[:,:,0]).astype(np.uint16)
        pts,_=cloud(raw,pose,ray,np.empty((0,3)));return grid(pts,anchor,basis),raw,pose
    a,raw,pose=surface(0,0,0);b,_,_=surface(.02,.015,0)
    y,_,_=surface(0,0,.05);z,_,_=surface(.02,.012,.05)
    static=delta(np.stack([a,b,b,a]));moved=delta(np.stack([a,b,y,z]));hand=np.array([[.5,0,0]])
    cleared,removed=cloud(raw,pose,ray,hand)
    if static is None or moved is None or static["changed"]>.01 or not .43<=moved["changed"]<=.57 or removed==0 or np.min(np.linalg.norm(cleared-hand,axis=1))<=.045:
        raise ValueError("Known geometry or hand-removal fixture failed")
    return dict(static_view_change_fraction=static["changed"],known_half_patch_shift_fraction=moved["changed"],
                known_shift_m=.05,recovered_mean_magnitude_mm=moved["magnitude_mm"],hand_points_removed=int(removed))


def explicit(text):
    phrase=" "+" ".join(re.findall(r"[a-z0-9']+",str(text).casefold()))+" "
    return {key for key,aliases in KEYWORDS.items() if any(" "+a+" " in phrase for a in aliases)}


def main(root,out,audit):
    groups=json.loads((root/"groups.json").read_text());catalog=json.loads((root/"annotations.json").read_text())
    original=json.loads(Path("/mnt/slwm-data/SLWM/g10/data/holo/labels.json").read_text())
    source={r["video_name"]:r for r in original};grades=json.loads(audit.read_text());rating={(r["recording"],r["id"]):r["grade"] for r in grades["rows"]}
    doc=dict(fixture=fixture(),manual_audit=grades["counts"],primary_new_recordings=96,folds={},asr_revisions=dict(primary="87c7102498dcde7456f24cfd30239ca606ed9063",stronger="e8727524f962ee844a7319d92be39ac1bd25655a"))
    for fold,roles in groups["folds"].items():
        fold_doc={}
        for role,names in roles.items():
            families=np.array([catalog[n]["family"] for n in names]);idx=[];samples=[];quality=[];qgroup=[];asr={k:[0,0] for k in ("tiny","small")}
            reasons=Counter();candidate=0;mentions={};noun_counts=Counter()
            for j,n in enumerate(names):
                src=catalog[n];speech={v["id"]:v for v in src["speech"]}
                h={}
                if src["events"]:
                    h={k:json.loads((root/d/(n+".json")).read_text())["utterances"] for k,d in (("tiny","speech"),("small","speech-small"))}
                    unique={e["cue"] for e in src["events"]}
                    for k in h:
                        for i in unique:
                            v=h[k][str(i)];asr[k][0]+=v["edits"];asr[k][1]+=v["words"]
                acts=sorted([v for v in source[n]["events"] if v["label"]=="Fine grained action"],key=lambda a:(a["start"],a["id"]))
                lookup={a["id"]:i for i,a in enumerate(acts)}
                for ev in src["events"]:
                    idx.append(j);samples.append(ev);start=lookup[ev["id"]];noun_counts[ev["noun"]]+=1
                    for channel in ("human","tiny","small"):
                        for which in ("cue","wrong"):
                            text=speech[ev[which]]["text"] if channel=="human" else h[channel][str(ev[which])]["hypothesis"]
                            predicted=explicit(text)
                            for horizon in (1,3,5):
                                upcoming={str(a["attributes"].get("Verb","")).casefold() for a in acts[start:start+horizon] if a["start"]<=ev["start"]+12}
                                mentions.setdefault(f"{channel}-{which}-{horizon}",[]).append(float(bool(predicted&upcoming)))
                geo=json.loads((root/"geometry"/(n+".json")).read_text());candidate+=geo["attempted"];reasons.update(geo["rejected"].values())
                for v in geo["valid"]:
                    quality.append({**v,"device_grade":rating.get((n,v["id"]))})
                    qgroup.append(j)
            idx=np.asarray(idx,int);qgroup=np.asarray(qgroup,int)
            def avg(values,ix):return estimate(values,ix,families,True)
            stats={k:avg([np.nan if v[k] is None else v[k] for v in quality],qgroup) for k in ("changed","neutral","background_changed","cells","head_shift_m","head_rotation_deg")}
            for control in ("neutral","background_changed"):
                vals=[v["changed"]-v[control] if v[control] is not None else np.nan for v in quality]
                stats["foreground_minus_"+control]=avg(vals,qgroup)
            exposure={k:avg(v,idx) for k,v in mentions.items()}
            for channel in ("human","tiny","small"):
                for horizon in (1,3,5):
                    exposure[f"{channel}-paired-minus-wrong-{horizon}"]=avg(np.asarray(mentions[f"{channel}-cue-{horizon}"])-mentions[f"{channel}-wrong-{horizon}"],idx)
            fold_doc[role]=dict(recordings=len(names),eligible_cues=len(samples),eligible_recordings=len(np.unique(idx)),explicit_conflicts=sum(v["conflict"] for v in samples),
                mean_gap_s=float(np.mean([v["gap"] for v in samples])),candidate_depth=candidate,measured=len(quality),measured_groups=len(np.unique(qgroup)),
                hand_adjacent_isolated=sum(v["isolated"] and v["anchor_source"]=="hand_adjacent" for v in quality),clear_device=sum(v["device_grade"]=="D" for v in quality),
                quality=stats,depth_exclusions=dict(reasons),word_error={k:dict(edit_count=v[0],reference_words=v[1],wer=v[0]/max(1,v[1])) for k,v in asr.items()},
                lexical_horizon=exposure)
        doc["folds"][fold]=fold_doc
    out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("registration and hand-exclusion fixtures passed",doc["fixture"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("out",type=Path);p.add_argument("audit",type=Path);a=p.parse_args();main(a.root,a.out,a.audit)
