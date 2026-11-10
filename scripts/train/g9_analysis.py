"""Aggregate paired mission, recording and physical-scene effects while preserving every registered seed."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from g9_storage import disks,write


def effect(a,b):
    d=np.asarray(a,np.float64)-np.asarray(b,np.float64)
    if d.ndim!=1 or not len(d):raise ValueError("Missing paired evaluation units")
    draw=np.random.default_rng(981).integers(0,len(d),(4000,len(d)))
    lo,hi=np.quantile(d[draw].mean(1),[.025,.975])
    return dict(mean=float(d.mean()),ci95=[float(lo),float(hi)],units=len(d))


def grouped(tag):
    size,core,field,act,seed,data=tag.split("-")
    return size,core,int(field[1:]),act,int(seed[1:]),data


def metrics(d):
    a={};vectors={}
    for k,v in d.get("live",{}).items():
        for metric in ("occupancy","success","hold","tap","missed_hold"):vectors["live/"+k+"/"+metric]=v[metric]
    for k in ("matched","wrong","reset","both"):
        if k in d.get("history",{}):vectors["history/"+k]=d["history"][k]
    for k,v in d.get("sim",{}).items():vectors["sim/"+k]=v
    if "room" in d:
        a["real_field_rank"]=d["room"]["rank"]
        for k,v in d["room"]["scores"].items():
            vectors["room/"+k]=v["loss"];vectors["prior/"+k]=v["prior"]
    if "contact" in d:
        for k,v in d["contact"]["rmse_newtons"].items():vectors["contact/"+k]=v
    a["mean"]={k:float(np.mean(v)) for k,v in vectors.items()}
    return a,vectors


def summarize(scores,dest,final=False):
    dest.mkdir(parents=True,exist_ok=True)
    primary,mirror=disks();ledger={};all_rows={};vectors={}
    for p in sorted(primary.glob("*/step-*/verified.json")):
        receipt=json.loads(p.read_text())
        if receipt!=json.loads((mirror/p.relative_to(primary)).read_text()):raise ValueError("Receipt disagrees between checkpoint HDDs")
        d=json.loads((p.parent/"receipt.json").read_text());key=d["tag"]+"/"+str(d["step"])
        ledger[key]={k:d[k] for k in ("tag","step","seed","trainable_parameters","inference_sha256","optimizer_sha256","backend","wall_this_process_s","started_from")}
        for role in ("dev","test"):
            result=scores/f"{d['tag']}-{d['step']}-{role}.json"
            if not result.exists():continue
            doc=json.loads(result.read_text())
            if doc["sha256"]!=d["inference_sha256"]:raise ValueError("Scored weight generation changed")
            item,arrays=metrics(doc);k=key+"/"+role
            all_rows[k]=item;vectors[k]=arrays
    comparisons={}
    lookup=lambda d:"/".join([d["tag"],str(d["step"]),d["role"]])
    def add(family,key,left,right):
        if right not in vectors or left not in vectors:return
        a,b=vectors[left],vectors[right]
        selected=["live/carry/occupancy","live/carry/success","history/matched","contact/frozen","room/a/a/aligned",
                  "room/v/v/aligned","room/n/n/aligned","room/w/w/aligned","sim/all"]
        comparisons.setdefault(family,{})[key]={k:effect(a[k],b[k]) for k in selected if k in a and k in b}
    for key in list(vectors):
        tag,step,role=key.split("/");size,core,field,act,seed,data=grouped(tag)
        if core=="hybrid":add("hybrid_minus_dense",key,key,key.replace("-hybrid-","-gpt2-"))
        if field==512:add("wide_minus_narrow",key,key,key.replace("-z512-","-z128-"))
        if act=="snake":add("snake_minus_gelu",key,key,key.replace("-snake-","-gelu-"))
        if data=="fresh":add("fresh_minus_replay",key,key,key.replace("-fresh/","-replay/"))
        if size=="confirm":add("large_minus_small_equal_exposure",key,key,key.replace("confirm-","exposure-"))
    extras={}
    aux=primary.parent/"auxiliary"
    for f in sorted((aux/"horizon").glob("*.json")):
        doc=json.loads(f.read_text());arms=doc["arms"]
        extras.setdefault("history_window",{})[doc["tag"]]={k:dict(occupancy=float(np.mean(v["occupancy"])),success=float(np.mean(v["success"])),
                    by_goal=v["by_goal"],cache_bytes_per_world=v["cache_bytes_per_world"],warm_seconds=v["warm_wall_s"],
                    against_native=effect(v["occupancy"],arms["carry"]["occupancy"])) for k,v in arms.items()}
    for f in sorted((aux/"heads").glob("*/posttraining.json")):
        d=json.loads(f.read_text());arms=d["arms"];parent=d["parent"]["occupancy"]
        extras.setdefault("decision_adaptation",{})[d["tag"]]=dict(parent=float(np.mean(parent)),oracle=float(np.mean(d["raw"]["occupancy"])),
              idle=float(np.mean(d["idle"]["occupancy"])),head_parameters=d["head_parameters"],labels_per_arm=d["labels_per_arm"],
              occupancy={k:float(np.mean(v["occupancy"])) for k,v in arms.items()},
              against_parent={k:effect(v["occupancy"],parent) for k,v in arms.items()},
              visited_minus_random=effect(arms["parent"]["occupancy"],arms["random"]["occupancy"]))
    document={"registered_stages_scored":len(all_rows),"two_disk_checkpoints":len(ledger),"rows":all_rows,"comparisons":comparisons,"followups":extras,
              "primary_unit":"held mission","bootstrap":"4000 paired replicates, percentile interval, training seeds separate"}
    if final:
        if len(ledger)!=54 or len(all_rows)!=92 or len(extras.get("history_window",{}))!=8 or len(extras.get("decision_adaptation",{}))!=4:
            raise RuntimeError(f"Incomplete frozen inventory: {len(ledger)}, {len(all_rows)}, {list(extras)}")
        for k,x in vectors.items():
            if "live/carry/occupancy" not in x or "room/w/w/aligned" not in x or "history/matched" not in x or (k.endswith("/test") and "contact/frozen" not in x):
                raise RuntimeError("Incomplete panel "+k)
        document["status"]="complete"
    else:document["status"]="interim"
    write(dest/"results.json",document);write(dest/"artifacts.json",ledger)
    if final:
        arrays={k+"|"+m:np.asarray(v,dtype=np.float32) for k,x in vectors.items() for m,v in x.items()}
        np.savez_compressed(dest/"paired-observations.npz",**arrays)
        for f in ("data-card.json","raw-oracle.json"):
            (dest/f).write_bytes((scores/f).read_bytes())
    return document


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("scores",type=Path);p.add_argument("dest",type=Path)
    p.add_argument("--final",action="store_true");a=p.parse_args();d=summarize(a.scores,a.dest,a.final);print(d["status"],d["two_disk_checkpoints"],d["registered_stages_scored"])
