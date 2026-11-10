"""Aggregate paired mission, recording and physical-scene effects while preserving every registered seed."""
import argparse
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
    controls={}
    def contrast(name,left,right):
        if left in vectors and right in vectors:controls[name]=effect(vectors[left],vectors[right])
    contrast("carry_minus_reset","live/carry/occupancy","live/reset/occupancy")
    contrast("carry_minus_wrong_action","live/carry/occupancy","live/wrong/occupancy")
    for kind in ("a","v","n","w"):
        prefix=f"{kind}/{kind}/"
        contrast(kind+"_training_prior_minus_aligned","prior/"+prefix+"aligned","room/"+prefix+"aligned")
        contrast(kind+"_shifted_minus_aligned","room/"+prefix+"shifted","room/"+prefix+"aligned")
    contrast("raw_force_rmse_minus_frozen","contact/raw","contact/frozen")
    contrast("constant_force_rmse_minus_frozen","contact/constant","contact/frozen")
    a["controls"]=controls
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
    for field in (128,512):
        for seed in (911,912):
            docs=[aux/"horizon"/f"confirm-{c}-z{field}-gelu-s{seed}-fresh.json" for c in ("hybrid","gpt2")]
            if all(p.exists() for p in docs):
                left,right=[json.loads(p.read_text())["arms"] for p in docs]
                extras.setdefault("history_architectures",{})[f"z{field}-s{seed}"]={k:effect(left[k]["occupancy"],right[k]["occupancy"]) for k in ("carry","full","block32")}
    for f in sorted((scores/"uncapped").glob("*.json")):
        u=json.loads(f.read_text());m=u["uncapped_rmse_newtons"]
        extras.setdefault("uncapped_contact",{})[u["tag"]]=dict(mean_rmse_newtons={k:float(np.mean(v)) for k,v in m.items()},
                raw_minus_frozen=effect(m["raw"],m["frozen"]),constant_minus_frozen=effect(m["constant"],m["frozen"]))
    native=scores/"native-parity.json"
    if native.exists():
        for tag,item in json.loads(native.read_text()).items():
            key=tag+"/4096/test";ref=vectors.get(key,{})
            if "live/carry/occupancy" in ref:
                if ledger[tag+"/4096"]["inference_sha256"]!=item["sha256"]:raise ValueError("Cross-backend parent mismatch")
                extras.setdefault("backend_check",{})[tag]=dict(tpu=float(np.mean(item["occupancy"])),cuda=float(np.mean(ref["live/carry/occupancy"])),
                     tpu_minus_cuda=effect(item["occupancy"],ref["live/carry/occupancy"]))
    timing={}
    for k,d in ledger.items():
        if k.rsplit("/",1)[1] not in ("384","4096"):continue
        path=primary/d["tag"]/"train.jsonl"
        logs=[json.loads(row) for row in path.read_text().splitlines() if row]
        warm=[v["warm_updates_per_s"] for v in logs if v.get("warm_updates_per_s") and v["step"]>=128]
        timing[d["tag"]]={"backend":d["backend"],"trainable_parameters":d["trainable_parameters"],"end_to_end_train_seconds":d["wall_this_process_s"],
             "median_warm_updates_per_second":float(np.median(warm)),"sampled_positions_per_parameter":d["step"]*128*32/d["trainable_parameters"]}
    for f in sorted((scores/"invariance").glob("*.json")):
        obj=json.loads(f.read_text());rows_similarity={}
        for direction,units in obj["scores"].items():
            direct=units["direct"];chance=np.full(len(direct),obj["chance"])
            rows_similarity[direction]=dict(direct=float(np.mean(direct)),train_aligned=float(np.mean(units["aligned_on_train"])),
                    raw_activity=float(np.mean(units["raw_activity"])),direct_minus_chance=effect(direct,chance),
                    train_alignment_gain=effect(units["aligned_on_train"],direct))
        extras.setdefault("source_retrieval",{})[obj["tag"]]=rows_similarity
    supplemental={}
    dose_root=primary.parent/"dose"
    if (dose_root/"scores.json").exists():
        study=json.loads((dose_root/"scores.json").read_text())
        if study!=json.loads((mirror.parent/"dose/scores.json").read_text()):raise ValueError("Independent-dose score copies disagree")
        for seed in (911,912):
            for c in ("hybrid","gpt2"):
                base=f"exposure-{c}-z512-gelu-s{seed}-";fresh,replay=study[base+"fresh"],study[base+"replay"]
                relative=Path(base+"replay")/"step-04096"
                rec=json.loads((dose_root/relative/"verified.json").read_text())
                if rec!=json.loads((mirror.parent/"dose"/relative/"verified.json").read_text()) or rec["inference_sha256"]!=replay["sha256"] or rec["optimizer_sha256"]!=replay["optimizer_sha256"]:
                    raise ValueError("Long replay artifact mismatch")
                if ledger.get(base+"fresh/4096",{}).get("inference_sha256")!=fresh["sha256"]:
                    if final:raise ValueError("Fresh long-dose parent missing")
                    continue
                supplemental[base+"replay/4096"]={**rec,"storage_directory":"dose"}
                extras.setdefault("independent_episode_dose",{})[f"{c}-s{seed}"]={mode:dict(fresh=float(np.mean(fresh[mode]["occupancy"])),
                    replay=float(np.mean(replay[mode]["occupancy"])),fresh_minus_replay=effect(fresh[mode]["occupancy"],replay[mode]["occupancy"])) for mode in ("carry","block32")}
    document={"registered_stages_scored":len(all_rows),"primary_two_disk_checkpoints":len(ledger),"two_disk_checkpoints":len(ledger)+len(supplemental),"rows":all_rows,"comparisons":comparisons,"followups":extras,
              "training_cost":timing,"primary_unit":"held mission","bootstrap":"4000 paired replicates, percentile interval, training seeds separate"}
    if final:
        from scripts.train.g9_fit import recipe_group,tag
        expected_ledger=set();expected_rows=set()
        for group in ("pilot","confirm","exposure"):
            for cell in recipe_group(group):
                steps=(384,) if group=="pilot" else (1024,4096)
                for step in steps:
                    name=tag(cell)+"/"+str(step);expected_ledger.add(name);expected_rows.add(name+"/dev")
                    if step==steps[-1]:expected_rows.add(name+"/test")
        if set(ledger)!=expected_ledger or set(all_rows)!=expected_rows or len(extras.get("history_window",{}))!=8 or len(extras.get("decision_adaptation",{}))!=4:
            raise RuntimeError(f"Incomplete frozen inventory: {len(ledger)}, {len(all_rows)}, {list(extras)}")
        for k,x in vectors.items():
            if "live/carry/occupancy" not in x or "room/w/w/aligned" not in x or "history/matched" not in x or (k.endswith("/test") and "contact/frozen" not in x):
                raise RuntimeError("Incomplete panel "+k)
        if len(extras.get("backend_check",{}))!=16 or len(extras.get("uncapped_contact",{}))!=8 or len(supplemental)!=4 or len(extras.get("source_retrieval",{}))!=8:
            raise RuntimeError("Native inference or force-range sensitivity panels are missing")
        document["status"]="complete"
    else:document["status"]="interim"
    write(dest/"results.json",document);write(dest/"artifacts.json",{**ledger,**supplemental})
    if final:
        arrays={k+"|"+m:np.asarray(v,dtype=np.float32) for k,x in vectors.items() for m,v in x.items()}
        np.savez_compressed(dest/"paired-observations.npz",**arrays)
        for f in ("data-card.json","raw-oracle.json"):
            (dest/f).write_bytes((scores/f).read_bytes())
    return document


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("scores",type=Path);p.add_argument("dest",type=Path)
    p.add_argument("--final",action="store_true");a=p.parse_args();d=summarize(a.scores,a.dest,a.final);print(d["status"],d["two_disk_checkpoints"],d["registered_stages_scored"])
