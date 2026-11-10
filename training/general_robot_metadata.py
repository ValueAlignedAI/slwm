"""Join official raw robot metadata to the100 RLDS records before model fitting."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request
import numpy as np

from training.general_data import ROOT,CACHE,order
from training.general_sources import write
from training.research_sources import fetch,file_sha


def lookup(prefix):
    url="https://storage.googleapis.com/storage/v1/b/gresearch/o?"+urllib.parse.urlencode({"prefix":prefix,"maxResults":30})
    with urllib.request.urlopen(url,timeout=40) as response: return json.load(response)


def verify_object(item,path):
    url="https://storage.googleapis.com/gresearch/"+urllib.parse.quote(item["name"],safe="/")+"?generation="+item["generation"]
    result=fetch(url,path); assert result["bytes"]==int(item["size"])
    assert base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode()==item["md5Hash"]
    write(path.with_suffix(path.suffix+".verified.json"),{"object":item,"source":result,"verified":True})
    return result


def acquire(row,family="droid"):
    match=re.search(r"/([^/]+)/(success|failure)/(\d{4}-\d{2}-\d{2})/([^/]+)/trajectory.h5$",row["raw_path"])
    if not match: raise ValueError("Unknown raw episode path")
    suffix="/".join(match.groups())+"/"; dest=ROOT/family/"raw_metadata"/row["id"]; dest.mkdir(parents=True,exist_ok=True); attempts=[]
    for version in ("1.0.1","1.0.0"):
        prefix="robotics/droid_raw/"+version+"/"+suffix; listing=lookup(prefix+"metadata_"); write(dest/(version+"_listing.json"),listing)
        candidates=[i for i in listing.get("items",[]) if i["name"].endswith(".json")]; attempts.append({"version":version,"matches":len(candidates)})
        if not candidates: continue
        if len(candidates)!=1: raise ValueError("Ambiguous metadata candidate")
        item=candidates[0]; path=dest/"metadata.json"; verify_object(item,path); metadata=json.loads(path.read_text())
        assert metadata["lab"]==row["collector"]
        result={"id":row["id"],"building":str(metadata["building"]),"scene_id":str(metadata["scene_id"]),"collector_hash":hashlib.sha256((metadata["lab"]+":"+metadata["user_id"]).encode()).hexdigest(),"lab":metadata["lab"],"source_metadata_sha256":file_sha(path),"source_version":version,"raw_trajectory_length":int(metadata["trajectory_length"]),"rlds_steps":row["steps"],"attempts":attempts}
        # Small low-dimensional trajectories can supply clock evidence without
        # downloading the raw high-resolution/stereo camera recordings.
        trajectory=lookup(prefix+"trajectory.h5").get("items",[])
        if len(trajectory)==1 and int(trajectory[0]["size"])<=16*1024**2:
            p=dest/"trajectory.h5"; verify_object(trajectory[0],p); result["trajectory_sha256"]=file_sha(p)
        else: result["trajectory_unavailable_or_over16MiB"]=[{"name":v["name"],"bytes":v["size"]} for v in trajectory]
        return result
    return {"id":row["id"],"metadata_unavailable":True,"attempts":attempts}


def connected_roles(rows,metadata,family="droid"):
    available={v["id"]:v for v in metadata if not v.get("metadata_unavailable") and not v.get("error")}; remaining=set(available); groups={}
    while remaining:
        connected={min(remaining)}; previous=-1
        while len(connected)!=previous:
            previous=len(connected); buildings={available[k]["building"] for k in connected}; collectors={available[k]["collector_hash"] for k in connected}
            connected|={k for k in remaining if available[k]["building"] in buildings or available[k]["collector_hash"] in collectors}
        remaining-=connected; group="DROID-ENV-"+order(":".join(sorted(connected)))[:20]
        for key in connected: groups[key]=group
    names=sorted(set(groups.values()),key=order)
    if len(names)<12: return {"suitable_for_building_holdout":False,"connected_groups":len(names),"available":len(available),"reason":"Too few verified building/collector components for4-way roles with >=5 final groups"}
    ntest=max(5,int(np.ceil(len(names)*.2))); ncal=max(2,int(np.ceil(len(names)*.1))); nselection=max(2,int(np.ceil(len(names)*.15)))
    roles={g:"test" if i>=len(names)-ntest else "calibration" if i>=len(names)-ntest-ncal else "selection" if i>=len(names)-ntest-ncal-nselection else "train" for i,g in enumerate(names)}
    selected=[]
    for row in rows:
        if row["id"] not in available: continue
        m=available[row["id"]]; selected.append({**row,"group":groups[row["id"]],"role":roles[groups[row["id"]]],"building":m["building"],"scene_id":m["scene_id"],"collector_hash":m["collector_hash"],"environment_id_verified":True})
    for one in ("train","selection","calibration","test"):
        for two in ("train","selection","calibration","test"):
            if one>=two: continue
            for field in ("building","collector_hash"):
                assert not {r[field] for r in selected if r["role"]==one}&{r[field] for r in selected if r["role"]==two}
    return {"suitable_for_building_holdout":True,"rows":selected,"connected_groups":len(names),"available":len(available),"excluded_unverified_ids":sorted({r["id"] for r in rows}-available.keys()),"claim":"Building and collector disjoint; bounded exploratory source only. No closed-loop physical success.","metadata_manifest_sha256":file_sha(ROOT/family/"raw_metadata/results.json"),"group_counts":{role:sum(v==role for v in roles.values()) for role in ("train","selection","calibration","test")}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--family",choices=("droid","droid_expanded"),default="droid");parser.add_argument("--roles-only",action="store_true");args=parser.parse_args();family=args.family
    folder=ROOT/family/"raw_metadata"; folder.mkdir(parents=True,exist_ok=True); rows=json.loads((CACHE/family/"rows.json").read_text()); results=[]
    if args.roles_only:
        results=json.loads((folder/"results.json").read_text());result=connected_roles(rows,results,family);result["source_sha256"]=file_sha(__file__);write(folder/"scene_role_plan.json",result);return
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(acquire,row,family):row for row in rows}
        for future in as_completed(futures):
            row=futures[future]
            try: value=future.result()
            except Exception as exc: value={"id":row["id"],"error":str(exc),"error_type":type(exc).__name__}
            results.append(value); write(folder/"results.json",results)
            print(json.dumps({"robot_metadata":len(results),"unavailable_or_failed":sum(bool(r.get("error") or r.get("metadata_unavailable")) for r in results)}),flush=True)
    results.sort(key=lambda r:r["id"]);write(folder/"results.json",results)
    result=connected_roles(rows,results,family); result["source_sha256"]=file_sha(__file__); write(folder/"scene_role_plan.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="rows"}),flush=True)


if __name__=="__main__": main()
