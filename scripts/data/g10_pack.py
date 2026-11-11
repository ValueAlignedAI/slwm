"""Materialize verified training or evaluation banks without mixing recording roles."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def main(root,role):
    selected=json.loads((root/"banks.json").read_text())[role]
    out=root/("holo-"+role+".npz")
    if out.exists() and out.with_suffix(".json").exists():return
    chunks={k:[] for k in ("a","v","n","w","x")};lengths=[];names=[];events=[];tasks=[]
    for row in selected:
        path=root/"shards"/(row["name"]+".npz")
        with path.open("rb") as f:
            if hashlib.file_digest(f,"sha256").hexdigest()!=row["sha256"]:raise ValueError("Immutable recording bank changed")
        with np.load(path,allow_pickle=False) as data:
            for k in chunks:chunks[k].append(data[k])
            lengths.append(len(data["a"]));names.append(row["name"]);events.append(data["events"]);tasks.append(row["task"])
    temp=out.with_suffix(".new.npz");np.savez_compressed(temp,**{k:np.concatenate(v) for k,v in chunks.items()},names=np.array(names),lengths=np.array(lengths,np.int32),tasks=np.array(tasks),events=np.concatenate(events),event_lengths=np.array([len(v) for v in events],np.int32))
    temp.replace(out)
    with out.open("rb") as f:sha=hashlib.file_digest(f,"sha256").hexdigest()
    out.with_suffix(".json").write_text(json.dumps(dict(role=role,sha256=sha,ticks=sum(lengths),recordings=len(names),bytes=out.stat().st_size))+"\n")
    print(role,len(names),sum(lengths),out.stat().st_size,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("role",choices=("train","dev","familiar","novel"));a=p.parse_args();main(a.root,a.role)
