"""Freeze all eight transfer weights after the two-seed gate, then open only the independent GRID role."""
import argparse
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time


def verify(root,data,deadline):
    out=root/"speaker-confirmation"; out.mkdir(exist_ok=True)
    lock=out/"lock.json"; selected={}
    while time.time()<deadline:
        selected={}
        for proof in root.glob("g6-*/step-002048/verified.json"):
            record=json.loads(proof.read_bytes()); tag=record["run"]
            other=Path("/mnt/slwm-checkpoints-b/SLWM/g6")/tag/"step-002048/verified.json"
            if not other.exists() or other.read_bytes()!=proof.read_bytes(): continue
            readout=root/"shared-probes"/(tag+".json")
            if not readout.is_file(): continue
            dev=json.loads(readout.read_bytes())
            if record["sha256"]["inference.npz"]!=dev["inference_sha256"]: raise ValueError("Mismatched development checkpoint")
            _,gen,size,core,seed=tag.split("-")
            key=(size,core,int(seed))
            if key in selected: raise ValueError("Ambiguous hardware replication")
            selected[key]=(tag,dev)
        if len(selected)==8: break
        time.sleep(45)
    if len(selected)!=8: return
    successes=[]
    for size in ("larger","anchor"):
        qualified=True
        for seed in (511,512):
            h=selected[size,"hybrid",seed][1]["shots"]["8"]; d=selected[size,"gpt2",seed][1]["shots"]["8"]
            delta=[h["cross_view_by_speaker"][g]-d["cross_view_by_speaker"][g] for g in sorted(h["cross_view_by_speaker"])]
            rng=random.Random(602048); boot=sorted(statistics.mean(rng.choices(delta,k=len(delta))) for _ in range(4000))
            qualified &= h["mean_cross_view"]>.25 and h["mean_cross_view"]-d["mean_cross_view"]>=.03 and boot[99]>0
        if qualified: successes.append(size)
    if not successes: raise RuntimeError("The registered development gate was unmet")
    manifest={"step":2048,"qualified_development_scales":successes,"confirmation_people":[25,26,27,28,29,30,31,32],"sentence_bucket":9,
              "sha256":{tag:r["inference_sha256"] for tag,r in selected.values()}}
    if lock.is_file() and json.loads(lock.read_bytes())!=manifest: raise ValueError("A different confirmation was frozen")
    if not lock.exists():
        tmp=lock.with_suffix(".pending"); tmp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n"); tmp.replace(lock)
    for key in sorted(selected,key=lambda k:(k[0]!="larger",k[2],k[1])):
        tag,_=selected[key]; result=out/(tag+".json")
        if result.is_file(): continue
        with (out/(tag+".log")).open("ab") as log:
            subprocess.run([sys.executable,"-u","-m","training.g6_state_transfer","--data",str(data),"--run",str(root/tag/"step-002048"),
                            "--out",str(result),"--speaker-lock",str(lock)],check=True,stdout=log,stderr=subprocess.STDOUT)
        print("Frozen speaker confirmation",tag,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--root",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True); p.add_argument("--deadline",type=float,required=True)
    a=p.parse_args(); verify(a.root,a.data,a.deadline)
