"""Quantify instruction retention and the explicit causal cache on independent cohorts."""
import argparse
import json
from pathlib import Path
import numpy as np
from g10_analysis import CORES,SEEDS,effect,custody,read
from g9_storage import write

def analyze(root,dest):
    src=root/"commands";gate=read(src/"gate.json");document={"transfer_gate":gate,"parent":{},"adapted":{},"paired":{},"latch":{}}
    ledger={};paired={}
    for c in CORES:
        for s in SEEDS:
            tag=f"context-{c}-variable-s{s}";p=read(src/(tag+"-parent.json"));a=p["scores"]
            if p["sha256"]!=custody(tag,5120)["inference_sha256"]:raise ValueError("Unverified command parent")
            document["parent"][tag]=dict(occupancy={k:float(np.mean(v["occupancy"])) for k,v in a.items()},brief_minus_continuous=effect(a["brief"]["occupancy"],a["continuous"]["occupancy"]))
            l=read(root/"latched"/(tag+".json"))
            if not l["exact_control_trace"] or l["sha256"]!=p["sha256"]:raise ValueError("Causal replay did not reproduce held decisions")
            document["latch"][tag]=dict(exact=True,worlds=l["worlds"],steps=l["decisions_per_world"],bytes_per_word_record=l["causal_word_slots"],
                occupancy={k:float(np.mean(v)) for k,v in l["occupancy"].items()})
            for key,value in a.items():paired[tag+"|"+key]=np.asarray(value["occupancy"],"f4")
    if gate["passed"]:
        for core in ("hybrid","gpt2"):
            for s in SEEDS:
                cases={}
                for mode in ("continuous","brief"):
                    tag=f"cue-{core}-{mode}-s{s}";doc=read(src/(tag+".json"));stage=custody(tag,768);ledger[tag+"/768"]=stage
                    if doc["sha256"]!=stage["inference_sha256"]:raise ValueError("Unverified follow-up weights")
                    cases[mode]=doc["scores"]
                    document["adapted"][tag]={k:float(np.mean(v["occupancy"])) for k,v in doc["scores"].items()}
                    for key,value in doc["scores"].items():paired[tag+"|"+key]=np.asarray(value["occupancy"],"f4")
                document["paired"][f"{core}-s{s}"]={k:effect(cases["brief"][k]["occupancy"],cases["continuous"][k]["occupancy"]) for k in ("continuous","brief","blank")}
        ref=read(src/"physical-reference.json");document["physical"]={k:float(np.mean(v["occupancy"])) for k,v in ref.items()}
    dest.mkdir(exist_ok=True,parents=True);write(dest/"instructions.json",document);write(dest/"instruction-artifacts.json",ledger)
    np.savez_compressed(dest/"instruction-paired.npz",**paired);return document

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("eval",type=Path);p.add_argument("dest",type=Path);a=p.parse_args();print("instruction cases",len(analyze(a.eval,a.dest)["adapted"]))
