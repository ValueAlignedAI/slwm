"""Publish coverage, agreement, and paired detector changes on blinded independent footage."""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import numpy as np
from evals.g13_stats import Groups
from scripts.train.g9_storage import write

ROOT=Path("/mnt/slwm-data/SLWM/g13")
CODES="0RBGY"

def join(index,ratings):
    if len(index)!=len(ratings):raise ValueError("Incomplete visual audit")
    grouped=defaultdict(list)
    for i,r in zip(index,ratings):
        if i["number"]!=r["number"] or r["review"] not in CODES+"U":raise ValueError("Blind key mismatch")
        grouped[i["episode"]].append(dict(**i,review=r["review"]))
    rows=[];repeats=Counter()
    for ls in grouped.values():
        labels={x["review"] for x in ls}
        if len(ls)>1:repeats["total"]+=1;repeats["agree"]+=len(labels)==1
        rows.append(dict(**ls[0],final=ls[0]["review"] if len(labels)==1 else "U",presentations=[r["number"] for r in ls]))
    return rows,repeats

def accuracy(rows,key):
    known=[r for r in rows if r["final"]!="U"]
    paired=[r for r in known if r[key]>=0]
    return dict(cases=len(rows),review_readable=len(known),detector_accepted=sum(r[key]>=0 for r in rows),
                both_readable=len(paired),correct=sum(CODES[r[key]]==r["final"] for r in paired),
                confusion=dict(Counter(f"{CODES[r[key]]}/{r['final']}" for r in paired)))

def main(public):
    def a(p):return json.loads(p.read_bytes())
    first,repeats=join(a(ROOT/"data/robot/audit/index.json"),a(public/"audit/ratings.json"))
    second,r2=join(a(ROOT/"data/robot-extra/audit/index.json"),a(public/"audit/external-ratings.json"))
    original={}
    for role in ("dev","test"):
        rows=[r for r in first if r["role"]==role]
        original[role]=accuracy(rows,"detector")
    panels={}
    for panel in ("uniform","blue"):
        rows=[r for r in second if r["panel"]==panel];bs=Groups([r["group"] for r in rows])
        def utility(key):
            return np.array([np.nan if r["final"]=="U" else 0 if r[key]<0 else 1 if CODES[r[key]]==r["final"] else -1 for r in rows])
        panels[panel]=dict(original=accuracy(rows,"old"),corrected=accuracy(rows,"new"),utility_gain=bs.score(utility("new")-utility("old")))
    dest=ROOT/"release/metrics";dest.mkdir(exist_ok=True,parents=True)
    write(dest/"measurement.json",dict(original=original,original_repeats=repeats,external=panels,external_repeats=r2,first=first,fresh=second,
         reviewer="one Codex visual reviewer, blinded to labels, modality cues and predictions"))
    for panel,r in panels.items():print(panel,[(k,r[k]["correct"],r[k]["both_readable"],r[k]["detector_accepted"]) for k in ("original","corrected")],r["utility_gain"],flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("public",type=Path);main(p.parse_args().public)
