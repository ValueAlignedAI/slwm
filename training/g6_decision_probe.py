"""Evaluate terminal same-exposure checkpoints on fresh long-horizon scene goals."""
import argparse
import json
from pathlib import Path

import numpy as np

from training.g4_train import write
from training.g5_planning import experiment


def evaluate(a):
    models={c:a.root/f"g6-{a.generation}-{a.scale}-{c}-{a.seed}"/"step-008192" for c in ("hybrid","gpt2")}
    cards=[json.loads((p/"config.json").read_bytes()) for p in models.values()]
    for k in ("params","seed","batch","data","lr","steps","code"):
        if cards[0][k]!=cards[1][k]:
            raise ValueError("Unmatched terminal decision pair")
    hashes={}
    for core,run in models.items():
        receipt=json.loads((run/"verified.json").read_bytes())
        if receipt["step"]!=8192 or len(set(receipt["disk_uuids"]))!=2:
            raise ValueError("Unverified terminal checkpoint")
        hashes[core]=receipt["sha256"]["inference.npz"]
    out=a.output
    out.parent.mkdir(exist_ok=True,parents=True)
    temp=out.with_suffix(".partial.json")
    experiment(a.root,a.data,temp,a.sim_site,runs=models,first_seed=402000)
    result=json.loads(temp.read_bytes())
    if not result["complete"] or result["episodes"]!=32:
        raise ValueError("Incomplete policy comparison")
    c=result["controls"]
    rng=np.random.default_rng(61402); indices=rng.integers(0,32,(4000,32))
    effects={}
    for first,second in (("hybrid","gpt2"),("hybrid","prior"),("hybrid","passive"),("hybrid-reset","hybrid"),
                         ("gpt2-reset","gpt2"),("hybrid-reset","gpt2-reset")):
        baseline=np.array(c[second]["final_distances_px"]); delta=baseline-np.array(c[first]["final_distances_px"])
        effects[first+"-over-"+second]={"mean_reduction_px":float(delta.mean()),"relative":float(delta.mean()/baseline.mean()),
                                      "paired_ci95_px":np.quantile(delta[indices].mean(1),[.025,.975]).tolist()}
    result.update(scale=a.scale,seed=a.seed,generation=a.generation,params=cards[0]["params"],weights_sha256=hashes,
                  paired_effects=effects,passes_at_this_seed=all(effects[k]["relative"]>=.05 and effects[k]["paired_ci95_px"][0]>0
                      for k in ("hybrid-over-gpt2","hybrid-over-prior")))
    write(out,result); temp.unlink(missing_ok=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--root",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True); p.add_argument("--sim-site",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--scale",choices=["larger","anchor"],required=True)
    p.add_argument("--generation",choices=["v5e","v6e"],required=True); p.add_argument("--seed",type=int,required=True)
    evaluate(p.parse_args())
