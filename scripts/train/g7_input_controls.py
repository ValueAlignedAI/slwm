"""Audit what the terminal Push-T forecasts depend on using the same held scenes, actions and weights."""
import argparse
from pathlib import Path
import json
import numpy as np
from scripts.train.g7_scale_lock import cases
from scripts.train.g7_readout_report import write


def main(root,out):
    lock=json.loads((root/"scale-test-lock.json").read_bytes());results={};common=None
    for name,step in cases():
        folder=root/"scale-eval"/name/f"step-{step:06d}"
        m=json.loads((folder/"test-metrics.json").read_bytes())
        if m["checkpoint"]!=lock["models"][name]:raise ValueError("Model outside confirmed test lock")
        with np.load(folder/"test-predictions.npz") as f:
            ids=f["sim_group"]
            if common is None:common=ids.copy()
            np.testing.assert_array_equal(ids,common)
            if len(np.unique(ids))!=len(ids):raise ValueError("Bootstrap requires independent simulator starts")
            draw=np.random.default_rng(77152).integers(len(ids),size=(2000,len(ids)))
            joint=f["push_joint"];rows={}
            for condition in ("joint","state","video","unplanned","persistence","velocity","ridge"):
                v=f["push_"+condition];diff=v-joint
                rows[condition]={"endpoint_px":float(v.mean()),"condition_minus_joint_px":float(diff.mean()),
                    "paired_scene_ci95_px":np.quantile(diff[draw].mean(1),[.025,.975]).tolist()}
            results[name]={"weights":m["checkpoint"],"conditions":rows}
    write(out,{"role":"512 independently held G7 simulator starts, all 16 terminal checkpoints and eight parents",
        "target":"Error in predicting final displacement under a proposed action; lower is better.",
        "inputs":{"joint":"past video, proprioception and prospective plan","state":"past proprioception and prospective plan","video":"past camera and prospective plan","unplanned":"past video and proprioception, prospective plan zeroed"},
        "uncertainty":"Paired whole-scene intervals on condition minus joint. Positive state-minus-joint favors adding the camera to proprioception.","models":results})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("out",type=Path);a=p.parse_args();main(a.root,a.out)
