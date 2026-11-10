"""Collect held controls and hash pins for the waveform, consecutive-decision and camera experiments."""
import argparse
import json
from pathlib import Path
import numpy as np


def read(path):return json.loads(path.read_bytes())
def write(path,value):
    tmp=path.with_suffix(".partial");tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n");tmp.replace(path)


def collect(root,data,out):
    out.mkdir(parents=True,exist_ok=True)
    policies={}
    for family in ("policy","policy-prior"):
        models={}
        for file in sorted((root/family).glob("*/dev.json")):
            d=read(file);rs=d["results"]
            models[file.parent.name]={"regret_cm":{k:v["regret_cm"] for k,v in rs.items() if "regret_cm" in v},
                "top1":rs["correct"]["top1"],"parameters":d["params"],"gate":rs["gate"],"core_sha256":d["parent"],
                "weights":d["weights_sha256"],"optimizer":d["optimizer_sha256"],"foundation_metrics_before":d["foundation_held_before"],
                "foundation_metrics_after":d["foundation_held_after"]}
        policies[family]=models
    policies["dynamics_prior"]=read(root/"reacher-prior/card.json")
    policies["promoted"]=read(root/"policy-prior/sft-lock.json")
    independent={}
    for file in sorted((root/"supervised-confirmation").glob("policy-*.json")):
        d=read(file);rs=d["results"];independent[file.stem]={"regret_cm":{k:v["regret_cm"] for k,v in rs.items() if "regret_cm" in v},
            "top1":rs["correct"]["top1"],"gate":rs["gate"],"weights_sha256":d["weights_sha256"],"fresh_starts":d["fresh_starts"]}
    policies["independent_test"]=independent
    write(out/"supervised.json",policies)
    rl={}
    for path in sorted((root/"rl").glob("*/test.json")):
        d=read(path);rl[path.parent.name]={"decisions":d["real_training_decisions"],"ppo_steps":d["ppo_updates"],"passed":d["replication_gate_this_seed"],
            "contrasts":d["success_contrasts"],"final":{k:v["mean"] for k,v in d["results"].items()},
            "long":d["long_192_cycle_diagnostic"],"model":d["checkpoint_sha256"],"optimizer":d["optimizer_sha256"]}
    if (root/"reacher-oracle.json").exists():rl["physical_oracle_post_hoc"]=read(root/"reacher-oracle.json")["summary"]
    write(out/"closed-loop.json",rl)
    speech={};align={}
    for f in ("crema","libri"):
        path=root/"alignment-audit"/(f+"-oracle.json")
        if not path.is_file():continue
        d=read(path);speech[f]={"clips":len(d["ids"]),"people":len(set(d["speakers"])),"renderer":d["renderer"],
            "recognizer_revision":d["recognizer_revision"],"conditions":{k:{n:v for n,v in s.items() if n not in ("hypotheses","word_edits")} for k,s in d["conditions"].items()}}
    for path in sorted((root/"alignment-audit").glob("*/*/waves.json")):
        d=read(path);tag=d["tag"]+"/"+d["variant"];all_scores=read(path.parent/"field.json")
        align[tag]={"sha256":d["head_sha256"],"groups":all_scores["groups"],"fields":all_scores["scores"],"waveforms":{f:{k:{n:v for n,v in s.items() if n not in ("hypotheses","word_edits")} for k,s in arr["conditions"].items()} for f,arr in d["scores"].items()}}
    write(out/"alignment.json",{"oracles":speech,"matched_heads":align,"renderer":read(root/"unit-renderer/complete.json"),
        "codebook":read(data/"units/card.json"),"test_lock":read(root/"alignment/test-lock.json")})
    camera={}
    for seed in (711,712):
        rows={}
        vals={}
        for core in ("raw","hybrid","gpt2"):
            p=root/"camera"/f"{core}-{seed}"
            if not (p/"test.json").is_file():continue
            rows[core]=read(p/"test.json")
            with np.load(p/"scenes.npz") as f:vals[core]={k:np.array(v) for k,v in f.items()}
        if "hybrid" in vals and "gpt2" in vals:
            from scripts.train.g6_camera_regret import bootstrap_rows,interval
            h,d=vals["hybrid"],vals["gpt2"]
            np.testing.assert_array_equal(h["donor"],d["donor"]);np.testing.assert_array_equal(h["scene"],d["scene"])
            draw=bootstrap_rows(h["donor"],8)
            contrast=(h["wrong"]-h["correct"])-(d["wrong"]-d["correct"])
            rows["wrong_by_architecture"]={"hybrid_excess_penalty_cm":float(contrast.mean()),"scene_block_ci95_cm":interval(contrast,draw),
                 "clean_dense_minus_hybrid_cm":float((d["correct"]-h["correct"]).mean()),"clean_ci95":interval(d["correct"]-h["correct"],draw)}
        camera[str(seed)]=rows
    write(out/"camera.json",camera)
    cards={r:read(data/"packets"/(r+".json")) for r in ("train","dev","test")}
    cards["sha256"]={r:read(data/"packets"/(r+".verified.json")) for r in ("train","dev","test")}
    if (data/"exposure.json").is_file():cards["complete_exposure"]=read(data/"exposure.json")
    write(out/"data.json",cards)
    print("Supervised",sum(len(policies[f]) for f in ("policy","policy-prior")),"RL",len(rl),"heads",len(align),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("root","data","out"):p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args();collect(a.root,a.data,a.out)
