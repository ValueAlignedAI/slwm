"""Replay both G3 input schedules without running a model or reading confirmation outcomes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

from training.joint_data import JointData, read
from training.joint_sources import CONFIG, digest
from training.joint_train import exposure_digest, seed_all
from training.joint_variant_config import read_variant_config


def replay(cfg, seed, steps):
    seed_all(seed)
    data = JointData(cfg,seed)
    chain = "00"*32
    calls = Counter()
    trace = []
    for step in range(steps):
        family,batch,ids = data.get(step)
        chain = exposure_digest(chain,family,ids,batch)
        calls[family] += 1
        trace.append(family)
    return chain,dict(calls),trace


def audit(physical):
    primary = read(CONFIG)
    if not physical.samefile(Path(primary["output"])):
        raise ValueError("Expected the actual primary study output")
    variants = {g:read(physical/g/"config.json") for g in ("depth6","structured")}
    if any(list(c["families"])!=sorted(primary["families"]) or c["families"]!=primary["families"] for c in variants.values()):
        raise ValueError("Stored schedules do not reproduce the reported failure")
    source = variants["depth6"]
    for name in ("data","batch","text_length","signal_steps"):
        if any(c[name]!=primary[name] for c in variants.values()): raise ValueError("Other sampler inputs changed")
    result = {"registered":list(primary["families"]),"stored":list(source["families"]),"seeds":{},
              "primary_lock_sha256":digest(physical/"training_lock.json"),"variant_locks":{},"rejected":{},"parents":{}}
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/"corrected.json"
        path.write_text(json.dumps({**source,"training_family_order":list(primary["families"])},sort_keys=True))
        corrected=read_variant_config(path,primary)
        for seed in primary["seeds"]:
            good,calls,order=replay(primary,seed,100)
            bad,other,bad_order=replay(source,seed,100)
            repaired,repair_calls,_=replay(corrected,seed,100)
            originals=[read(physical/f"{p}-{seed}"/"step-100.json") for p in primary["profiles"]]
            if good!=repaired or good==bad or calls!=other or calls!=repair_calls or not all(c["exposure_sha256"]==good for c in originals):
                raise ValueError("Failed to reproduce the primary and corrected tensor hashes")
            result["seeds"][seed]={"primary_and_repaired_sha256":good,"rejected_sha256":bad,"call_counts":calls,
                                   "first_different_source_step":next(i for i,(a,b) in enumerate(zip(order,bad_order)) if a!=b)}
    for group,cfg in variants.items():
        base=physical/group; result["variant_locks"][group]=digest(base/"training_lock.json")
        result["rejected"][group]={}; result["parents"][group]={}
        if (base/"confirmation_lock.json").exists() or list(base.glob("*/step-1000.json")) or list(base.glob("*/step-8000.json")):
            raise ValueError("Unexpected later variant stage")
        for profile in cfg["profiles"]:
            for seed in cfg["seeds"]:
                name=f"{profile}-{seed}"; folder=base/name; card=read(folder/"step-100.json")
                if card["exposure_sha256"]!=result["seeds"][seed]["rejected_sha256"] or digest(folder/"step-100.pt")!=card["delta_sha256"]:
                    raise ValueError(f"Unexpected rejected state at {group}/{name}")
                result["rejected"][group][name]={"delta_sha256":card["delta_sha256"],"resume_sha256":digest(folder/"resume.pt")}
        parent_root=Path(primary["data"])/("depth_parents" if group=="depth6" else "structured_parents")
        for key,(name,expected) in cfg["parents"].items():
            parent=parent_root/name/"best.pt"
            if digest(parent)!=expected: raise ValueError(f"Parent mismatch at {group}/{key}")
            result["parents"][group][key]={"name":name,"sha256":expected,"bytes":parent.stat().st_size}
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--physical",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    proof=audit(args.physical)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    tmp=args.out.with_suffix(".tmp"); tmp.write_text(json.dumps(proof,sort_keys=True,indent=2)+"\n"); tmp.replace(args.out)
    print("reproduced both rejected and corrected tensor chains and verified six parent files",flush=True)
