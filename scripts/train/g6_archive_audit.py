"""Independently rehash every registered Colab milestone from each physical Fedora disk."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

from scripts.train.g6_artifacts import roots


def check_disk(root,cases):
    counts=0; payload=0; configs={}; summaries={}
    for relative in cases:
        stage=root/relative
        record=json.loads((stage/"verified.json").read_bytes())
        complete=json.loads((stage/"complete.json").read_bytes())
        config=json.loads((stage/"config.json").read_bytes())
        expected=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        if complete["identity"]!=expected or config["run"]!=record["run"] or complete["step"]!=record["step"]:
            raise ValueError(f"Model/config mismatch at {relative}")
        configs[record["run"]]=config["params"]
        for k,digest in record["sha256"].items():
            p=stage/k
            with p.open("rb") as f:
                actual=hashlib.file_digest(f,"sha256").hexdigest()
            if actual!=digest: raise ValueError(f"Physical read-back failed at {p}")
            counts+=1; payload+=p.stat().st_size
        summaries[relative]={"run":record["run"],"step":record["step"],"inference_sha256":record["sha256"]["inference.npz"],
                              "optimizer_sha256":record["sha256"]["resume.npz"],"inference_bytes":record["inference_bytes"],
                              "optimizer_bytes":record["optimizer_bytes"]}
    return {"files":counts,"bytes":payload,"models":configs,"milestones":summaries}


def audit(out,deadline):
    a,b=roots()
    cfg=json.loads(Path("configs/t6/tpu_scale.json").read_bytes())
    while time.time()<deadline:
        by_seed={}
        for seed in cfg["seeds"]:
            for gen in ("v5e","v6e"):
                names=[f"g6-{gen}-{scale}-{core}-{seed}/step-{step:06d}" for scale,step in cfg["stages"] for core in cfg["cores"]]
                if all((a/name/"verified.json").is_file() and (b/name/"verified.json").is_file() for name in names):
                    if seed in by_seed: raise ValueError("Ambiguous replication")
                    by_seed[seed]=names
        if len(by_seed)==2:break
        time.sleep(12)
    else: raise TimeoutError("Not all registered same-chip milestones are present")
    cases=[p for group in by_seed.values() for p in group]; start=time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as executor:
        one,two=[future.result() for future in [executor.submit(check_disk,r,cases) for r in (a,b)]]
    if one!=two: raise ValueError("Independent physical disks did not return identical artifacts")
    doc={"study":"G6","physical_replica_count":2,"physical_root_paths":[str(a),str(b)],"files_per_replica":one["files"],
         "total_verified_bytes_per_replica":one["bytes"],"unique_models":one["models"],"milestone_count":len(cases),
         "wall_seconds_read_back":round(time.monotonic()-start,2),"milestones":one["milestones"]}
    tmp=out.with_suffix(".pending"); tmp.write_text(json.dumps(doc,sort_keys=True,indent=2)+"\n"); tmp.replace(out)
    print("Verified both disks:",len(cases),"stages",one["files"],"files per disk",one["bytes"],"bytes per disk",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--out",type=Path,required=True); p.add_argument("--deadline",type=float,required=True)
    a=p.parse_args();audit(a.out,a.deadline)
