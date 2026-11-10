"""Advance matched G5C stages once, retain both physical copies, and open the sealed split once."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from training.g4_train import write
from training.g5_compare import compare


def file_sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def mirror(run,disk):
    if run.stat().st_dev==disk.stat().st_dev:
        raise RuntimeError("Physical checkpoint devices must be independent")
    other=disk/run.name
    other.mkdir(exist_ok=True)
    hashes=json.loads((run/"complete.json").read_bytes())["sha256"]
    keys={**hashes,"complete.json":file_sha(run/"complete.json")}
    for key,expected in keys.items():
        src=run/key
        if file_sha(src)!=expected:
            raise ValueError(f"Invalid completed payload {src}")
        dst=other/key
        if not dst.exists() or file_sha(dst)!=expected:
            temp=dst.with_suffix(".mirror-tmp")
            shutil.copyfile(src,temp)
            if file_sha(temp)!=expected:
                raise ValueError("Mirror transfer changed bytes")
            temp.replace(dst)
    write(run/"disk-receipt.json",{"run":run.name,"sha256":keys,"independent_filesystems":True})
    return keys


def better(result):
    scores={}
    for key in ("hybrid","gpt2"):
        action=[]; impact=[]
        for row in result["seeds"].values():
            x=row["controls"][key]
            action.append(1-x["push"]["joint"]["endpoint_px"]/x["push"]["ridge"]["endpoint_px"])
            impact.append(x["impact"]["correct"]/x["impact"]["raw_motion"]-1)
        scores[key]=[float(np.mean(action)),float(np.mean(impact))]
    return any(min(v)>=.01 for v in scores.values()),scores


def main(a):
    root,disk=a.root,a.mirror
    label="residual-gpu"
    tag=lambda scale,core,seed:f"g5-{scale}-{core}-{seed}-{label}"
    root.mkdir(exist_ok=True); disk.mkdir(exist_ok=True)
    def gpu(args,log):
        with log.open("ab") as stream:
            return subprocess.run(["flock","/tmp/slwm_research_gpu.lock","bash","scripts/train/g4_cuda.sh",*map(str,args)],
                                  stdout=stream,stderr=subprocess.STDOUT)
    def wait_for(paths,timeout=5400):
        until=time.monotonic()+timeout
        while not all(p.is_file() for p in paths):
            if time.monotonic()>until:
                raise TimeoutError(f"Expected {paths}")
            time.sleep(12)
    for scale in ("medium","anchor"):
        runs=[root/tag(scale,c,s) for s in (511,512) for c in ("hybrid","gpt2")]
        if scale=="anchor":
            for s in (511,512):
                for core in ("hybrid","gpt2"):
                    run=root/tag(scale,core,s)
                    if not (run/"complete.json").is_file():
                        for micro in (128,64,32):
                            log=root/(run.name+f"-micro{micro}.log")
                            proc=gpu(["-m","training.g5_residual_fit","--backend","gpu","--data",a.data,"--output",root,
                                      "--core",core,"--seed",s,"--scale","anchor","--steps",8192,"--label",label,
                                      "--micro",micro,"--save-every",1024],log)
                            if proc.returncode==0:
                                break
                            msg=log.read_bytes()[-20000:].lower()
                            if b"out of memory" not in msg and b"resource_exhausted" not in msg:
                                raise RuntimeError(f"Training failed: {log}")
                        else:
                            raise RuntimeError("All registered microbatches exceeded device memory")
                    if not (run/"dev-metrics.json").is_file():
                        gpu(["-m","training.g5_eval","--backend","gpu","--data",a.data,"--role","dev","--run",run,"--out",run],
                            root/(run.name+"-dev.log")).check_returncode()
        wait_for([r/"dev-metrics.json" for r in runs])
        dev=compare(root,scale,label,"dev")
        advance,gains=better(dev)
        write(root/(scale+"-residual-dev.json"),dev)
        write(root/(scale+"-residual-decision.json"),{"advance_to_anchor":scale=="medium" and advance,
              "relative_to_physical_priors":gains,"matched_sha256":dev["sha256"]})
        print(scale,"1% promotion gate",advance,gains,flush=True)
        for r in runs:
            mirror(r,disk)
        if scale=="medium" and advance:
            continue
        wait_for([root/"integrity.json"])
        lock=root/"terminal-held-lock.json"
        manifest={"promote":True,"meaning":"terminal test admissible; both physical copies verified","scale":scale,"sha256":dev["sha256"]}
        if lock.exists() and json.loads(lock.read_bytes())!=manifest:
            raise ValueError("A different terminal test lock exists")
        write(lock,manifest)
        for run in runs:
            if not (run/"test-metrics.json").is_file():
                gpu(["-m","training.g5_eval","--backend","gpu","--role","test","--run",run,"--out",run,
                     "--data",a.data,"--lock",lock],root/(run.name+"-test.log")).check_returncode()
        summary=compare(root,scale,label,"test")
        write(root/"terminal-test.json",summary)
        print("Terminal test complete",scale,"joint win",summary["joint_architecture_win"],flush=True)
        break


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True,type=Path); p.add_argument("--mirror",required=True,type=Path)
    p.add_argument("--data",required=True,type=Path)
    main(p.parse_args())
