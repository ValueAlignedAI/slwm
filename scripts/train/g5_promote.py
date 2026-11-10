"""Verify immutable weight and optimizer payloads on Drive and optionally on a second physical disk."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def promote(run,mirror=None):
    card=json.loads((run/"complete.json").read_bytes())
    checks=dict(card["sha256"])
    checks["complete.json"]=sha(run/"complete.json")
    if run.name!=card["run"]:
        raise ValueError("Run directory and identity differ")
    cloud=f"slwm-drive:SLWM/g5/checkpoints/{run.name}"
    hdd=None
    if mirror:
        if run.stat().st_dev==mirror.stat().st_dev:
            raise ValueError("Mirror must be a second physical filesystem")
        hdd=mirror/run.name
        hdd.mkdir(parents=True,exist_ok=True)
    for key in ("inference.npz","resume.npz","config.json","progress.json","complete.json"):
        path=run/key
        if sha(path)!=checks[key]:
            raise ValueError(f"Local checkpoint changed: {key}")
        if hdd and (not (hdd/key).exists() or sha(hdd/key)!=checks[key]):
            staging=hdd/(key+".partial")
            shutil.copyfile(path,staging)
            if sha(staging)!=checks[key]:
                raise ValueError("HDD copy mismatch")
            staging.replace(hdd/key)
        obj=cloud+"/"+key
        subprocess.run(["rclone","copyto","--immutable","--checksum","--retries","3",str(path),obj],check=True)
        remote=subprocess.check_output(["rclone","hashsum","SHA-256",obj],text=True).split()[0]
        if remote!=checks[key]:
            raise ValueError(f"Drive payload differs: {obj}")
    receipt={"run":run.name,"verified":checks,"drive":cloud,"second_disk":str(hdd) if hdd else None}
    tmp=run/"verified.partial"
    tmp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    tmp.replace(run/"verified.json")
    print(json.dumps(receipt),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs",type=Path,nargs="+"); p.add_argument("--mirror",type=Path)
    a=p.parse_args()
    for run in a.runs:
        promote(run,a.mirror)
