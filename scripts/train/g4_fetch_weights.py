"""Restore a verified, portable G4 inference group directly onto a compute host."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time


def sha(path):
    with path.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()


def fetch(root,group,remote):
    root.mkdir(parents=True,exist_ok=True)
    for _ in range(60):
        check=subprocess.run(["rclone","cat",f"{remote}/{group}-inference.json"],capture_output=True)
        if check.returncode==0: break
        if check.returncode not in (3,4): raise RuntimeError(f"Drive failed with code {check.returncode}")
        time.sleep(10)
    else: raise TimeoutError("Inference bundle receipt still pending")
    card=json.loads(check.stdout)
    if card["group"]!=group or card["object"]!=f"{remote}/{group}-{card['sha256']}.tar":
        raise ValueError("Wrong archive identity")
    with tempfile.TemporaryDirectory(dir=root,prefix=".fetch-") as tmp:
        temp=Path(tmp); archive=temp/"weights.tar"
        subprocess.run(["rclone","copyto",card["object"],archive],check=True)
        if sha(archive)!=card["sha256"]: raise ValueError("Inference archive checksum mismatch")
        dest=temp/"models"; dest.mkdir()
        with tarfile.open(archive) as tar:
            for item in tar:
                if not (item.isfile() or item.isdir()): raise ValueError("Unexpected archive entry")
                tar.extract(item,dest,filter="data")
        if {p.name for p in dest.iterdir()}!=set(card["runs"]): raise ValueError("Wrong model inventory")
        for run in dest.iterdir():
            record=json.loads((run/"complete.json").read_text())
            if sha(run/"model.npz")!=record["inference_sha256"]: raise ValueError("Damaged model")
            target=root/run.name
            if not target.exists(): run.rename(target)
            elif json.loads((target/"complete.json").read_text())["weight_file_sha256"]!=record["weight_file_sha256"]:
                raise ValueError("Conflicting local run")
    (root/(group+"-inference.json")).write_bytes(check.stdout)
    print("verified",len(card["runs"]),"inference models",card["sha256"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--group",required=True)
    p.add_argument("--remote",default="slwm-drive:SLWM/g4/weights")
    a=p.parse_args(); fetch(a.root,a.group,a.remote)
