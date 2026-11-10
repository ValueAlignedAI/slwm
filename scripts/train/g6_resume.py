"""Carry the latest independently verified optimizer per arm into a replacement TPU of the same generation."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import tempfile

from scripts.train.g6_artifacts import roots,sha

STEPS={128,512,2048,8192}
META=("config.json","progress.json","complete.json","dev-metrics.json")


def emit(generation,seed):
    first,second=roots(); files={}; snapshots={}
    for scale in ("larger","anchor"):
        for core in ("hybrid","gpt2"):
            tag=f"g6-{generation}-{scale}-{core}-{seed}"
            stages=[]
            for receipt in sorted((first/tag).glob("step-*/verified.json")):
                row=json.loads(receipt.read_bytes()); stage=receipt.parent
                if row["run"]!=tag or row["step"] not in STEPS or (second/tag/stage.name/"verified.json").read_bytes()!=receipt.read_bytes():
                    raise ValueError("Invalid independent receipt during restore")
                stages.append((row["step"],stage,row))
                for k in META:
                    files[f"{tag}/{stage.name}/{k}"]=(stage/k,row["sha256"][k])
            if stages:
                step,stage,row=max(stages)
                snapshots[tag]=step
                for k in ("resume.npz","config.json","progress.json"):
                    files[f"{tag}/{k}"]=(stage/k,row["sha256"][k])
    record={"generation":generation,"seed":seed,"steps":snapshots,"files":{k:v[1] for k,v in files.items()}}
    payload=json.dumps(record,sort_keys=True).encode()
    with tarfile.open(mode="w|",fileobj=sys.stdout.buffer,dereference=True) as tar:
        hdr=tarfile.TarInfo("restore.json"); hdr.size=len(payload); tar.addfile(hdr,io.BytesIO(payload))
        for name,(path,digest) in files.items():
            tar.add(path,arcname=name,recursive=False)


def unpack(root,generation,seed):
    root.mkdir(exist_ok=True,parents=True)
    with tempfile.TemporaryDirectory(dir=root,prefix=".restoring-") as tmp,tarfile.open(fileobj=sys.stdin.buffer,mode="r|") as tar:
        tmp=Path(tmp); members=iter(tar); header=next(members)
        if header.name!="restore.json" or not header.isfile() or header.size>100000:
            raise ValueError("Invalid restore manifest")
        plan=json.loads(tar.extractfile(header).read())
        if (plan["generation"],plan["seed"])!=(generation,seed):
            raise ValueError("Checkpoint generation or seed changed")
        if not plan["files"]:
            print("No existing matching checkpoint; training from initialized weights",flush=True); return
        seen=set()
        pattern=rf"g6-{generation}-(anchor|larger)-(hybrid|gpt2)-{seed}/(step-(000128|000512|002048|008192)/)?(config.json|progress.json|complete.json|dev-metrics.json|resume.npz)"
        for member in members:
            name=member.name
            if (name in seen or name not in plan["files"] or not re.fullmatch(pattern,name) or not member.isfile()
                    or member.size>4*1024**3):
                raise ValueError("Unexpected restore member")
            seen.add(name); dest=tmp/name; dest.parent.mkdir(parents=True,exist_ok=True)
            dig=hashlib.sha256()
            with tar.extractfile(member) as src,dest.open("wb") as out:
                while buf:=src.read(4<<20):
                    dig.update(buf); out.write(buf)
            if dig.hexdigest()!=plan["files"][name]:
                raise ValueError("Restored checkpoint checksum mismatch")
        if seen!=set(plan["files"]): raise ValueError("Truncated restore")
        for tag,step in plan["steps"].items():
            if not re.fullmatch(rf"g6-{generation}-(anchor|larger)-(hybrid|gpt2)-{seed}",tag):
                raise ValueError("Invalid restore run ID")
            folder=tmp/tag; config=json.loads((folder/"config.json").read_bytes())
            with __import__("numpy").load(folder/"resume.npz",allow_pickle=False) as f:
                identity=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
                if int(f["step"])!=step or str(f["identity"])!=identity: raise ValueError("Restored optimizer identity mismatch")
            if (root/tag).exists(): raise FileExistsError("Refusing to cover an existing active run")
        for tag in plan["steps"]:
            (tmp/tag).rename(root/tag)
        print("Restored verified optimizers",json.dumps(plan["steps"]),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("action",choices=("send","receive"))
    p.add_argument("--generation",required=True,choices=("v5e","v6e")); p.add_argument("--seed",type=int,choices=(511,512),required=True)
    p.add_argument("--output",default=Path("/content/g6-out"),type=Path); a=p.parse_args()
    if a.action=="send": emit(a.generation,a.seed)
    else: unpack(a.output,a.generation,a.seed)
