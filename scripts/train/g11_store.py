"""Verify immutable G11 adapter generations and matched missing recurrent parents on two distinct disks."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import re
import shutil
import time
from g9_storage import disks,sha,write

MAIN=re.compile(r"g11-[AB]-(hybrid|gpt2|gated)-(paired|deranged|absent)-s100[123]")
SCALE=re.compile(r"g11s-[AB]-(small|large)-(hybrid|gpt2|gated)-paired-s91[12]")
PARENT=re.compile(r"(exposure|confirm)-gated-z128-gelu-s91[12]-fresh")

def roots(group):
    if group not in ("runs","scale","scale-parents"):raise ValueError(group)
    return tuple(p.parent.parent/"g11"/group for p in disks())

def eligible(group,tag,step):
    return (group=="runs" and MAIN.fullmatch(tag) and step in (256,640,1024)) or (group=="scale" and SCALE.fullmatch(tag) and step in (256,640,1024)) or (group=="scale-parents" and PARENT.fullmatch(tag) and step in (1024,2048,3072,4096))

def mirror(group,tag,step):
    if not eligible(group,tag,step):raise ValueError("Unregistered generation")
    a,b=roots(group);rel=Path(tag)/f"step-{step:05d}";src=a/rel;dst=b/rel
    if (src/"verified.json").exists() and (dst/"verified.json").exists():return
    dst.mkdir(parents=True,exist_ok=True)
    with (src/".g11-custody.lock").open("a") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        if (src/"verified.json").exists() and (dst/"verified.json").exists():return
        info=json.loads((src/"receipt.json").read_text())
        if (info["tag"],info["step"])!=(tag,step):raise ValueError("Immutable identity conflict")
        for name,key in (("weights.npz","inference_sha256"),("optimizer.npz","optimizer_sha256")):
            if sha(src/name)!=info[key]:raise ValueError("Primary checksum mismatch")
            target=dst/name
            if not target.is_file() or sha(target)!=info[key]:
                tmp=target.with_suffix(".copying");shutil.copyfile(src/name,tmp)
                if sha(tmp)!=info[key]:raise ValueError("Replica readback mismatch")
                tmp.replace(target)
        for root in (src,dst):
            write(root/"receipt.json",info);write(root/"verified.json",dict(tag=tag,step=step,parent_sha256=info.get("parent_sha256"),inference_sha256=info["inference_sha256"],optimizer_sha256=info["optimizer_sha256"],copies=2))
        if (src.parent/"recipe.json").is_file():shutil.copyfile(src.parent/"recipe.json",dst.parent/"recipe.json")
        if step==info["goal_steps"]:
            for root in (src,dst):write(root.parent/"done.json",info)

def main():
    a,_=roots("runs")
    while not (a.parent/"finalized").exists():
        for group in ("runs","scale","scale-parents"):
            primary,_=roots(group)
            for p in sorted(primary.glob("*/step-*/receipt.json")):
                tag=p.parent.parent.name;step=int(p.parent.name.split("-")[1])
                if eligible(group,tag,step) and not (p.parent/"verified.json").exists():mirror(group,tag,step)
        time.sleep(5)

if __name__=="__main__":main()
