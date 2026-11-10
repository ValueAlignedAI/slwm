"""Verified two-HDD checkpoint custody and lossless bounded Colab transfer packets."""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

VALID=re.compile(r"(pilot|confirm)-(hybrid|gpt2|gated)-z(128|512)-(gelu|snake)-s(911|912)-(fresh|replay)")
FILES={"weights.npz":"inference_sha256","optimizer.npz":"optimizer_sha256"}
CHUNK=16*2**20


def write(path,value):
    tmp=path.with_suffix(".new");tmp.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n");tmp.replace(path)


def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def disks():
    pair=(("/mnt/slwm-checkpoints-a","fbbb53d9-dd42-4f19-95e6-dadab02a201e"),("/mnt/slwm-checkpoints-b","c961a363-8d77-4fba-a48d-622b00a6d8c3"))
    for mount,expected in pair:
        if subprocess.check_output(["findmnt","-nro","UUID","-M",mount],text=True).strip()!=expected:raise ValueError("Registered disk unavailable")
    return tuple(Path(m)/"SLWM/g9/runs" for m,_ in pair)


def location(tag,step):
    if not VALID.fullmatch(tag) or step not in (384,1024,4096):raise ValueError("Unknown G9 generation")
    return Path(tag)/f"step-{step:05d}"


def bundle(tag,step):
    rel=location(tag,step);stage=Path("/content/g9-out")/rel;folder=Path("/content/g9-packets")/rel
    folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/"manifest.json"
    if manifest.exists():return
    meta=json.loads((stage/"receipt.json").read_text())
    if meta["tag"]!=tag or meta["step"]!=step:raise ValueError("Stage mismatch")
    doc={"receipt":meta,"logs":(stage.parent/"train.jsonl").read_text(),"files":{}}
    for filename,field in FILES.items():
        src=stage/filename;parts=[];digest=hashlib.sha256()
        with src.open("rb") as f:
            while raw:=f.read(CHUNK):
                digest.update(raw);wire=zlib.compress(raw,1);name=f"{filename}.{len(parts):04d}"
                (folder/name).write_bytes(wire)
                parts.append(dict(name=name,raw=len(raw),wire=len(wire),sha=hashlib.sha256(wire).hexdigest()))
        if digest.hexdigest()!=meta[field]:raise ValueError("Runtime source checksum changed")
        doc["files"][filename]={"bytes":src.stat().st_size,"parts":parts}
    write(manifest,doc)


def receive(tag,step,file,expected,size,offset):
    rel=location(tag,step)
    if file not in FILES or not re.fullmatch("[a-f0-9]{64}",expected) or not 0<size<2*2**30:raise ValueError("Unknown file")
    stage=disks()[0]/rel;stage.mkdir(parents=True,exist_ok=True);p=stage/(file+".partial")
    if offset!=(p.stat().st_size if p.exists() else 0):raise ValueError("Incoming offset changed")
    with p.open("ab") as out:
        shutil.copyfileobj(sys.stdin.buffer,out,1<<20);out.flush();os.fsync(out.fileno())
    if p.stat().st_size!=size:raise ValueError("Incomplete incoming checkpoint")
    if sha(p)!=expected:
        p.rename(p.with_suffix(".invalid"));raise ValueError("Incoming checksum mismatch")
    p.replace(stage/file)


def mirror(tag,step):
    rel=location(tag,step);a,b=disks();stage,second=a/rel,b/rel
    receipt=json.loads((stage/"receipt.json").read_text())
    if receipt["tag"]!=tag or receipt["step"]!=step:raise ValueError("Conflicting local generation")
    second.mkdir(parents=True,exist_ok=True)
    if (second/"verified.json").exists() and (stage/"verified.json").exists():return
    for file,field in FILES.items():
        if sha(stage/file)!=receipt[field]:raise ValueError("Primary checksum mismatch")
        if not (second/file).exists() or sha(second/file)!=receipt[field]:
            tmp=second/(file+".partial");shutil.copyfile(stage/file,tmp)
            if sha(tmp)!=receipt[field]:raise ValueError("Secondary checksum mismatch")
            tmp.replace(second/file)
    for p in (stage,second):
        write(p/"receipt.json",receipt);write(p/"verified.json",{"tag":tag,"step":step,"inference_sha256":receipt["inference_sha256"],"optimizer_sha256":receipt["optimizer_sha256"],"verified_copies":2})
    if step==receipt["goal_steps"]:write(stage.parent/"done.json",receipt)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("op",choices=["bundle","offset","receive","finish","mirror"])
    p.add_argument("tag");p.add_argument("step",type=int);p.add_argument("more",nargs="*");a=p.parse_args()
    if a.op=="bundle":bundle(a.tag,a.step)
    elif a.op=="mirror":mirror(a.tag,a.step)
    elif a.op=="receive":receive(a.tag,a.step,a.more[0],a.more[1],int(a.more[2]),int(a.more[3]))
    elif a.op=="offset":
        if a.more[0] not in FILES:raise ValueError("Unknown component")
        stage=disks()[0]/location(a.tag,a.step);file=stage/a.more[0];partial=stage/(a.more[0]+".partial")
        print(-1 if file.is_file() and sha(file)==a.more[1] else partial.stat().st_size if partial.exists() else 0)
    else:
        doc=json.load(sys.stdin);stage=disks()[0]/location(a.tag,a.step)
        write(stage/"receipt.json",doc["receipt"]);write(stage.parent/"recipe.json",doc["receipt"])
        (stage.parent/"train.jsonl").write_text(doc["logs"]);mirror(a.tag,a.step)
