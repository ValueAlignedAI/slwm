"""Resume a Colab-to-Fedora optimizer stream and independently verify both physical copies."""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from training.g6_checkpoint import export

DISKS=("/mnt/slwm-checkpoints-a","/mnt/slwm-checkpoints-b")
UUIDS=("fbbb53d9-dd42-4f19-95e6-dadab02a201e","c961a363-8d77-4fba-a48d-622b00a6d8c3")
SMALL=("config.json","progress.json","complete.json","dev-metrics.json","dev-predictions.npz")
HASH=re.compile(r"[0-9a-f]{64}")
TAG=re.compile(r"g6-(v5e|v6e)-(larger|anchor)-(hybrid|gpt2)-(511|512)")


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def write(p,doc):
    tmp=p.with_suffix(".new"); tmp.write_text(json.dumps(doc,sort_keys=True,indent=2)+"\n"); tmp.replace(p)


G7_TAG=re.compile(r"g7-(v5e|v6e)-(larger|anchor)-(hybrid|gpt2)-(511|512)-(replay|fresh)")


def metadata_files(tag):
    return SMALL+("wall.json",) if G7_TAG.fullmatch(tag) else SMALL


def stage_name(tag,step):
    if G7_TAG.fullmatch(tag) and step in {10240,12288,16384}: return Path(tag)/f"step-{step:06d}"
    if not TAG.fullmatch(tag) or step not in {128,512,2048,8192}:
        raise ValueError("Unregistered checkpoint name or update")
    return Path(tag)/f"step-{step:06d}"


def roots(study="g6"):
    if study not in ("g6","g7"): raise ValueError("Unknown archive study")
    for m,u in zip(DISKS,UUIDS):
        got=subprocess.check_output(["findmnt","-nro","UUID","-M",m],text=True).strip()
        if got!=u:
            raise ValueError("Missing registered physical checkpoint volume")
    return tuple(Path(m)/"SLWM"/study for m in DISKS)


def offset(a):
    name=stage_name(a.run,a.step); primary,_=roots(a.run[:2]); final=primary/name
    receipt=final/"verified.json"
    if receipt.is_file():
        rec=read(receipt)
        if rec["sha256"]["resume.npz"]!=a.sha:
            raise ValueError("An unrelated immutable result owns this stage")
        print(json.dumps({"verified":True,"offset":0})); return
    partial=final.parent/("."+final.name+"-"+a.sha)/"resume.part"
    print(json.dumps({"verified":False,"offset":partial.stat().st_size if partial.is_file() else 0}))


def send(a):
    folder=a.output/stage_name(a.run,a.step)
    done=read(folder/"complete.json")
    if done["run"]!=a.run or done["step"]!=a.step or done["sha256"]["resume.npz"]!=a.sha:
        raise ValueError("Remote milestone identity changed")
    meta={}
    for k in metadata_files(a.run):
        source=folder/k
        if source.stat().st_size>8*1024**2:
            raise ValueError("Unexpected auxiliary payload size")
        meta[k]=base64.b64encode(source.read_bytes()).decode("ascii")
    n=(folder/"resume.npz").stat().st_size
    if not 0<=a.offset<=n:
        raise ValueError("Invalid resume offset")
    header={"run":a.run,"step":a.step,"offset":a.offset,"bytes":n,"sha":a.sha,"metadata":meta}
    output=sys.stdout.buffer
    output.write(json.dumps(header,separators=(",",":")).encode()+b"\n")
    with (folder/"resume.npz").open("rb") as f:
        f.seek(a.offset)
        shutil.copyfileobj(f,output,4<<20)
    output.flush()


def receive(a):
    primary,mirror=roots(a.run[:2])
    src=sys.stdin.buffer
    line=src.readline(16*1024**2)
    if not line.endswith(b"\n"):
        raise ValueError("Invalid archive stream header")
    h=json.loads(line); name=stage_name(h["run"],h["step"])
    if (h["run"],h["step"],h["sha"])!=(a.run,a.step,a.sha):
        raise ValueError("Unexpected incoming checkpoint")
    if set(h["metadata"])!=set(metadata_files(a.run)) or not 8*1024**2<=h["bytes"]<=4*1024**3:
        raise ValueError("Invalid checkpoint envelope")
    final=primary/name; second=mirror/name
    for d in (final.parent,second.parent):
        d.mkdir(exist_ok=True,parents=True)
    folder=final.parent/("."+final.name+"-"+a.sha); folder.mkdir(exist_ok=True)
    replica=second.parent/("."+second.name+"-"+a.sha); replica.mkdir(exist_ok=True)
    with (final.parent/(final.name+".lock")).open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (final/"verified.json").exists():
            raise ValueError("Stage is already durably archived")
        # A crash between directory publication and the receipt is recoverable.
        staging=final if final.exists() else folder
        for k,value in h["metadata"].items():
            raw=base64.b64decode(value,validate=True)
            path=staging/k
            if path.exists() and path.read_bytes()!=raw:
                raise ValueError("Immutable auxiliary input changed")
            if not path.exists():
                path.write_bytes(raw)
        cfg=read(staging/"config.json"); done=read(staging/"complete.json")
        identity=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        prior=Path("/mnt/slwm-data/SLWM/g5/prior.npz")
        if cfg["run"]!=a.run or done["run"]!=a.run or done["step"]!=a.step or done["identity"]!=identity or cfg["prior_sha256"]!=sha(prior):
            raise ValueError("Checkpoint configuration or calibration changed")
        hashes=done["sha256"]
        for k in ("resume.npz","inference.npz","config.json","progress.json"):
            if not HASH.fullmatch(hashes[k]):
                raise ValueError("Invalid checkpoint hash")
        if hashes["resume.npz"]!=a.sha or any(sha(staging/k)!=hashes[k] for k in ("config.json","progress.json")):
            raise ValueError("Checkpoint metadata checksum mismatch")
        partial=staging/"resume.part"
        finished=staging/"resume.npz"
        if finished.exists() and not partial.exists():
            os.link(finished,partial)
        existing=partial.stat().st_size if partial.exists() else 0
        if h["offset"]!=existing and not (finished.exists() and a.offset==0 and h["offset"]==0):
            raise ValueError("Receiver offset changed")
        remaining=h["bytes"]-h["offset"]
        if finished.exists() and existing==h["bytes"] and h["offset"]==0:
            # Drop a redundant restart payload after crash-safe publication.
            while remaining:
                buf=src.read(min(4<<20,remaining))
                if not buf: raise EOFError("Short redundant stream")
                remaining-=len(buf)
        else:
            with partial.open("ab") as f:
                while remaining:
                    buf=src.read(min(4<<20,remaining))
                    if not buf: raise EOFError("Partial optimizer kept for resumption")
                    f.write(buf); remaining-=len(buf)
                f.flush(); os.fsync(f.fileno())
        if sha(partial)!=a.sha:
            partial.unlink()
            raise ValueError("Optimizer checksum mismatch")
        if not finished.exists():
            partial.replace(finished)
        else:
            partial.unlink(missing_ok=True)
        if not (staging/"inference.npz").is_file():
            export(finished,prior,staging/"inference.npz")
        if sha(staging/"inference.npz")!=hashes["inference.npz"]:
            raise ValueError("Regenerated inference weights differ from the accelerator")
        if read(staging/"dev-metrics.json")["checkpoint"]!=hashes["inference.npz"]:
            raise ValueError("Development scores refer to a different model")
        all_hashes=dict(hashes,**{k:sha(staging/k) for k in metadata_files(a.run) if k not in hashes})
        dst=second if second.exists() else replica
        for k,expected in all_hashes.items():
            out=dst/k
            if not out.exists() or sha(out)!=expected:
                tmp=dst/(k+".part"); shutil.copyfile(staging/k,tmp)
                if sha(tmp)!=expected:
                    raise ValueError("Second HDD mismatch")
                tmp.replace(out)
        if not second.exists(): replica.rename(second)
        if not final.exists(): folder.rename(final)
        receipt={"run":a.run,"step":a.step,"device":done["device"],"sha256":all_hashes,"disk_uuids":UUIDS,
                 "optimizer_bytes":h["bytes"],"inference_bytes":(final/"inference.npz").stat().st_size}
        write(second/"verified.json",receipt)
        write(final/"verified.json",receipt)
        print(json.dumps({"verified":a.run,"step":a.step,"inference_sha256":hashes["inference.npz"]}),flush=True)


if __name__=="__main__":
    q=argparse.ArgumentParser(description=__doc__); q.add_argument("action",choices=["send","receive","offset"])
    q.add_argument("--run",required=True); q.add_argument("--step",type=int,required=True); q.add_argument("--sha",required=True)
    q.add_argument("--offset",type=int,default=0); q.add_argument("--output",type=Path,default=Path("/content/g6-out"))
    a=q.parse_args()
    if not HASH.fullmatch(a.sha): q.error("Invalid digest")
    {"send":send,"receive":receive,"offset":offset}[a.action](a)
