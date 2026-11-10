"""Stream verifiable G4 model files through SSH without creating a Mac-side data file."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import numpy as np


def sha(path):
    with path.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()


def durable(root,run,card):
    if (root/"full-optimizer-receipts"/(run.name+".json")).exists(): return True
    return any(Path("/content/g4-spool/receipts",run.name).glob(f"{card['step']:012d}-*.json"))


def send(root):
    out=sys.stdout.buffer
    runs=sorted((p.parent for p in root.glob("g4-*/complete.json")),key=lambda r:(
        "tied-medium" not in r.name, "all-medium" not in r.name, "anchor" not in r.name, r.name))
    with tempfile.TemporaryDirectory(dir=root,prefix=".handoff-") as tmp:
        for kind in ("model","resume"):
            for r in runs:
                complete=json.loads((r/"complete.json").read_bytes())
                if kind=="model" and r.name.startswith("g4-v5e-pilot-"): continue
                if kind=="resume" and durable(root,r,complete): continue
                source=r/"resume.npz"
                if kind=="model":
                    dest=Path(tmp)/"model.npz"
                    with np.load(source,allow_pickle=False) as s:
                        np.savez(dest,**{k:s[k] for k in s.files if k.startswith("p/") or k in ("step","identity","stats")})
                    source=dest
                digest=sha(source) if kind=="model" else complete["weight_file_sha256"]
                cfg=json.loads((r/"config.json").read_bytes())
                head={"run":r.name,"kind":kind,"bytes":source.stat().st_size,"sha":digest,"complete":complete,"config":cfg}
                out.write((json.dumps(head,separators=(",",":"))+"\n").encode())
                with source.open("rb") as f: shutil.copyfileobj(f,out,4<<20)
                out.flush()
    out.write(b'{"done":true}\n'); out.flush()


def receive(root):
    root.mkdir(parents=True,exist_ok=True); stream=sys.stdin.buffer
    while True:
        line=stream.readline(65536)
        if not line: raise EOFError("Transport ended between checked files")
        meta=json.loads(line)
        if meta.get("done"): return
        name,kind=meta["run"],meta["kind"]
        if not re.fullmatch(r"g4-[a-z0-9-]+",name) or kind not in ("model","resume") or meta["config"]["name"]!=name:
            raise ValueError("Invalid endpoint")
        if not 0<meta["bytes"]<=4*(1<<30): raise ValueError("Invalid file size")
        directory=root/name; directory.mkdir(exist_ok=True)
        target=directory/(kind+".npz"); part=directory/(kind+".partial")
        remaining=meta["bytes"]; h=hashlib.sha256()
        with part.open("wb") as f:
            while remaining:
                block=stream.read(min(4<<20,remaining))
                if not block: raise EOFError(f"Verified earlier files; partial {name} retained")
                f.write(block); h.update(block); remaining-=len(block)
        if h.hexdigest()!=meta["sha"]: raise ValueError("Transport checksum failed")
        if target.exists() and sha(target)!=meta["sha"]: raise ValueError("Conflicting committed weights")
        part.replace(target)
        complete=meta["complete"]
        if kind=="model": complete["inference_sha256"]=meta["sha"]
        elif (directory/"model.npz").exists(): complete["inference_sha256"]=sha(directory/"model.npz")
        for file,body in (("config.json",meta["config"]),("complete.json",complete)):
            temp=directory/(file+".new"); temp.write_text(json.dumps(body,sort_keys=True)+"\n"); temp.replace(directory/file)
        print("verified Fedora handoff",name,kind,meta["bytes"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("mode",choices=["send","receive"]); p.add_argument("--root",type=Path,required=True)
    a=p.parse_args(); (send if a.mode=="send" else receive)(a.root)
