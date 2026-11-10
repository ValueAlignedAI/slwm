"""Independently re-read registered G8 optimizers and mirror complete heads, panels and inference examples."""
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from scripts.train.g6_artifacts import roots,stage_name
from scripts.train.g6_archive_audit import check_disk
from scripts.train.g7_lock_heads import sha
from scripts.train.g7_mirror import copy
from training.g7_units import put
from training.g8_models import read,CONFIG

STABLE=("live-oracle/complete.json","full-speech/complete.json","data/complete.json","acoustic-codec/complete.json","acoustic-heads/*/complete.json","speech-audit/*/*/scores.json",
        "rate-codec/complete.json","rate-heads/*/complete.json","rate-audit/*/*/scores.json","rate-example/verified.json",
        "reacher-heads/*/complete.json","confirmation/*/*/metrics.json","live/*/*/metrics.json","retention/*/metrics.json")
SUFFIX={".npz",".pt",".json",".wav",".bin",".ogg",".webm",".slwma",".png",".zst",".gz"}


def artifacts(a,b):
    index=a/"readouts-archive.json";files=read(index)["files"] if index.is_file() else {};markers=[]
    for pattern in STABLE:
        for mark in a.glob(pattern):
            markers.append(str(mark.relative_to(a)));folder=mark.parent
            for f in folder.iterdir():
                if f.is_file() and f.suffix in SUFFIX:copy(a,b,f.relative_to(a),files)
    for key in ("confirmation-lock.json","speech-audit/lock.json","rate-audit/lock.json"):
        if (a/key).is_file():copy(a,b,Path(key),files)
    for f in (a/"bridges").glob("*.json"):copy(a,b,f.relative_to(a),files)
    info={"files":files,"verified_files":len(files),"bytes":sum(v["bytes"] for v in files.values()),"finished_markers":sorted(markers)}
    for root in (a,b):put(root/"readouts-archive.json",info)
    return info


def pass_once(a,b):
    p=a/"archive-audit.json";checked=read(p)["rungs"] if p.exists() else {}
    stages=sorted(str(f.parent.relative_to(a)) for f in a.glob("g8-*/step-*/verified.json") if (b/f.relative_to(a)).is_file())
    unseen=[relative for relative in stages if relative not in checked]
    if unseen:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first,second=[f.result() for f in [pool.submit(check_disk,d,unseen) for d in (a,b)]]
        if first!=second:raise ValueError("Two independent G8 disk reads disagree")
        checked.update(first["milestones"])
    bridges=list((a/"bridges").glob("*.json"))
    expected=sum((2 if read(f)["exact"] else 4)*len(read(CONFIG)["rungs"]) for f in bridges)
    done=len(bridges)==4 and len(checked)==expected
    index={"complete":done,"verified_rungs":len(checked),"expected":expected if len(bridges)==4 else None,"rungs":checked,
           "optimizer_and_inference_bytes_per_disk":sum(v["optimizer_bytes"]+v["inference_bytes"] for v in checked.values())}
    for root in (a,b):put(root/"archive-audit.json",index)
    info=artifacts(a,b)
    if unseen:print("G8 independent disk readback",len(checked),"auxiliary files",info["verified_files"],flush=True)
    return done and (a/"public/release-ready.json").is_file()


def audit(watch):
    a,b=roots("g8");deadline=time.time()+12*3600
    while time.time()<deadline:
        with (a/".g8-custody.lock").open("a") as guard:
            fcntl.flock(guard,fcntl.LOCK_EX);finished=pass_once(a,b)
        if not watch or finished:return
        time.sleep(45)
    raise TimeoutError("G8 disk audit horizon elapsed")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--watch",action="store_true");audit(p.parse_args().watch)
