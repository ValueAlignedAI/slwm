"""Drain verified G9 jobs into locked development and held scores on the research GPU."""
import argparse
import json
from pathlib import Path
import time
import jax
from scripts.train.g9_evaluate import evaluate
from scripts.train.g9_storage import disks,mirror
from evals.g9_live import raw_receipt
from scripts.train.g9_fit import save_json

ROOT=Path("/mnt/slwm-data/SLWM/g9")


def run():
    primary,_=disks();data=ROOT/"data";dest=ROOT/"scores";dest.mkdir(exist_ok=True)
    jax.config.update("jax_compilation_cache_dir",str(ROOT/"eval-cache"))
    save_json(dest/"raw-oracle.json",raw_receipt())
    while True:
        jobs=[]
        for p in primary.glob("*/step-*/receipt.json"):
            d=json.loads(p.read_text())
            if d["step"] not in (384,1024,4096):continue
            terminal=d["step"]==d["goal_steps"]
            if d["tag"].startswith(("confirm","exposure")) and not (p.parent/"verified.json").exists():continue
            for role in (("dev","test") if terminal else ("dev",)):
                key=d["tag"]+"-"+str(d["step"])+"-"+role
                record=dest/(key+".json");done=json.loads(record.read_text()) if record.exists() else {}
                needed=["live","history","sim","room"]+(["contact"] if role=="test" else [])
                available=[k for k in needed if k not in done and
                    (k!="room" or (data/f"starss-{role}.npz").stat().st_size>100000) and
                    (k!="contact" or (data/"contact/roma-held.npz").is_file())]
                if available:
                    # Score the primary held endpoint for a matched wide pair early; the promotion uses development only.
                    priority=(role!="test",d["tag"].find("-z512-")<0,"-gelu-" not in d["tag"],"fresh" not in d["tag"],d["seed"],d["tag"],d["step"])
                    jobs.append((priority,p.parent,role,available))
        if jobs:
            _,stage,role,parts=min(jobs)
            evaluate(stage,data,dest,role,parts)
            continue
        finished=[p for p in dest.glob("*-test.json") if all(k in json.loads(p.read_bytes()) for k in ("live","history","sim","room","contact"))]
        dev=[p for p in dest.glob("*-dev.json") if all(k in json.loads(p.read_bytes()) for k in ("live","history","sim","room"))]
        if len(finished)>=38 and len(dev)>=54:return
        time.sleep(8)


if __name__=="__main__":run()
