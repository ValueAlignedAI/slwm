"""Drain independent replacement assignments and close each as soon as its matched stage is verified."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time
from g10_observe import collect
from g9_storage import write

def monitor(ctl,name,seed):
    required={f"align-{core}-{mode}-s{seed}-1536.json" for core in ("hybrid","gpt2") for mode in ("paired","shifted","reconstruct")}
    expected=required|{f"align-{core}-rooms-s{seed}-1536.json" for core in ("hybrid","gpt2")}
    deadline=json.loads((ctl/"bounds.json").read_text())["deadline"]
    while time.time()<deadline:
        if (ctl/(name+"-release")).exists():return
        try:
            state=ctl/(name+"-restore.json")
            if state.exists() and json.loads(state.read_text()).get("phase")=="training":
                collect(name,ctl)
                marker=ctl/(name+"-eval.json")
                done=set(json.loads(marker.read_text())["files"]) if marker.exists() else set()
                receipts={p.name for p in (ctl/"receipts").glob("*.json")}
                if expected<=done and required<=receipts:
                    write(ctl/(name+"-completed.json"),dict(scores=len(expected),new_generations=len(required)))
                    (ctl/(name+"-release")).touch();return
        except (ValueError,OSError,RuntimeError,StopIteration,subprocess.SubprocessError) as exc:
            write(ctl/(name+"-collection-retry.json"),dict(error=str(exc)[:230],utc=time.time()))
        time.sleep(15)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("ctl",type=Path);a=ap.parse_args()
    with ThreadPoolExecutor(2) as pool:
        jobs=[pool.submit(monitor,a.ctl,n,s) for n,s in (("slwm-g10-ground-a",1001),("slwm-g10-ground-b",1002))]
        for j in jobs:j.result()
