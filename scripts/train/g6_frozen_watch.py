"""Evaluate newly durable checkpoints at the preregistered common exposure, one CPU job at a time."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run(root,data,deadline):
    dest=root/"shared-probes"; dest.mkdir(exist_ok=True)
    while time.time()<deadline:
        pending=[]
        for file in sorted(root.glob("g6-*/step-002048/verified.json")):
            tag=file.parent.parent.name
            if not (dest/(tag+".json")).exists():
                pending.append((tag,file.parent))
        for tag,stage in pending:
            with (dest/(tag+".log")).open("ab") as output:
                subprocess.run([sys.executable,"-u","-m","training.g6_state_transfer","--data",str(data),
                                "--run",str(stage),"--out",str(dest/(tag+".json"))],stdout=output,stderr=subprocess.STDOUT,check=True)
            print("Ran frozen transfer",tag,flush=True)
        if len(list(dest.glob("g6-*.json")))>=8: return
        time.sleep(50)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--root",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True); p.add_argument("--deadline",type=float,required=True)
    a=p.parse_args(); run(a.root,a.data,a.deadline)
