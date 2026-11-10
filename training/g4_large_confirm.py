"""Run both frozen medium and anchor pairs on new tasks and benchmark batched confirmation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from training.g4_eval import evaluate
from training.g4_lock import freeze
from training.g4_train import Data,read,sha_file,write
from training.g4_transfer import transfer


def main():
    q=argparse.ArgumentParser()
    for field in ("root","data","probes","pins"): q.add_argument("--"+field,type=Path,required=True)
    a=q.parse_args()
    locks={}
    for scale in ("medium","anchor"):
        file=a.root/(scale+"-confirmation.json")
        freeze(a.root,"v5e",scale,[81],["gpt2-cross","signal-cross"],file)
        locks[scale]=file
    subprocess.run(["python","scripts/train/g4_cloud_fetch.py","test","--root",str(a.data)],check=True)
    a.probes.mkdir(parents=True,exist_ok=True)
    subprocess.run(["rclone","copy","slwm-drive:SLWM/g4/probes-v1",a.probes,"--include","*.npz","--transfers","6"],check=True)
    for name,digest in read(a.pins).items():
        if sha_file(a.probes/name)!=digest: raise ValueError("Unseen task arrays changed: "+name)
    data=Data(a.data,"test"); done=[]
    for scale in ("medium","anchor"):
        for profile in ("gpt2-cross","signal-cross"):
            run=a.root/f"g4-v5e-{scale}-{profile}-81"
            batch=96 if scale=="medium" else 64
            write(a.root/"large-confirm-status.json",{"active":run.name,"completed":done,"time":time.time()})
            evaluate(run,a.data,"test",batch,"tpu",locks[scale],data_cache=data)
            transfer(run,a.data,a.probes,"test",batch,"tpu",locks[scale])
            done.append(run.name)
    write(a.root/"large-confirm-status.json",{"stage":"complete","completed":done,"time":time.time()})


if __name__=="__main__": main()
