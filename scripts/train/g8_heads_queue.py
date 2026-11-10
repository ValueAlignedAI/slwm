"""Fit the fixed 512-start transfer heads as soon as each verified parent is durable."""
import time
import subprocess
from pathlib import Path
from training.g8_models import cases,key,resolve
from scripts.train.g6_artifacts import roots

def main():
    root,_=roots("g8");until=time.monotonic()+10*3600
    while time.monotonic()<until:
        ready=["physical","raw-511","raw-512"];total=len(ready)+len(cases())
        for case in cases():
            if resolve(*case):ready.append(key(*case))
        pending=[k for k in ready if not (root/"reacher-heads"/k/"complete.json").exists()]
        if pending:
            subprocess.run(["flock","/tmp/slwm_research_gpu.lock","bash","scripts/train/g4_cuda.sh","-m","training.g8_reacher","train",pending[0]],check=True)
        elif len(ready)==total:
            print("All",total,"transfer heads frozen",flush=True);return
        else:time.sleep(24)
    raise TimeoutError("Transfer head queue did not finish")

if __name__=="__main__":main()
