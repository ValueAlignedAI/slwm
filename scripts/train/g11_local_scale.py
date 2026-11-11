"""Execute dated equal-exposure parent/transfer jobs with a shared CUDA lock and resumable weights."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT=Path("/mnt/slwm-data/SLWM/g11");DISK=Path("/mnt/slwm-checkpoints-a/SLWM")
TARGET=DISK/"g11/scale-parents";OUT=DISK/"g11/scale";CACHE=str(ROOT/"cache")
PY=["flock","/tmp/slwm_research_gpu.lock","bash","scripts/train/g4_cuda.sh"]
SEEDS=(911,912)

def call(args,log):
    with log.open("a") as f:
        subprocess.run(PY+args,env=dict(os.environ,PYTHONPATH="."),stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    TARGET.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    for family,kind in (("small","exposure"),("large","confirm")):
        for seed in SEEDS:
            for core in ("hybrid","gpt2"):
                tag=f"{kind}-{core}-z128-gelu-s{seed}-fresh";src=DISK/"g9/runs"/tag;dst=TARGET/tag
                if not (src/"step-04096/verified.json").is_file():raise ValueError("Missing matched G9 parent")
                if not dst.exists():dst.symlink_to(src,target_is_directory=True)
    for size in ("exposure","confirm"):
        for seed in SEEDS:
            tag=f"{size}-gated-z128-gelu-s{seed}-fresh";stage=TARGET/tag/"done.json"
            if not stage.exists():
                code="import jax;from pathlib import Path;from scripts.train.g9_fit import bank_data,one;jax.config.update('jax_compilation_cache_dir',%r);b,s,n=bank_data(Path(%r));one((%r,'gated',128,'gelu',%d,False),b,s,n,Path(%r))"%(CACHE,"/mnt/slwm-data/SLWM/g9/data/starss-train.npz",size,seed,str(TARGET))
                call(["-c",code],ROOT/"logs/scale-parents.log")
    while not (ROOT/"data/banks/effects-ready").exists():time.sleep(6)
    for family in ("small","large"):
        for seed in SEEDS:
            for fold in ("A","B"):
                allocation=ROOT/"scale-routing.json"
                if allocation.exists() and str(seed) in json.loads(allocation.read_text()).get("colab_seeds",[]):continue
                for phase,stop in (("action",256),("joint",1024)):
                    for core in ("hybrid","gpt2","gated"):
                        tag=f"g11s-{fold}-{family}-{core}-paired-s{seed}"
                        if (OUT/tag/f"step-{stop:05d}/receipt.json").exists():continue
                        call(["scripts/train/g11_fit.py",str(ROOT/"data/banks"),str(TARGET),str(OUT),phase,"--folds",fold,"--seeds",str(seed),"--cores",core,
                              "--treatments","paired","--family",family,"--cache",CACHE],ROOT/"logs/scale-training.log")
    print("local scale assignment drained",flush=True)

if __name__=="__main__":main()
