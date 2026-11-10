"""Stream pinned G7 inputs and exact G6 optimizer states to one native TPU, then run the registered lane."""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
from g7_inputs import BASE_FILES,FED,parent,shell,stream

G7_REPORT="experiments/multimodal/t7/continuation-alignment-control-20261110/metrics/checkpoint-inventory.json"
CODE=("data","models","training","configs/t5","configs/t6","configs/t7","configs/t8","scripts/train",G7_REPORT)


def marker(name,path):
    return shell(name,"test -s "+path+" && echo yes || echo no",capture_output=True,text=True).stdout.strip()=="yes"


def prepare(a):
    control=a.control; deadline=json.loads((control/"bounds.json").read_bytes())["deadline"]
    budget=json.loads((control/"budget.json").read_bytes())
    bounds=json.loads((control/"bounds.json").read_bytes())
    if a.name not in bounds["sessions"] or budget["phase"]!="watching":raise ValueError("Unmetered assignment")
    with (control/(a.name+".transport")).open("a") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        subprocess.run(["bash","scripts/train/colab_session.sh","exec",a.name,"-f","scripts/train/g4_colab_topology.py","--timeout","90"],check=True,timeout=140)
        stream(a.name,["git","archive",a.revision,*CODE],"mkdir -p /content/g8-code && tar -xf - -C /content/g8-code")
        if not marker(a.name,"/content/g8-base-ready"):
            src=[*FED,"tar","-cf","-","-C","/mnt/slwm-data/SLWM/g5/local-packs","dev.sha256","train.sha256","prior.sha256",*BASE_FILES,"-C","/mnt/slwm-data/SLWM/g5","prior.npz"]
            stream(a.name,src,"set -eu; mkdir -p /content/g5-landing; cd /content/g5-landing; tar -xf -; sha256sum -c dev.sha256 train.sha256 prior.sha256")
            shell(a.name,"set -eu; cd /content/g8-code; python -c 'import optax,scipy,zstandard' || python -m pip -q install optax scipy zstandard; "
                 "mkdir -p /content/g8-data; python scripts/train/g5_cloud_data.py dev --root /content/g8-data --pins /content/g5-landing & child=$!; "
                 "python scripts/train/g5_cloud_data.py train --root /content/g8-data --pins /content/g5-landing; wait $child; "
                 "cp /content/g5-landing/prior.npz /content/g8-data/prior.npz; python scripts/train/g6_ram_inputs.py /content/g8-data; echo ready > /content/g8-base-ready",timeout=750)
        pin=json.loads(subprocess.check_output([*FED,"cat","/mnt/slwm-data/SLWM/g7/packets/train.verified.json"],text=True))
        if not marker(a.name,"/content/g8-fresh/.verified"):
            sha=pin["sha256"]
            if len(sha)!=64 or not all(c in "0123456789abcdef" for c in sha):raise ValueError("Invalid immutable packet hash")
            stream(a.name,[*FED,"cat","/mnt/slwm-data/SLWM/g7/packets/g7-fresh-train.tar.zst"],
                "set -eu; cd /content/g8-code; mkdir -p /dev/shm/g8-fresh; test -e /content/g8-fresh || ln -s /dev/shm/g8-fresh /content/g8-fresh; cat > /content/g8-fresh-train.tar.zst; "
                f"python scripts/train/g7_extract_fresh.py /content/g8-fresh-train.tar.zst {sha} /content/g8-fresh train")
        if shell(a.name,"cat /content/g8-fresh/.verified",capture_output=True,text=True).stdout.strip()!=pin["sha256"]:raise ValueError("Fresh data changed")
        first="hybrid" if a.seed==511 else "gpt2"
        parent(a.name,a.seed,"anchor",first)
        # Starting the first bridge promptly overlaps the second immutable parent transfer.
        script=("set -eu; cd /content/g8-code; mkdir -p /content/g8-out; "
                f"touch /content/g8-out/seed-{a.seed}; source /content/g4-topology.env; "
                "export JAX_PLATFORMS=tpu JAX_COMPILATION_CACHE_DIR=/content/g8-xla JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2; "
                f"nohup flock -n /content/g8-lane.lock python -u scripts/train/g8_lane.py --deadline {int(deadline)} </dev/null >>/content/g8-lane.log 2>&1 &")
        shell(a.name,script)
        parent(a.name,a.seed,"anchor","gpt2" if first=="hybrid" else "hybrid")
        print("Prepared both parents, immutable replay/fresh data, and active lane",a.name,a.seed,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--name",required=True);p.add_argument("--seed",type=int,choices=(511,512),required=True)
    p.add_argument("--control",type=Path,required=True);p.add_argument("--revision",required=True);prepare(p.parse_args())
