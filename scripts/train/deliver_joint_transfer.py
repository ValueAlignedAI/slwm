"""Mirror G3 inference artifacts, verify locked milestones and publish focused reports."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

BRANCH = "training/joint-transfer-20261109"
REL = Path("experiments/multimodal/t4/joint-transfer-20261109")
REMOTE = os.environ.get("SLWM_G3_REMOTE","db@192.168.1.200:/home/db/dev/slwm-g3")
OPS = Path("artifacts/joint_transfer/g3/delivery")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as f:
        h = hashlib.sha256()
        while block:=f.read(8<<20):
            h.update(block)
        return h.hexdigest()


def write(path,data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")
    temp.replace(path)


def git(*args):
    return subprocess.check_output(["git",*map(str,args)],text=True).strip()


def commit_files(paths,message):
    if git("branch","--show-current")!=BRANCH:
        raise RuntimeError("Not on G3 publication branch")
    git("add","-f","--",*paths)
    if git("diff","--cached","--name-only","--",*paths):
        git("commit","--only","-m",message,"--",*paths)


def collect():
    REL.mkdir(parents=True,exist_ok=True)
    OPS.mkdir(parents=True,exist_ok=True)
    include = ["*/","step-*.pt","step-*.json","*-32shot-head.npz","dev.json","test.json","status.json","training_lock.json",
        "confirmation_lock.json","summary-*.json","summary-*.md","depth-*.json","results.json","lock.json","progress.json","inference_check.json","config.json"]
    patterns = ["--exclude=*.tmp*"]
    patterns += ["--include="+x for x in include]+["--exclude=*"]
    cmd = ["rsync","-art","--partial","--timeout=90","-e","ssh -o BatchMode=yes -o ConnectTimeout=8",*patterns,
           REMOTE+"/"+str(REL)+"/",str(REL)+"/"]
    subprocess.run(cmd,check=True)
    for stage,role in ((100,"dev"),(1000,"dev"),(8000,"test")):
        report = REL/f"summary-{stage}-{role}.json"
        if not report.exists():
            continue
        proof = REL/f"verified-{stage}.json"
        if not proof.exists():
            hashes = {}
            for run in read(report)["rows"]:
                card = read(REL/run/f"step-{stage}.json")
                local = REL/run/f"step-{stage}.pt"
                if sha(local)!=card["delta_sha256"]:
                    raise ValueError(f"Delta verification failed for {run}")
                hashes[run] = {"delta_sha256":card["delta_sha256"],"parent_sha256":card["parent_sha256"],
                               "bytes":local.stat().st_size,"exposure_sha256":card["exposure_sha256"]}
            write(proof,{"milestone":stage,"verified_inference_deltas":hashes})
        paths = [report,report.with_suffix(".md"),proof,REL/"training_lock.json"]
        if role=="test":
            paths.append(REL/"confirmation_lock.json")
        commit_files(paths,f"research: publish verified {stage}-update {role} transfer comparison")
    depth = REL/"depth6"
    for stage,role in ((100,"dev"),(1000,"dev"),(8000,"test")):
        report = depth/f"depth-{stage}-{role}.json"
        if report.exists():
            proof = depth/f"verified-{stage}.json"
            if not proof.exists():
                hashes = {}
                for run in read(report)["rows"]:
                    card = read(depth/run/f"step-{stage}.json")
                    local = depth/run/f"step-{stage}.pt"
                    if sha(local)!=card["delta_sha256"]:
                        raise ValueError(f"Reduced-depth delta verification failed for {run}")
                    hashes[run] = {"delta_sha256":card["delta_sha256"],"parent_sha256":card["parent_sha256"],"bytes":local.stat().st_size}
                write(proof,{"milestone":stage,"verified_inference_deltas":hashes})
            paths = [report,proof,depth/"training_lock.json",depth/"config.json"]
            if role=="test":
                paths.append(depth/"confirmation_lock.json")
            commit_files(paths,f"research: publish matched retained-depth transfer at {stage} updates")
    for role in ("screen","scale"):
        r = REL/"pusht"/role/"results.json"
        if r.exists():
            commit_files([r,r.parent/"lock.json"],f"research: publish {role} held-episode physical forecasting results")
    if git("rev-list","--count",f"origin/{BRANCH}..HEAD")!="0":
        git("push","origin",BRANCH)
    phase = read(REL/"status.json") if (REL/"status.json").exists() else {}
    complete = (REL/"summary-8000-test.json").exists() and (REL/"pusht/scale/results.json").exists() and (depth/"depth-8000-test.json").exists()
    write(OPS/"latest.json",{"time":datetime.now(timezone.utc).isoformat(),"main":phase,"complete":complete,"published":git("rev-parse","HEAD")})
    return complete


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--watch",action="store_true")
    a = p.parse_args()
    OPS.mkdir(parents=True,exist_ok=True)
    with (OPS/"watch.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                if collect() or not a.watch:
                    return
            except (OSError,ValueError,subprocess.CalledProcessError) as e:
                if not a.watch:
                    raise
                print(f"{datetime.now(timezone.utc).isoformat()} {e}",flush=True)
            time.sleep(50)


if __name__=="__main__":
    main()
