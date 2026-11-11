"""Publish the exact development-selected panel before authorizing the first held physics branch."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time
from g7_integrate_release import git

BRANCH = "training/g14-spoken-closed-loop-20261111"
FOLD = Path("experiments/multimodal/t14/spoken-closed-loop-20261111")
HOST = "db@192.168.1.200"
ROOT = "/mnt/slwm-data/SLWM/g14"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", HOST]


def read(name):
    return subprocess.check_output([*SSH, "cat " + ROOT + "/" + name], timeout=40)


def main(repo):
    os.chdir(repo)
    if git("branch", "--show-current") != BRANCH:
        raise ValueError("Study checkout changed")
    end = time.monotonic() + 3 * 3600
    while time.monotonic() < end:
        r = subprocess.run(
            [*SSH, "test -s " + ROOT + "/work/final-control-lock.json"], timeout=20
        )
        if r.returncode == 0:
            break
        time.sleep(12)
    else:
        raise TimeoutError("Development panel did not finish")
    lock_bytes = read("work/final-control-lock.json")
    lock = json.loads(lock_bytes)
    if (
        len(lock["policies"]) != 22
        or len(lock["selection"]) != 3
        or len(lock["contact_gates"]) != 2
    ):
        raise ValueError("Incomplete held comparison")
    sources = {
        "final-control-lock.json": lock_bytes,
        "paired-training-lock.json": read("data/aggregation-paired-lock.json"),
        "isolated-training-lock.json": read("data/aggregation-isolated-lock.json"),
        "metrics/development-physical.json": read("work/development-physical.json"),
    }
    training = {}
    for stage in (
        "initial",
        "expert-plus",
        "visited-plus",
        "expert-paired",
        "visited-paired",
        "expert-isolated",
        "visited-isolated",
    ):
        raw = read("work/" + stage + "/lock.json")
        training[stage] = dict(
            models=json.loads(raw),
            sha256=sha256(raw).hexdigest(),
            reference=json.loads(read("work/" + stage + "/reference.json")),
        )
    sources["metrics/physical-training.json"] = (
        json.dumps(training, indent=2, sort_keys=True) + "\n"
    ).encode()
    for name, data in sources.items():
        p = FOLD / name
        p.parent.mkdir(exist_ok=True, parents=True)
        tmp = p.with_suffix(".publishing")
        tmp.write_bytes(data)
        tmp.replace(p)
    names = [str(FOLD / n) for n in sources]
    git("add", "--", *names)
    if subprocess.run(["git", "diff", "--quiet", "--cached", "--", *names]).returncode:
        git(
            "commit",
            "--only",
            "-m",
            "Freeze the matched G14 physical policies before all held rollouts",
            "--",
            *names,
        )
    head = git("rev-parse", "HEAD")
    git("push", "origin", "HEAD:refs/heads/" + BRANCH)
    published = git("ls-remote", "origin", "refs/heads/" + BRANCH).split()[0]
    if head != published:
        raise ValueError("Remote freeze verification failed")
    receipt = {
        "revision": head,
        "lock_sha256": sha256(lock_bytes).hexdigest(),
        "policies": len(lock["policies"]),
    }
    args = [
        *SSH,
        "python3 -c 'import sys,json,pathlib; data=json.load(sys.stdin); root=pathlib.Path(\""
        + ROOT
        + '/work"); p=root/".released-lock.new"; p.write_text(json.dumps(data,sort_keys=True));p.replace(root/"released-lock.json")\'',
    ]
    subprocess.run(args, input=json.dumps(receipt), text=True, check=True, timeout=25)
    print("Frozen and published", receipt, flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("repo", type=Path)
    main(a.parse_args().repo)
