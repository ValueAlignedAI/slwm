"""Resume verified G14 stage outputs on the research SSD and record each child's progress in tmux."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import os
from concurrent.futures import ThreadPoolExecutor
from scripts.train.g9_storage import write

ROOT = Path("/mnt/slwm-data/SLWM/g14")
SIM = str(Path.home() / ".local/state/slwm-native/g14-py311/bin/python")
GPU = ["flock", "/tmp/slwm_research_gpu.lock", "bash", "scripts/train/g4_cuda.sh"]


def await_files(files):
    while not all(p.is_file() for p in files):
        time.sleep(8)


def stage(name, goal, args):
    if goal.is_file():
        return name
    logs = ROOT / "logs"
    for attempt in range(3):
        with (logs / (name + ".log")).open("a") as out:
            print("Executing", name, attempt + 1, flush=True)
            run = subprocess.run(args, stdout=out, stderr=subprocess.STDOUT)
        if run.returncode == 0 and goal.is_file():
            return name
        time.sleep(5)
    raise RuntimeError("Stage failed; see " + str(logs / (name + ".log")))


def main(lane):
    os.environ.update(
        PYTHONPATH=".:scripts/data:" + str(ROOT / "upstream/runtime"),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="2",
    )
    work = ROOT / "work"
    data = ROOT / "data"
    os.chdir(ROOT / "code")
    if lane == "development":
        await_files(
            [data / (x + ".npz") for x in ("fresh-teacher", "visited", "value-dev")]
        )
        for n in ("expert-plus", "visited-plus"):
            stage(n, work / n / "lock.json", GPU + ["scripts/train/g14_continue.py", n])
        stage(
            "refine-dev",
            work / "executed/refine/dev/complete.json",
            [
                SIM,
                "-u",
                "scripts/train/g14_execute.py",
                "dev",
                "refine",
                "--workers",
                "12",
            ],
        )
        stage(
            "stage-choice",
            work / "development-selection.json",
            [SIM, "-u", "scripts/train/g14_control_lock.py", "--preliminary"],
        )
        stage(
            "contact-dev",
            work / "executed/gate/dev/complete.json",
            [
                SIM,
                "-u",
                "scripts/train/g14_execute.py",
                "dev",
                "gate",
                "--workers",
                "12",
            ],
        )
        stage(
            "final-lock",
            work / "final-control-lock.json",
            [SIM, "-u", "scripts/train/g14_control_lock.py"],
        )
    else:
        await_files([work / "final-control-lock.json", work / "released-lock.json"])
        if (
            json.loads((work / "released-lock.json").read_bytes())["lock_sha256"]
            != __import__("hashlib")
            .sha256((work / "final-control-lock.json").read_bytes())
            .hexdigest()
        ):
            raise ValueError("Publication does not match selected policies")

        def execute(item):
            name, n = item
            return stage(
                "final-" + name,
                work / "executed/final" / name / "complete.json",
                [
                    SIM,
                    "-u",
                    "scripts/train/g14_execute.py",
                    name,
                    "final",
                    "--workers",
                    str(n),
                ],
            )

        with ThreadPoolExecutor(2) as pool:
            list(pool.map(execute, [("held", 11), ("fresh", 3)]))
    write(work / (lane + "-finished.json"), dict(lane=lane, unix=time.time()))
    print("Finished", lane, flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("lane", choices=("development", "held"))
    main(a.parse_args().lane)
