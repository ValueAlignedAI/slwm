"""Keep the independently recoverable inference, trajectory and scientific-release lanes running."""

import argparse
import os
from pathlib import Path
import subprocess
from scripts.train.g14_stages import ROOT, SIM, await_files, stage

CPU = str(Path.home() / "dev/slwm-g4/.jax-cpu/bin/python")


def main(lane):
    os.chdir(ROOT / "code")
    os.environ.update(
        PYTHONPATH=".:scripts/data:" + str(ROOT / "upstream/runtime"),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="2",
    )
    work = ROOT / "work"
    metrics = ROOT / "release/metrics"
    await_files([work / "released-lock.json"])
    if lane == "inference":
        subprocess.run([SIM, "-u", "scripts/train/g14_export.py", "fields"], check=True)
        stage(
            "field-replay",
            work / "field-replay.json",
            [CPU, "-u", "scripts/train/g14_field_check.py"],
        )
    elif lane == "missions":
        stage(
            "missions",
            work / "missions/summary.json",
            [SIM, "-u", "scripts/train/g14_missions.py"],
        )
    elif lane == "traces":
        await_files(
            [
                work / "executed/final" / cohort / (str(seed) + ".json")
                for cohort, seed in (
                    ("held", 250000),
                    ("held", 250128),
                    ("fresh", 260000),
                    ("fresh", 260032),
                )
            ]
        )
        stage(
            "native-audit",
            work / "audit-traces/audit.json",
            [SIM, "-u", "scripts/train/g14_trace_audit.py"],
        )
    else:
        await_files(
            [
                work / "executed/final" / name / "complete.json"
                for name in ("fresh", "held")
            ]
        )
        stage(
            "physical-results",
            metrics / "physical-results.json",
            [SIM, "-u", "evals/g14_results.py"],
        )
        await_files(
            [
                work / "missions/summary.json",
                work / "field-replay.json",
                work / "audit-traces/audit.json",
            ]
        )
        stage(
            "two-disk-readback",
            metrics / "custody.json",
            ["python3", "-u", "scripts/train/g14_custody.py", "snapshot"],
        )


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("lane", choices=("inference", "missions", "traces", "release"))
    main(a.parse_args().lane)
