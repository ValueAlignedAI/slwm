"""Observe the existing G3 trainer's atomic optimizer saves without modifying the training lock."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import torch

from colab_artifacts import capture, sha256


def stage_optimizers(output, config, spool, seen, prefix):
    cfg = json.loads(config.read_text())
    training_lock = output / "training_lock.json"
    lock_sha = sha256(training_lock)
    for profile, settings in cfg["profiles"].items():
        for seed in cfg["seeds"]:
            name = f"{profile}-{seed}"
            run = f"{prefix}-{name}"
            source = output / name / "resume.pt"
            if not source.is_file():
                continue
            # Keep one pending immutable save per run during a storage outage.
            if any((spool / "pending" / run).glob("[0-9]*")):
                continue
            stat = source.stat()
            stamp = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            if seen.get(run) == stamp:
                continue
            with tempfile.TemporaryDirectory(dir=spool, prefix=".pin-") as temp:
                pin = Path(temp) / "resume.pt"
                os.link(source, pin)
                checkpoint = torch.load(pin, mmap=True, map_location="cpu", weights_only=True)
                step = checkpoint["step"]
                parent = cfg["parents"][f"{settings['core']}-{seed}"][1]
                if (checkpoint["name"], checkpoint["seed"], checkpoint["lock"], checkpoint["parent"]) != (
                        profile, seed, lock_sha, parent):
                    raise ValueError(f"Wrong identity for {run}/{step}")
                if not checkpoint["opt"]["state"] or not checkpoint["delta"]:
                    raise ValueError(f"Missing training state for {run}/{step}")
                if not any((spool / "receipts" / run).glob(f"{step:012d}-*.json")):
                    capture(spool, run, step, {"resume.pt": pin, "training_lock.json": training_lock,
                                               "joint_transfer.json": config})
                    print(f"staged {run} step={step}", flush=True)
                del checkpoint
            seen[run] = stamp


def stage_runs(root, spool, seen, checkpoint_root=None):
    config = root / "configs/t4/joint_transfer.json"
    cfg = json.loads(config.read_text())
    logical = root / cfg["output"]
    output = checkpoint_root or logical
    if not output.samefile(logical):
        raise ValueError("Physical checkpoint root differs from the registered trainer output")
    stage_optimizers(output, config, spool, seen, "g3")
    for group in ("depth6", "structured"):
        variant = output/group
        if (variant/"config.json").exists() and (variant/"training_lock.json").exists():
            stage_optimizers(variant, variant/"config.json", spool, seen, f"g3-{group}")
    # Each completed comparison is also an inference package. Receipts from
    # these groups let the Mac publish results without downloading the weights.
    for folder, prefix in (("", "summary"), ("depth6", "depth"), ("structured", "structured")):
        source = output / folder
        run = f"g3-{folder or 'main'}-exports"
        for step in cfg["milestones"]:
            role = "test" if step == cfg["milestones"][-1] else "dev"
            report = source / f"{prefix}-{step}-{role}.json"
            if (not report.is_file() or (spool / "pending" / run / f"{step:012d}").exists()
                    or any((spool / "receipts" / run).glob(f"{step:012d}-*.json"))):
                continue
            files = {"summary.json": report, "training_lock.json": source / "training_lock.json",
                     "config.json": source / "config.json" if folder else config}
            for name in json.loads(report.read_text())["rows"]:
                files[f"{name}.pt"] = source / name / f"step-{step}.pt"
                files[f"{name}.json"] = source / name / f"step-{step}.json"
                for head in (source / name / f"eval-{step}").glob("*-32shot-head.npz"):
                    files[f"{name}-{head.name}"] = head
            capture(spool, run, step, files)
            print(f"staged {run} step={step}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--spool", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, help="Same mounted path as the spool for zero-copy links")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    args.spool.mkdir(parents=True, exist_ok=True)
    seen = {}
    while True:
        stage_runs(args.repo, args.spool, seen, args.checkpoint_root)
        if not args.watch:
            break
        time.sleep(45)
