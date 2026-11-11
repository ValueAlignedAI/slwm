"""Mirror immutable G11 prepared inputs and queue private Drive uploads separately from training."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from g11_store import roots
from g9_storage import write, sha


def inputs(data):
    a, b = roots("runs")
    banks = data / "banks"
    while not (banks / "effects-ready").exists():
        time.sleep(10)
    names = [
        p for p in banks.iterdir() if p.is_file() and p.suffix in (".npz", ".json")
    ]
    names += [
        data / "groups.json",
        data / "annotations.json",
        data / "fine_grained_actions_map.txt",
        data / "fresh/complete.json",
        data / "geometry/complete",
    ]
    stamp = {}
    for p in names:
        name = str(p.relative_to(data))
        h = sha(p)
        for r in (a.parent, b.parent):
            dest = r / "inputs" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.is_file() or sha(dest) != h:
                tmp = dest.with_name("." + dest.name + ".copying")
                shutil.copyfile(p, tmp)
                if sha(tmp) != h:
                    raise ValueError("Input archive mismatch")
                tmp.replace(dest)
            if sha(dest) != h:
                raise ValueError("Input readback changed")
        stamp[name] = dict(sha256=h, bytes=p.stat().st_size)
    for r in (a.parent, b.parent):
        write(r / "inputs/verified.json", stamp)
    print(
        "G11 verified inputs",
        len(stamp),
        sum(v["bytes"] for v in stamp.values()),
        flush=True,
    )


def drive():
    from colab_artifacts import put_verified

    a, _ = roots("runs")
    root = a.parent
    os.environ["PATH"] = str(Path.home() / ".local/bin") + ":" + os.environ["PATH"]
    os.environ.setdefault(
        "RCLONE_CONFIG", str(Path.home() / ".config/slwm-colab/rclone.conf")
    )
    record = root / "drive-verified.json"
    sent = json.loads(record.read_text()) if record.exists() else {}
    delay = 180
    until = time.time() + 36 * 3600
    while time.time() < until:
        files = []
        for group in ("runs", "scale", "scale-parents"):
            for f in sorted((root / group).glob("*/step-*/verified.json")):
                if f.parent.parent.is_symlink():
                    continue
                d = json.loads(f.read_text())
                for n, k in (
                    ("weights.npz", "inference_sha256"),
                    ("optimizer.npz", "optimizer_sha256"),
                ):
                    p = f.parent / n
                    files.append((p, str(p.relative_to(root)), d[k]))
        for sub in ("inference", "inputs"):
            p = root / sub / "verified.json"
            if p.is_file():
                for name, meta in json.loads(p.read_text()).items():
                    files.append((p.parent / name, sub + "/" + name, meta["sha256"]))
        pending = [v for v in files if sent.get(v[1]) != v[2]]
        if not pending:
            write(
                root / "drive-state.json",
                dict(phase="current", verified_files=len(sent), pending=0),
            )
            if (root / "finalized").is_file():
                return
            time.sleep(45)
            continue
        defer = 0
        for old in ("g7", "g8", "g9", "g10"):
            state = root.parent / old / "drive-state.json"
            if state.is_file():
                d = json.loads(state.read_text())
                if "query_limit" in d.get("reason", ""):
                    defer = max(defer, d.get("retry_unix", 0))
        if defer > time.time():
            write(
                root / "drive-state.json",
                dict(
                    phase="backoff",
                    reason="shared_drive_project_query_limit",
                    retry_unix=defer,
                    verified_files=len(sent),
                    pending=len(pending),
                ),
            )
            time.sleep(min(120, max(1, defer - time.time())))
            continue
        try:
            p, name, h = pending[0]
            subprocess.run(
                [
                    "rclone",
                    "lsf",
                    "--dirs-only",
                    "--max-depth",
                    "1",
                    "--retries",
                    "1",
                    "--low-level-retries",
                    "1",
                    "--timeout",
                    "15s",
                    "slwm-drive:SLWM",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=35,
            )
            put_verified(p, "slwm-drive:SLWM/g11/objects/" + h, h)
            sent[name] = h
            write(record, sent)
            d = sha(record)
            put_verified(record, "slwm-drive:SLWM/g11/inventories/" + d + ".json", d)
            write(
                root / "drive-state.json",
                dict(
                    phase="publishing",
                    verified_files=len(sent),
                    pending=len(pending) - 1,
                ),
            )
            delay = 180
        except (OSError, RuntimeError, subprocess.SubprocessError) as ex:
            msg = str(getattr(ex, "stderr", "")).lower()
            reason = (
                "drive_project_query_limit"
                if "quota" in msg or "ratelimit" in msg
                else type(ex).__name__
            )
            write(
                root / "drive-state.json",
                dict(
                    phase="backoff",
                    reason=reason,
                    retry_unix=time.time() + delay,
                    verified_files=len(sent),
                    pending=len(pending),
                ),
            )
            time.sleep(delay)
            delay = min(1800, delay * 2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("inputs", "drive"))
    p.add_argument("--data", type=Path, default=Path("/mnt/slwm-data/SLWM/g11/data"))
    a = p.parse_args()
    inputs(a.data) if a.mode == "inputs" else drive()
