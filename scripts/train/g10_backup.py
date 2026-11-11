"""Verify prepared public data on both research disks, then back off private Drive independently."""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from g10_store import roots
from g9_storage import sha, write


def local_data():
    a, b = roots()
    source = Path("/mnt/slwm-data/SLWM/g10/data/holo")
    required = ["labels.json", "split.zip", "selection.json", "banks.json"]
    required += [
        "holo-" + role + suffix
        for role in ("train", "dev", "familiar", "novel")
        for suffix in (".npz", ".json")
    ]
    required += [
        name + "-index.json"
        for name in ("imu", "video_compress", "video_pitch_shifted")
    ]
    while not all((source / name).is_file() for name in required):
        time.sleep(15)
    rows = {}
    for name in required:
        src = source / name
        digest = sha(src)
        if name.startswith("holo-") and name.endswith(".npz"):
            if digest != json.loads(src.with_suffix(".json").read_text())["sha256"]:
                raise ValueError("Prepared source identity mismatch")
        for volume in (a.parent, b.parent):
            target = volume / "inputs" / name
            target.parent.mkdir(exist_ok=True, parents=True)
            if not target.exists() or sha(target) != digest:
                tmp = target.with_suffix(".copying")
                shutil.copyfile(src, tmp)
                if sha(tmp) != digest:
                    raise ValueError("Prepared data copy mismatch")
                tmp.replace(target)
        rows[name] = dict(bytes=src.stat().st_size, sha256=digest)
    for volume in (a.parent, b.parent):
        write(volume / "inputs/verified.json", rows)
    print(
        "Two copies of",
        len(rows),
        "input objects",
        sum(v["bytes"] for v in rows.values()),
        flush=True,
    )


def drive():
    from colab_artifacts import put_verified

    a, _ = roots()
    root = a.parent
    dest = "slwm-drive:SLWM/g10"
    os.environ["PATH"] = str(Path.home() / ".local/bin") + ":" + os.environ["PATH"]
    os.environ.setdefault(
        "RCLONE_CONFIG", str(Path.home() / ".config/slwm-colab/rclone.conf")
    )
    inventory = root / "drive-verified.json"
    sent = json.loads(inventory.read_text()) if inventory.exists() else {}
    delay = 120
    until = time.time() + 30 * 3600
    while time.time() < until:
        items = []
        for p in sorted(
            a.glob("*/step-*/verified.json"),
            key=lambda x: (x.name != "verified.json", x.parent.name),
        ):
            data = json.loads(p.read_text())
            for file, field in (
                ("weights.npz", "inference_sha256"),
                ("optimizer.npz", "optimizer_sha256"),
            ):
                target = p.parent / file
                items.append((target, str(target.relative_to(root)), data[field]))
        for folder in ("inputs", "public-inference", "risk-heads"):
            manifest = root / folder / "verified.json"
            if manifest.exists():
                for name, attr in json.loads(manifest.read_text()).items():
                    target = root / folder / name
                    items.append((target, folder + "/" + name, attr["sha256"]))
        pending = [v for v in items if sent.get(v[1]) != v[2]]
        if not pending:
            write(
                root / "drive-state.json",
                dict(phase="current", verified_files=len(sent), pending=0),
            )
            if (root / "finalized").exists():
                return
            time.sleep(32)
            continue
        deferred = 0
        for study in ("g7", "g8", "g9"):
            other = root.parent / study / "drive-state.json"
            if other.exists():
                d = json.loads(other.read_text())
                if "query_limit" in d.get("reason", ""):
                    deferred = max(deferred, d.get("retry_unix", 0))
        if deferred > time.time():
            write(
                root / "drive-state.json",
                dict(
                    phase="backoff",
                    reason="shared_drive_project_query_limit",
                    retry_unix=deferred,
                    verified_files=len(sent),
                    pending=len(pending),
                ),
            )
            time.sleep(min(120, max(1, deferred - time.time())))
            continue
        try:
            path, name, h = pending[0]
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
                capture_output=True,
                text=True,
                timeout=35,
                check=True,
            )
            put_verified(path, dest + "/objects/" + h, h)
            sent[name] = h
            write(inventory, sent)
            pin = sha(inventory)
            put_verified(inventory, dest + "/inventories/" + pin + ".json", pin)
            write(
                root / "drive-state.json",
                dict(
                    phase="publishing",
                    verified_files=len(sent),
                    pending=len(pending) - 1,
                ),
            )
            delay = 120
        except (OSError, RuntimeError, subprocess.SubprocessError) as e:
            detail = str(getattr(e, "stderr", "")).lower()
            reason = (
                "drive_project_query_limit"
                if "quota" in detail or "ratelimit" in detail
                else type(e).__name__
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
            delay = min(1800, 2 * delay)


if __name__ == "__main__":
    import sys

    if sys.argv[1] == "data":
        local_data()
    elif sys.argv[1] == "drive":
        drive()
    else:
        raise ValueError("Expected data or drive")
