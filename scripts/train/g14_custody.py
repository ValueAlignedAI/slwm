"""Read back G14 checkpoints on both registered drives and transfer content-addressed objects asynchronously."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
from g9_storage import sha, write, disks
from g13_custody import copy_file

ROOT = Path("/mnt/slwm-data/SLWM/g14")
STEPS = {
    "cue": (9, (128, 640, 1536)),
    "cue-semantic": (6, (128, 640, 1536)),
    "fields": (60, (96, 384, 1024)),
}


def bases():
    return tuple(p.parent.parent / "g14" for p in disks())


def archive(group, items, volumes):
    records = {}
    for name, p in sorted(items.items()):
        if p.is_symlink() or not p.is_file():
            raise ValueError("Missing required scientific file: " + name)
        h = sha(p)
        for b in volumes:
            copy_file(p, b / group / name, h)
        records[name] = dict(sha256=h, bytes=p.stat().st_size)
    for b in volumes:
        write(b / group / "verified.json", records)
    return len(records)


def inventory(complete=True):
    a, b = bases()

    def verify(base):
        result = {}
        for name, (count, cuts) in STEPS.items():
            dirs = sorted(d for d in (base / name).glob("*") if d.is_dir())
            if complete and len(dirs) != count:
                raise ValueError(f"{name} has {len(dirs)} runs, expected {count}")
            for d in dirs:
                for step in cuts:
                    folder = d / f"step-{step:05d}"
                    rec = json.loads((folder / "verified.json").read_bytes())
                    if rec["tag"] != d.name or rec["step"] != step:
                        raise ValueError("Wrong generation")
                    for k, h in rec["sha256"].items():
                        if sha(folder / k) != h:
                            raise ValueError("Checkpoint content differs")
                    result[str(folder.relative_to(base))] = rec["sha256"]
        return result

    with ThreadPoolExecutor(2) as pool:
        x, y = pool.map(verify, (a, b))
    if x != y:
        raise ValueError("Independent disk checksums disagree")
    return x


def snapshot():
    volumes = bases()
    data = ROOT / "data"
    work = ROOT / "work"
    out = ROOT / "release/metrics"
    out.mkdir(exist_ok=True, parents=True)
    x = inventory()
    write(out / "checkpoint-inventory.json", x)
    banks = {p.name: p for p in data.glob("*.npz")}
    banks.update({p.name: p for p in data.glob("*.json")})
    for sub in ("utterances", "semantic-utterances", "cases", "cases-base", "grid-new"):
        for p in (data / sub).iterdir():
            if p.is_file() and p.suffix in (".npz", ".json", ".zip", ".tar"):
                banks[str(p.relative_to(data))] = p
    transfer = {p.name: p for p in work.glob("*.json")}
    transfer["telemetry-coefficients.npz"] = work / "telemetry-coefficients.npz"
    for n in (
        "cue",
        "cue-semantic",
        "initial",
        "expert-plus",
        "visited-plus",
        "expert-paired",
        "visited-paired",
        "expert-isolated",
        "visited-isolated",
    ):
        for p in (work / n).iterdir():
            if p.is_file() and p.suffix in (".npz", ".json"):
                transfer[str(p.relative_to(work))] = p
    assets = {p.name: p for p in (ROOT / "release/inference").glob("*") if p.is_file()}
    physical = {
        str(p.relative_to(work)): p for p in (work / "executed").rglob("*.json")
    }
    for p in (work / "missions").iterdir():
        if p.is_file():
            physical[str(p.relative_to(work))] = p
    for p in (work / "audit-traces").glob("*"):
        if p.is_file():
            physical[str(p.relative_to(work))] = p
    counts = {
        k: archive(k, v, volumes)
        for k, v in (
            ("inputs", banks),
            ("transfer", transfer),
            ("inference", assets),
            ("executions", physical),
        )
    }
    receipt = dict(
        generations=len(x),
        groups={
            k: dict(fits=c, generations=c * len(steps))
            for k, (c, steps) in STEPS.items()
        },
        copies=2,
        verified_objects=counts,
        checkpoint_manifest_sha256=sha(out / "checkpoint-inventory.json"),
    )
    write(out / "custody.json", receipt)
    print("Independent disk readback", receipt, flush=True)


def publish(revision):
    import re

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Invalid revision")
    file = ROOT / "staging" / ("g14-" + revision + ".tar.gz")
    with tarfile.open(file, "r:gz") as a:
        names = a.getnames()
        if (
            not any(p.endswith("/metrics/release.json") for p in names)
            or not any(p.endswith("/report.md") for p in names)
            or a.pax_headers.get("comment") != revision
        ):
            raise ValueError("Study archive missing landmarks or Git identity")
    volumes = bases()
    archive("published", {file.name: file}, volumes)
    for b in volumes:
        (b / "finalized").touch()
    print("Mirrored Git release", revision, sha(file), flush=True)


def drive():
    from colab_artifacts import put_verified

    base = bases()[0]
    state = base / "drive-state.json"
    index = base / "drive-verified.json"
    os.environ["PATH"] = str(Path.home() / ".local/bin") + ":" + os.environ["PATH"]
    os.environ.setdefault(
        "RCLONE_CONFIG", str(Path.home() / ".config/slwm-colab/rclone.conf")
    )
    sent = json.loads(index.read_bytes()) if index.exists() else {}
    backoff = 180
    while True:
        objects = {}
        for group in ("inference", "published", "transfer", "inputs", "executions"):
            p = base / group / "verified.json"
            if p.is_file():
                objects.update(
                    {
                        group + "/" + k: v["sha256"]
                        for k, v in json.loads(p.read_bytes()).items()
                    }
                )
        for kind in STEPS:
            for f in sorted((base / kind).glob("*/step-*/verified.json")):
                for k, h in json.loads(f.read_bytes())["sha256"].items():
                    objects[str((f.parent / k).relative_to(base))] = h
        pending = [(k, h) for k, h in objects.items() if sent.get(k) != h]

        def note(phase, **kw):
            write(
                state,
                dict(phase=phase, pending=len(pending), verified_files=len(sent), **kw),
            )

        if not pending:
            note("current")
            if (base / "finalized").exists():
                return
            time.sleep(45)
            continue
        deferred = []
        for seq in range(7, 14):
            other = base.parent / f"g{seq}" / "drive-state.json"
            if other.is_file():
                r = json.loads(other.read_bytes())
                if "query_limit" in r.get("reason", ""):
                    deferred.append(r.get("retry_unix", 0))
        until = max(deferred + [0])
        if until > time.time():
            note("backoff", reason="shared_drive_project_query_limit", retry_unix=until)
            time.sleep(min(60, max(1, until - time.time())))
            continue
        try:
            # One object at a time leaves the SSD and the control loop independent of Drive latency.
            key, h = pending[0]
            put_verified(base / key, "slwm-drive:SLWM/g14/objects/" + h, h)
            sent[key] = h
            write(index, sent)
            ih = sha(index)
            put_verified(index, "slwm-drive:SLWM/g14/inventories/" + ih + ".json", ih)
            backoff = 180
            note("transferring")
        except (subprocess.SubprocessError, RuntimeError, OSError) as ex:
            msg = str(getattr(ex, "stderr", "")).lower()
            reason = (
                "drive_project_query_limit"
                if "quota" in msg or "ratelimit" in msg
                else type(ex).__name__
            )
            note("backoff", reason=reason, retry_unix=time.time() + backoff)
            time.sleep(backoff)
            backoff = min(1800, 2 * backoff)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("stage", choices=("snapshot", "publish", "drive"))
    a.add_argument("--revision")
    arg = a.parse_args()
    if arg.stage == "publish":
        publish(arg.revision)
    else:
        (snapshot if arg.stage == "snapshot" else drive)()
