"""Verify G13 science artifacts on both mounted HDDs and transfer asynchronously to the scoped Drive."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from g9_storage import disks, sha, write

ROOT = Path("/mnt/slwm-data/SLWM/g13")
COUNTS = {
    "robot": (60, (128, 512, 1024)),
    "robot-corrected": (62, (128, 512, 1024)),
    "speech": (6, (128, 512, 1024)),
    "speech-portable": (6, (128, 512, 1024)),
    "components": (48, (32, 96, 256, 512)),
}


def bases():
    return tuple(r.parent.parent / "g13" for r in disks())


def copy_file(p, d, h):
    d.parent.mkdir(exist_ok=True, parents=True)
    if not d.exists() or sha(d) != h:
        tmp = d.with_name("." + d.name + ".copying")
        shutil.copyfile(p, tmp)
        if sha(tmp) != h:
            raise ValueError("Archive transfer mismatch")
        tmp.replace(d)
    if sha(d) != h:
        raise ValueError("Readback error")


def snapshot():
    banks = []
    for group in ("robot", "robot-extra", "robot-prospective", "robot-readout"):
        root = ROOT / "data" / group
        banks.extend(
            p
            for folder in ("banks", "corrected")
            for p in (root / folder).glob("*")
            if p.suffix in (".npz", ".json")
        )
        banks.extend(
            root / "raw" / n
            for n in (
                "selection.json",
                "sources.json",
                "features.json",
                "dataset_info.json",
            )
        )
    for name in ("holo", "components"):
        root = ROOT / "data" / name
        banks.extend(
            p for p in (root / "banks").glob("*") if p.suffix in (".npz", ".json")
        )
        banks.extend(
            root / f
            for f in (
                "groups.json",
                "records.json",
                "annotations.json",
                "fresh/complete.json",
            )
        )
    banks += list((ROOT / "data/components/component-banks").glob("*.npz")) + list(
        (ROOT / "data/components/component-banks").glob("*.json")
    )
    banks += [
        ROOT / "data/holo/extra-language.json",
        ROOT / "data/speech/directions.npz",
        ROOT / "data/speech/directions.json",
        ROOT / "data/components/minilm/source.json",
    ]
    transfers = [
        *sorted((ROOT / "work/voice-portable/probes").glob("*.npz")),
        *sorted((ROOT / "work/voice/probes").glob("*.npz")),
        *sorted((ROOT / "work/direct/probes").glob("*.npz")),
        ROOT / "work/prospective/raw-action.npz",
        ROOT / "work/robot/system-id.npz",
        ROOT / "work/repaired/system-id.npz",
        ROOT / "work/repaired/adaptive.npz",
    ]
    assets = sorted(p for p in (ROOT / "release/inference").glob("*") if p.is_file())
    work = {}
    for group, files, base in (
        ("inputs", banks, ROOT / "data"),
        ("transfer", transfers, ROOT / "work"),
        ("inference", assets, ROOT / "release/inference"),
    ):
        manifest = {}
        for p in files:
            if not p.is_file():
                raise ValueError("Prepared input missing " + str(p))
            name = str(p.relative_to(base))
            h = sha(p)
            for root in bases():
                copy_file(p, root / group / name, h)
            manifest[name] = dict(sha256=h, bytes=p.stat().st_size)
        for root in bases():
            write(root / group / "verified.json", manifest)
        work[group] = len(manifest)
    write(ROOT / "work/storage.json", work)
    print("Verified input, transfer and inference objects", work, flush=True)


def verify(root):
    all_hashes = {}
    for group, (total, steps) in COUNTS.items():
        folders = [f for f in (root / group).iterdir() if f.is_dir()]
        if len(folders) != total:
            raise ValueError(f"Missing {group} runs: {len(folders)}")
        for run in sorted(folders):
            for step in steps:
                stage = run / f"step-{step:05d}"
                r = json.loads((stage / "verified.json").read_bytes())
                if r["tag"] != run.name or r["step"] != step:
                    raise ValueError("Checkpoint identity mismatch")
                for key, h in r["sha256"].items():
                    if sha(stage / key) != h:
                        raise ValueError("Mutated checkpoint " + str(stage / key))
                all_hashes[str(stage.relative_to(root))] = r["sha256"]
    for g in ("inputs", "transfer", "inference"):
        for name, r in json.loads((root / g / "verified.json").read_bytes()).items():
            if sha(root / g / name) != r["sha256"]:
                raise ValueError("Changed verified data or inference")
    return all_hashes


def verify_all():
    a, b = bases()
    with ThreadPoolExecutor(2) as pool:
        x, y = pool.map(verify, (a, b))
    if x != y or len(x) != 594:
        raise ValueError("Unmatched independent disk inventories")
    write(ROOT / "release/metrics/checkpoints.json", x)
    print("Independent checkpoint readback", len(x), flush=True)


def publish(revision):
    """Keep the exact Git experiment tree with the weights and queue it for the same verified Drive uploader."""
    import re
    import tarfile

    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("Expected a full Git revision")
    file = ROOT / "staging" / ("g13-results-" + revision + ".tar.gz")
    with tarfile.open(file, "r:gz") as archive:
        names = set(archive.getnames())
        report = "experiments/multimodal/t13/grounded-transfer-20261111/report.md"
        if (
            report not in names
            or report.replace("report.md", "metrics/release.json") not in names
        ):
            raise ValueError("Incomplete public results archive")
    h = sha(file)
    for root in bases():
        dest = root / "published"
        copy_file(file, dest / file.name, h)
        write(
            dest / "verified.json",
            {file.name: dict(sha256=h, bytes=file.stat().st_size, commit=revision)},
        )
    print("Published tree mirrored", revision, h, flush=True)


def drive():
    from colab_artifacts import put_verified

    base = bases()[0]
    os.environ["PATH"] = str(Path.home() / ".local/bin") + ":" + os.environ["PATH"]
    os.environ.setdefault(
        "RCLONE_CONFIG", str(Path.home() / ".config/slwm-colab/rclone.conf")
    )
    rec = base / "drive-verified.json"
    sent = json.loads(rec.read_bytes()) if rec.exists() else {}
    delay = 180
    while True:
        objects = {}
        for group in COUNTS:
            for f in sorted((base / group).glob("*/step-*/verified.json")):
                for name, h in json.loads(f.read_bytes())["sha256"].items():
                    objects[str((f.parent / name).relative_to(base))] = h
        for group in ("inputs", "transfer", "inference", "published"):
            f = base / group / "verified.json"
            if f.exists():
                for n, m in json.loads(f.read_bytes()).items():
                    objects[group + "/" + n] = m["sha256"]
        pending = [(k, h) for k, h in objects.items() if sent.get(k) != h]

        def note(phase, **kw):
            write(
                base / "drive-state.json",
                dict(phase=phase, pending=len(pending), verified_files=len(sent), **kw),
            )

        if not pending:
            note("current")
            if (base / "finalized").exists():
                return
            time.sleep(40)
            continue
        waits = []
        for previous in ("g7", "g8", "g9", "g10", "g11", "g12"):
            f = base.parent / previous / "drive-state.json"
            if f.exists():
                d = json.loads(f.read_bytes())
                if "query_limit" in d.get("reason", ""):
                    waits.append(d.get("retry_unix", 0))
        defer = max(waits + [0])
        if defer > time.time():
            note("backoff", reason="shared_drive_project_query_limit", retry_unix=defer)
            time.sleep(min(60, max(1, defer - time.time())))
            continue
        try:
            name, h = pending[0]
            put_verified(base / name, "slwm-drive:SLWM/g13/objects/" + h, h)
            sent[name] = h
            write(rec, sent)
            rh = sha(rec)
            put_verified(rec, "slwm-drive:SLWM/g13/inventories/" + rh + ".json", rh)
            pending = pending[1:]
            note("transferring")
            delay = 180
        except (subprocess.SubprocessError, RuntimeError, OSError) as exc:
            msg = str(getattr(exc, "stderr", "")).lower()
            reason = (
                "drive_project_query_limit"
                if "quota" in msg or "ratelimit" in msg
                else type(exc).__name__
            )
            note("backoff", reason=reason, retry_unix=time.time() + delay)
            time.sleep(delay)
            delay = min(delay * 2, 1800)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("snapshot", "verify", "drive", "publish"))
    p.add_argument("--revision")
    a = p.parse_args()
    if a.stage == "publish":
        publish(a.revision)
    else:
        {"snapshot": snapshot, "verify": verify_all, "drive": drive}[a.stage]()
