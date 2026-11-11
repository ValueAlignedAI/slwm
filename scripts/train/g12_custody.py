"""Mirror reusable data on physical disks and import fully checked native TPU generations."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from g9_storage import disks, sha, write

ROOT = Path("/mnt/slwm-data/SLWM/g12")


def vaults():
    return tuple(p.parent.parent / "g12" for p in disks())


def copy_checked(file, to, expected):
    to.parent.mkdir(exist_ok=True, parents=True)
    if not to.exists() or sha(to) != expected:
        tmp = to.with_name("." + to.name + ".g12-copy")
        shutil.copyfile(file, tmp)
        if sha(tmp) != expected:
            raise ValueError("New copy failed hash verification")
        tmp.replace(to)
    if sha(to) != expected:
        raise ValueError("Archive mismatch")


def inputs():
    root = ROOT / "data"
    if not (root / "banks/ready").is_file():
        raise ValueError("Input source incomplete")
    names = [p for p in (root / "banks").iterdir() if p.suffix in (".npz", ".json")]
    names.extend(
        root / name
        for name in (
            "groups.json",
            "records.json",
            "annotations.json",
            "state-labels.json",
            "panel.json",
            "sheets/index.json",
            "fresh/complete.json",
            "vision-nv/receipt.json",
        )
    )
    manifest = {}
    vault = vaults()
    for p in names:
        name = str(p.relative_to(root))
        h = sha(p)
        for dst in vault:
            copy_checked(p, dst / "inputs" / name, h)
        manifest[name] = dict(sha256=h, bytes=p.stat().st_size)
    for dst in vault:
        write(dst / "inputs/verified.json", manifest)
    print(
        "Verified prepared data objects on independent disks", len(manifest), flush=True
    )


def import_tpu(step):
    if step not in (256, 512, 1024):
        raise ValueError("Invalid generation")
    for f in ("A", "B"):
        for c in ("gpt2", "gated", "oscillator", "associative"):
            tag = f"{f}-{c}-paired-1202"
            rel = Path(tag) / f"step-{step:05d}"
            src = ROOT / "tpu-in" / rel
            doc = json.loads((src / "verified.json").read_bytes())
            cfg = json.loads((src / "config.json").read_bytes())
            if (
                doc["step"],
                doc["tag"],
                cfg["fold"],
                cfg["core"],
                cfg["seed"],
                cfg["spec"]["width"],
            ) != (step, tag, f, c, 1202, 256) or "v6" not in cfg["device_kind"].lower():
                raise ValueError("Unregistered TPU result")
            if cfg["train_sha"] != sha(ROOT / "data/banks" / (f + "-train.npz")) or (
                step > 256 and cfg["state_sha"] != sha(ROOT / "data/state-labels.json")
            ):
                raise ValueError("The TPU used different inputs")
            if set(doc["sha256"]) != {"weights.npz", "optimizer.npz", "config.json"}:
                raise ValueError("Incomplete optimizer")
            for dest in vaults():
                stage = dest / "scale" / rel
                if (stage / "verified.json").exists() and json.loads(
                    (stage / "verified.json").read_bytes()
                ) != doc:
                    raise ValueError("Conflicting existing generation")
                for name, h in doc["sha256"].items():
                    copy_checked(src / name, stage / name, h)
                write(stage / "verified.json", doc)
    print("Imported matched native optimizer generations", step, flush=True)


def drive():
    from colab_artifacts import put_verified

    base = vaults()[0]
    os.environ["PATH"] = str(Path.home() / ".local/bin") + ":" + os.environ["PATH"]
    os.environ.setdefault(
        "RCLONE_CONFIG", str(Path.home() / ".config/slwm-colab/rclone.conf")
    )
    record = base / "drive-verified.json"
    sent = json.loads(record.read_bytes()) if record.exists() else {}
    delay = 180
    while True:
        registered = {}
        for group in ("runs", "scale", "early"):
            for file in (base / group).glob("*/step-*/verified.json"):
                receipt = json.loads(file.read_bytes())
                for name, h in receipt["sha256"].items():
                    registered[str((file.parent / name).relative_to(base))] = h
        for group in ("inputs", "inference"):
            receipt = base / group / "verified.json"
            if receipt.is_file():
                for name, r in json.loads(receipt.read_bytes()).items():
                    registered[group + "/" + name] = r["sha256"]
        pending = [(n, h) for n, h in registered.items() if sent.get(n) != h]
        if not pending:
            write(
                base / "drive-state.json",
                dict(phase="current", verified_files=len(sent), pending=0),
            )
            if (base / "finalized").exists():
                return
            time.sleep(30)
            continue
        defer = 0
        for study in ("g7", "g8", "g9", "g10", "g11"):
            prev = base.parent / study / "drive-state.json"
            if prev.exists():
                d = json.loads(prev.read_bytes())
                if "query_limit" in d.get("reason", ""):
                    defer = max(defer, d.get("retry_unix", 0))
        if time.time() < defer:
            write(
                base / "drive-state.json",
                dict(
                    phase="backoff",
                    verified_files=len(sent),
                    pending=len(pending),
                    reason="shared_drive_project_query_limit",
                    retry_unix=defer,
                ),
            )
            time.sleep(min(60, max(1, defer - time.time())))
            continue
        try:
            name, h = pending[0]
            put_verified(base / name, "slwm-drive:SLWM/g12/objects/" + h, h)
            sent[name] = h
            write(record, sent)
            digest = sha(record)
            put_verified(
                record, "slwm-drive:SLWM/g12/inventories/" + digest + ".json", digest
            )
            write(
                base / "drive-state.json",
                dict(
                    phase="publishing",
                    verified_files=len(sent),
                    pending=len(pending) - 1,
                ),
            )
            delay = 180
        except (subprocess.SubprocessError, OSError, RuntimeError) as exc:
            msg = str(getattr(exc, "stderr", "")).lower()
            reason = (
                "drive_project_query_limit"
                if "quota" in msg or "ratelimit" in msg
                else type(exc).__name__
            )
            write(
                base / "drive-state.json",
                dict(
                    phase="backoff",
                    verified_files=len(sent),
                    pending=len(pending),
                    reason=reason,
                    retry_unix=time.time() + delay,
                ),
            )
            time.sleep(delay)
            delay = min(1800, delay * 2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("inputs", "drive", "import-tpu"))
    p.add_argument("--step", type=int)
    a = p.parse_args()
    if a.mode == "inputs":
        inputs()
    elif a.mode == "drive":
        drive()
    else:
        import_tpu(a.step)
