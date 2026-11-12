"""Dual-HDD SHA-256 verified checkpoint custody on Fedora for Study X1B.

Verifies physical disk UUIDs via findmnt:
    Primary   : /mnt/slwm-checkpoints-a (UUID fbbb53d9-dd42-4f19-95e6-dadab02a201e)
    Secondary : /mnt/slwm-checkpoints-b (UUID c961a363-8d77-4fba-a48d-622b00a6d8c3)
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

DISK_PAIR = (
    ("/mnt/slwm-checkpoints-a", "fbbb53d9-dd42-4f19-95e6-dadab02a201e"),
    ("/mnt/slwm-checkpoints-b", "c961a363-8d77-4fba-a48d-622b00a6d8c3"),
)
TAG_RE = re.compile(r"x1b-[a-z0-9_-]{3,80}")
ALLOWED_FILES = {
    "portable_int8.npz",
    "weights_bf16.npz",
    "eval_metrics.json",
    "config.json",
    "train.jsonl",
    "receipt.json",
}
MAX_BYTES = 16 * (1 << 30)  # Up to 16 GiB per 1B-scale checkpoint component


def sha256_path(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def verified_roots() -> tuple[Path, Path]:
    roots = []
    for mount, expected_uuid in DISK_PAIR:
        actual = subprocess.check_output(["findmnt", "-nro", "UUID", "-M", mount], text=True).strip()
        if actual != expected_uuid:
            raise RuntimeError(f"Mount {mount} UUID mismatch: expected {expected_uuid}, got {actual}")
        roots.append(Path(mount) / "SLWM/x1b/runs")
    return roots[0], roots[1]


def receive_stream(tag: str, filename: str, expected_sha: str, expected_bytes: int) -> None:
    if not TAG_RE.fullmatch(tag):
        raise ValueError(f"Invalid X1B run tag: {tag!r}")
    if filename not in ALLOWED_FILES:
        raise ValueError(f"Disallowed artifact filename: {filename!r}")
    if not (0 < expected_bytes <= MAX_BYTES):
        raise ValueError(f"Invalid size {expected_bytes}")

    root_a, _ = verified_roots()
    run_a = root_a / tag
    run_a.mkdir(parents=True, exist_ok=True)
    dest = run_a / filename
    if dest.is_file() and dest.stat().st_size == expected_bytes and sha256_path(dest) == expected_sha:
        print("ALREADY_PRESENT", flush=True)
        return

    partial_p = run_a / f".{filename}.partial"
    with (run_a / f".{filename}.lock").open("a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with partial_p.open("wb") as out:
            shutil.copyfileobj(sys.stdin.buffer, out, 4 << 20)
            out.flush()
            os.fsync(out.fileno())
        actual_size = partial_p.stat().st_size
        if actual_size != expected_bytes:
            raise RuntimeError(f"Incomplete stream for {tag}/{filename}: {actual_size} != {expected_bytes}")
        actual_sha = sha256_path(partial_p)
        if actual_sha != expected_sha:
            partial_p.replace(run_a / f".{filename}.corrupt")
            raise RuntimeError(f"SHA256 mismatch for {tag}/{filename}: {actual_sha} != {expected_sha}")
        partial_p.replace(dest)
    print("RECEIVED_OK", actual_sha, flush=True)


def mirror_and_verify(tag: str) -> None:
    if not TAG_RE.fullmatch(tag):
        raise ValueError(f"Invalid X1B run tag: {tag!r}")
    root_a, root_b = verified_roots()
    run_a = root_a / tag
    run_b = root_b / tag
    run_b.mkdir(parents=True, exist_ok=True)

    receipt = json.loads((run_a / "receipt.json").read_text())
    if receipt["tag"] != tag:
        raise ValueError("Receipt tag mismatch")

    must_check = [("portable_int8.npz", receipt["portable_int8_sha256"])]
    if receipt.get("weights_bf16_sha256"):
        must_check.append(("weights_bf16.npz", receipt["weights_bf16_sha256"]))

    aux_files = ["receipt.json", "eval_metrics.json", "config.json", "train.jsonl"]
    digests: dict[str, str] = {}

    with (run_a / ".mirror.lock").open("a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for fname, exp_sha in must_check:
            src = run_a / fname
            if sha256_path(src) != exp_sha:
                raise RuntimeError(f"Primary disk SHA mismatch on {fname}")
            dst = run_b / fname
            if not dst.is_file() or sha256_path(dst) != exp_sha:
                tmp = run_b / f".{fname}.partial"
                shutil.copyfile(src, tmp)
                if sha256_path(tmp) != exp_sha:
                    raise RuntimeError(f"Secondary disk SHA mismatch on {fname}")
                tmp.replace(dst)
            digests[fname] = exp_sha

        for fname in aux_files:
            src = run_a / fname
            if src.is_file():
                d_sha = sha256_path(src)
                dst = run_b / fname
                tmp = run_b / f".{fname}.partial"
                shutil.copyfile(src, tmp)
                if sha256_path(tmp) != d_sha:
                    raise RuntimeError(f"Secondary aux SHA mismatch on {fname}")
                tmp.replace(dst)
                digests[fname] = d_sha

        ver = {
            "tag": tag,
            "verified_copies": 2,
            "disk_uuids": [u for _, u in DISK_PAIR],
            "sha256": digests,
        }
        atomic_json(run_a / "verified.json", ver)
        atomic_json(run_b / "verified.json", ver)
    print("MIRROR_VERIFIED_2_COPIES", json.dumps(ver), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("op", choices=["receive", "mirror"])
    ap.add_argument("tag")
    ap.add_argument("rest", nargs="*")
    args = ap.parse_args()
    if args.op == "receive":
        receive_stream(args.tag, args.rest[0], args.rest[1], int(args.rest[2]))
    elif args.op == "mirror":
        mirror_and_verify(args.tag)


if __name__ == "__main__":
    main()
