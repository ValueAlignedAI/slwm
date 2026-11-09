"""Stage immutable training artifacts locally and publish verified generations to Drive.

The producer atomically replaces its files before calling capture. The spool and
sources share a filesystem. Only the separate sync process reads the large files.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}")
SHA = re.compile(r"[0-9a-f]{64}")
GENERATION = re.compile(r"([0-9]{12})-([0-9a-f]{64})\.json")
LIMIT = 10**12


class IntegrityError(ValueError):
    pass


def safe_name(value):
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise ValueError(f"Invalid artifact name: {value!r}")
    return value


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextlib.contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def capture(spool, run, step, files):
    """Pin an already-published generation without reading or duplicating its file data."""
    safe_name(run)
    if not 0 <= step < LIMIT or not files:
        raise ValueError("Expected a nonempty generation and a valid nonnegative step")
    root = spool / "pending" / run
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{step:012d}"
    with lock(spool / "locks" / f"{run}.lock"):
        if target.exists():
            entries = {path.name for path in target.iterdir()}
            if entries == set(files) and all((target / k).samefile(p) for k, p in files.items()):
                return
            raise IntegrityError(f"Different files already staged for {run}/{step}")
        with tempfile.TemporaryDirectory(dir=root, prefix=".capture-") as directory:
            generation = Path(directory) / "generation"
            generation.mkdir()
            for name, source in files.items():
                safe_name(name)
                if not source.is_file() or source.is_symlink():
                    raise ValueError(f"Expected a regular published file: {source}")
                os.link(source, generation / name)
            generation.rename(target)


def rclone(*args):
    return subprocess.run(["rclone", *map(str, args)], check=True, stdout=subprocess.PIPE).stdout


def remote_sha(path):
    # Drive supplies a server-side checksum. Fall back to a streamed read-back
    # if a backend has no SHA-256 field for an older binary.
    output = rclone("hashsum", "SHA-256", path).decode().split()
    if output and SHA.fullmatch(output[0]):
        return output[0]
    with subprocess.Popen(["rclone", "cat", path], stdout=subprocess.PIPE) as reader:
        digest = hashlib.file_digest(reader.stdout, "sha256").hexdigest()
        if reader.wait():
            raise OSError(f"Could not verify {path}")
        return digest


def put_verified(source, destination, digest):
    rclone("copyto", "--checksum", "--immutable", source, destination)
    if remote_sha(destination) != digest:
        raise IntegrityError(f"Remote checksum mismatch: {destination}")


def manifest_for(directory, run, step):
    return {"schema": 1, "run": run, "step": step, "files": {
        p.name: {"size": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(directory.iterdir())}}


def publish(spool, remote, directory):
    run, step = safe_name(directory.parent.name), int(directory.name)
    manifest = manifest_for(directory, run, step)
    for name, info in manifest["files"].items():
        digest = info["sha256"]
        put_verified(directory / name, f"{remote}/objects/{digest}", digest)
    body = encode(manifest)
    digest = hashlib.sha256(body).hexdigest()
    marker = f"{step:012d}-{digest}.json"
    receipts = spool / "receipts" / run
    receipts.mkdir(parents=True, exist_ok=True)
    previous = list(receipts.glob(f"{step:012d}-*.json"))
    if any(p.name != marker for p in previous):
        raise IntegrityError(f"Conflicting committed step: {run}/{step}")
    with tempfile.NamedTemporaryFile(dir=receipts, delete=False) as out:
        pending = Path(out.name)
        out.write(body)
    try:
        put_verified(pending, f"{remote}/runs/{run}/{marker}", digest)
        pending.replace(receipts / marker)
    finally:
        pending.unlink(missing_ok=True)
    with lock(spool / "locks" / f"{run}.lock"):
        shutil.rmtree(directory)
    print(f"verified {run} step={step} files={len(manifest['files'])}", flush=True)


def sync(spool, remote):
    with lock(spool / "locks" / "uploader"):
        for directory in sorted((spool / "pending").glob("*/*")):
            if re.fullmatch("[0-9]{12}", directory.name):
                publish(spool, remote, directory)


def unpack(remote, marker, run, staging):
    match = GENERATION.fullmatch(marker)
    step, expected = int(match[1]), match[2]
    body = rclone("cat", f"{remote}/runs/{run}/{marker}")
    if hashlib.sha256(body).hexdigest() != expected:
        raise IntegrityError("Manifest checksum mismatch")
    manifest = json.loads(body)
    if not isinstance(manifest, dict) or (manifest.get("schema"), manifest.get("run"), manifest.get("step")) != (1, run, step):
        raise IntegrityError("Manifest identity mismatch")
    if not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise IntegrityError("Empty or invalid file inventory")
    for name, info in manifest["files"].items():
        path = staging / safe_name(name)
        if not isinstance(info, dict) or not isinstance(info.get("sha256"), str):
            raise IntegrityError("Invalid object inventory")
        digest = info["sha256"]
        if not SHA.fullmatch(digest) or type(info.get("size")) is not int or info["size"] < 0:
            raise IntegrityError("Invalid object identity")
        rclone("copyto", f"{remote}/objects/{digest}", path)
        if path.stat().st_size != info["size"] or sha256(path) != digest:
            raise IntegrityError(f"Restored checksum mismatch: {name}")
    return manifest


def restore(remote, run, destination):
    safe_name(run)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Choose a new restore directory: {destination}")
    names = set(rclone("lsf", "--files-only", f"{remote}/runs/{run}").decode().splitlines())
    markers = sorted((m for m in names if GENERATION.fullmatch(m)), reverse=True)
    steps = [m.split("-", 1)[0] for m in markers]
    if len(steps) != len(set(steps)):
        raise IntegrityError(f"Conflicting manifests for the same step of {run}")
    for marker in markers:
        with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".restore-") as tmp:
            staging = Path(tmp) / "verified"
            staging.mkdir()
            try:
                manifest = unpack(remote, marker, run, staging)
            except subprocess.CalledProcessError as error:
                if error.returncode not in (3, 4):
                    raise
                print(f"Skipping {marker}: missing remote file", file=sys.stderr)
                continue
            except (ValueError, KeyError, TypeError) as error:
                print(f"Skipping {marker}: {error}", file=sys.stderr)
                continue
            staging.rename(destination)
            print(f"restored {run} step={manifest['step']} from {marker}", flush=True)
            return manifest
    raise IntegrityError(f"No complete verified generation for {run}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("capture")
    add.add_argument("--spool", type=Path, required=True)
    add.add_argument("--run", required=True)
    add.add_argument("--step", type=int, required=True)
    add.add_argument("--file", action="append", required=True, metavar="NAME=PATH")
    push = commands.add_parser("sync")
    push.add_argument("--spool", type=Path, required=True)
    push.add_argument("--watch", action="store_true")
    pull = commands.add_parser("restore")
    pull.add_argument("--run", required=True)
    pull.add_argument("--to", type=Path, required=True)
    for cmd in (push, pull):
        cmd.add_argument("--remote", default="slwm-drive:SLWM/v1")
    args = parser.parse_args()
    if args.command == "capture":
        files = {}
        for item in args.file:
            name, sep, path = item.partition("=")
            if not sep or name in files:
                parser.error("--file requires a distinct NAME=PATH")
            files[name] = Path(path)
        capture(args.spool, args.run, args.step, files)
    elif args.command == "restore":
        restore(args.remote.rstrip("/"), args.run, args.to)
    else:
        delay = 15
        while True:
            try:
                sync(args.spool, args.remote.rstrip("/"))
                delay = 15
            except (OSError, subprocess.CalledProcessError, IntegrityError) as error:
                if not args.watch:
                    raise
                print(f"Upload pending: {error}. Retrying in {delay}s", file=sys.stderr, flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 300)
                continue
            if not args.watch:
                return
            time.sleep(delay)


if __name__ == "__main__":
    main()
