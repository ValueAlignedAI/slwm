"""Move a quiescent research directory to a verified Linux volume, preserving its original path."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def filesystem(path):
    return subprocess.check_output(["findmnt", "-nro", "UUID", "-T", path], text=True).strip()


def inventory(root):
    tree = {}
    for folder, dirs, files in os.walk(root):
        for name in sorted(dirs + files):
            path = Path(folder) / name
            key = str(path.relative_to(root))
            if path.is_symlink():
                target = os.readlink(path)
                if not os.path.isabs(target) and not path.resolve().is_relative_to(root.resolve()):
                    raise ValueError(f"External relative symlink requires review: {path}")
                tree[key] = ["link", target]
            elif path.is_dir():
                tree[key] = ["directory"]
            elif path.is_file():
                before = path.stat()
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                after = path.stat()
                if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                    raise OSError(f"File changed while verifying: {path}")
                tree[key] = ["file", after.st_size, digest]
            else:
                raise ValueError(f"Unexpected file type: {path}")
    return tree


def exchange(a, b):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.renameat2(-100, os.fsencode(a), -100, os.fsencode(b), 2):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def migrate(source, dest, expected_uuid, bind=False):
    source, dest = source.absolute(), dest.absolute()
    mount = dest.parent
    while not mount.exists():
        mount = mount.parent
    if not expected_uuid or filesystem(mount) != expected_uuid:
        raise OSError("Destination filesystem is missing or has the wrong UUID")
    old = source.with_name("." + source.name + ".slwm-verified-origin")
    done = source.exists() and dest.is_dir() and source.samefile(dest)
    if bind and not done:
        entry = subprocess.run(["findmnt","--fstab","--mountpoint",str(source),"-nro","SOURCE,OPTIONS"],
                               capture_output=True,text=True)
        fields = entry.stdout.strip().split()
        if entry.returncode or len(fields)!=2 or fields[0]!=str(dest) or not {"bind","user"} <= set(fields[1].split(",")):
            raise OSError("Configure the exact noauto,user,bind fstab entry before cutover")
        if old.is_dir() and not old.is_symlink():
            if source.is_dir() and not any(source.iterdir()) and not os.path.ismount(source):
                source.rmdir()
            if not source.exists():
                old.rename(source)
    if done and not old.exists():
        if not bind and source.is_symlink() and source.readlink() != dest:
            link = source.with_name("." + source.name + ".slwm-new-link")
            link.symlink_to(dest, target_is_directory=True)
            link.replace(source)
        print(f"already relocated {source}", flush=True)
        return
    if not done:
        if source.is_symlink() or not source.is_dir() or os.path.lexists(old):
            raise ValueError(f"Unexpected migration state for {source}")
        if source.stat().st_dev == mount.stat().st_dev or dest.is_relative_to(source):
            raise ValueError("Choose a different mounted filesystem")
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-aHS", "--partial", "--", str(source)+"/", str(dest)+"/"], check=True)
    original = old if done else source
    print(f"verifying {original} against {dest}", flush=True)
    first = inventory(original)
    if first != inventory(dest):
        raise ValueError("Source and destination inventory or SHA-256 differ")
    subprocess.run(["sync", "-f", dest], check=True)
    receipt = {"source": str(source), "destination": str(dest), "filesystem_uuid": expected_uuid,
               "files": sum(x[0]=="file" for x in first.values()), "bytes": sum(x[1] for x in first.values() if x[0]=="file"),
               "inventory_sha256": hashlib.sha256(json.dumps(first,sort_keys=True).encode()).hexdigest(),
               "verified_unix": time.time()}
    receipt_file = dest.parent / ("." + dest.name + ".slwm-receipt.json")
    receipt_file.write_text(json.dumps(receipt, indent=2)+"\n")
    if not done and bind:
        source.rename(old)
        try:
            source.mkdir(mode=0o555)
            subprocess.run(["mount", str(source)], check=True)
        except BaseException:
            if source.is_dir() and not os.path.ismount(source):
                source.rmdir()
            if not source.exists():
                old.rename(source)
            raise
    elif not done:
        old.symlink_to(dest, target_is_directory=True)
        try:
            exchange(old, source)
        except BaseException:
            old.unlink()
            raise
    if not old.is_dir() or old.is_symlink() or not source.samefile(dest) or filesystem(source.resolve()) != expected_uuid:
        raise OSError("Path cutover failed; original retained")
    shutil.rmtree(old)
    subprocess.run(["sync", "-f", source.parent], check=True)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--uuid", required=True)
    p.add_argument("--bind", action="store_true", help="Keep the checkout path a directory using a user-mountable fstab entry")
    args = p.parse_args()
    migrate(args.source, args.destination, args.uuid, args.bind)
