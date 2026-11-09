"""Install exact user-mountable research volumes or reattach them after LUKS unlock."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import uuid


BEGIN = "# BEGIN SLWM research storage"
END = "# END SLWM research storage"


def path(value):
    p = Path(value)
    if not p.is_absolute() or ".." in p.parts or not re.fullmatch(r"[/A-Za-z0-9_.-]+", str(p)):
        raise ValueError(f"Unsupported mount path: {p}")
    return p


def load(config):
    data = json.loads(config.read_text())
    volumes = {path(p): str(uuid.UUID(u)) for p,u in data["volumes"].items()}
    binds = {path(p): path(d) for p,d in data["binds"].items()}
    if not all(p.parent == Path("/mnt") for p in volumes):
        raise ValueError("Use distinct dedicated directories under /mnt")
    for target, source in binds.items():
        if not target.is_relative_to("/home") or not any(source.is_relative_to(v / "SLWM") for v in volumes):
            raise ValueError("Expected a checkout and dedicated SLWM storage as bind endpoints")
    return volumes, binds


def install(config):
    volumes, binds = load(config)
    fstab = Path("/etc/fstab")
    original = fstab.read_text()
    if original.count(BEGIN) != original.count(END) or original.count(BEGIN) > 1:
        raise ValueError("Unexpected existing managed block")
    base = re.sub(re.escape(BEGIN)+r".*?"+re.escape(END)+r"\n?", "", original, flags=re.S)
    existing = {line.split()[1] for line in base.splitlines() if line.strip() and not line.lstrip().startswith("#") and len(line.split()) >= 2}
    if existing & {str(p) for p in (*volumes, *binds)}:
        raise ValueError("An existing fstab entry already manages one of these paths")
    rows = [BEGIN]
    for target, device in volumes.items():
        target.mkdir(mode=0o755, exist_ok=True)
        rows.append(f"UUID={device} {target} btrfs noauto,nofail,user,noatime,nodev,nosuid 0 0")
    for target, source in binds.items():
        rows.append(f"{source} {target} none noauto,nofail,user,bind,nodev,nosuid 0 0")
    rows.append(END)
    new = base.rstrip()+"\n\n"+"\n".join(rows)+"\n"
    if new != original:
        with tempfile.NamedTemporaryFile(dir=fstab.parent,delete=False) as f:
            pending = Path(f.name)
            os.fchmod(f.fileno(), fstab.stat().st_mode & 0o777)
            f.write(new.encode())
            f.flush()
            os.fsync(f.fileno())
        try:
            subprocess.run(["findmnt","--verify","--tab-file",pending],check=True)
            shutil.copy2(fstab, fstab.with_name(f"fstab.slwm-backup-{int(time.time())}"))
            pending.replace(fstab)
        finally:
            pending.unlink(missing_ok=True)
        subprocess.run(["systemctl","daemon-reload"],check=True)
    print("Research mounts registered; existing fstab entries preserved.")


def attach(config):
    volumes, binds = load(config)
    for target, device in volumes.items():
        if not os.path.ismount(target):
            subprocess.run(["mount",str(target)],check=True)
        actual = subprocess.check_output(["findmnt","-nro","UUID","-T",target],text=True).strip()
        if actual != device:
            raise OSError(f"Wrong volume at {target}")
    for target, source in binds.items():
        if target.exists() and target.samefile(source):
            continue
        receipt = source.parent / ("." + source.name + ".slwm-receipt.json")
        if not receipt.is_file():
            continue
        if not target.is_dir() or any(target.iterdir()) or target.stat().st_mode & 0o222:
            raise OSError(f"Refusing to cover an unprepared directory: {target}")
        subprocess.run(["mount",str(target)],check=True)
        if not target.samefile(source):
            raise OSError(f"Bind verification failed: {target}")
    print("Research volumes attached.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["install","attach"])
    p.add_argument("config",type=Path)
    args = p.parse_args()
    (install if args.action=="install" else attach)(args.config)
