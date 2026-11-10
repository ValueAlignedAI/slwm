"""Install restartable per-user Fedora services for Drive transfers and G3 capture."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def quote(value):
    value = str(value)
    if "\n" in value or "\r" in value:
        raise ValueError("Unexpected newline in unit setting")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$") + '"'


def install(args):
    here = Path(__file__).resolve().parent
    folder = Path.home() / ".config/systemd/user"
    folder.mkdir(parents=True, exist_ok=True)
    rclone = shutil.which("rclone")
    if not rclone or not args.config.is_file():
        raise ValueError("Install rclone and configure the dedicated Drive remote first")
    environment = [f"PATH={Path(rclone).parent}:/usr/local/bin:/usr/bin:/bin",
                   f"RCLONE_CONFIG={args.config.resolve()}", "RCLONE_DRIVE_CHUNK_SIZE=64M",
                   "RCLONE_LOG_LEVEL=ERROR", "PYTHONUNBUFFERED=1"]
    commands = {
        "slwm-drive-capture": [args.torch_python, here / "colab_archive_joint.py", "--repo", args.repo,
                               "--spool", args.spool, "--watch"],
        "slwm-drive-checkpoints": [sys.executable, here / "colab_artifacts.py", "sync", "--spool", args.spool, "--watch"],
        "slwm-drive-inputs": [sys.executable, here / "colab_artifacts.py", "sync", "--spool", args.assets, "--watch"],
    }
    if args.replica:
        commands["slwm-drive-checkpoints"].extend(["--replica", args.replica, "--replica-uuid", args.replica_uuid])
    if args.checkpoint_root:
        commands["slwm-drive-capture"].extend(["--checkpoint-root", args.checkpoint_root])
    if args.protect:
        runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
        guard = [sys.executable, "-u", here/"slwm_storage_guard.py", "watch", "--state", runtime/"slwm-storage-guard.json"]
        for repo in [args.repo, *args.other_repo]:
            guard.extend(["--repo", repo])
        for protected in args.protect:
            guard.extend(["--path", protected])
        for location, device in args.required_volume:
            guard.extend(["--volume", location, device])
        commands["slwm-storage-guard"] = guard
    units = []
    changed = []
    for name, command in commands.items():
        unit = name + ".service"
        body = ["[Unit]", f"Description=SLWM {name}", "", "[Service]", "Type=simple", "UMask=0077",
                "Nice=10", "Restart=on-failure", "RestartSec=30", "NoNewPrivileges=yes",
                *["Environment=" + quote(v) for v in environment],
                "ExecStart=" + " ".join(map(quote, command)), "", "[Install]", "WantedBy=default.target", ""]
        target = folder / unit
        content = "\n".join(body)
        if not target.exists() or target.read_text() != content:
            temp = folder / (unit + ".new")
            temp.write_text(content)
            temp.replace(target)
            changed.append(unit)
        units.append(unit)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", *units], check=True)
    if changed:
        subprocess.run(["systemctl", "--user", "restart", *changed], check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("repo", "spool", "assets", "torch-python", "config"):
        p.add_argument("--" + option, type=lambda s: Path(os.path.abspath(s)), required=True)
    p.add_argument("--replica", type=Path)
    p.add_argument("--checkpoint-root", type=Path)
    p.add_argument("--replica-uuid")
    p.add_argument("--protect", type=Path, action="append", default=[])
    p.add_argument("--other-repo", type=Path, action="append", default=[])
    p.add_argument("--required-volume", nargs=2, action="append", default=[])
    args = p.parse_args()
    if bool(args.replica) != bool(args.replica_uuid):
        p.error("Supply both --replica and --replica-uuid")
    install(args)
