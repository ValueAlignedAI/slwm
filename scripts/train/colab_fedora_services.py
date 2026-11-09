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
    units = []
    for name, command in commands.items():
        unit = name + ".service"
        body = ["[Unit]", f"Description=SLWM {name}", "", "[Service]", "Type=simple", "UMask=0077",
                "Nice=10", "Restart=on-failure", "RestartSec=30", "NoNewPrivileges=yes",
                *["Environment=" + quote(v) for v in environment],
                "ExecStart=" + " ".join(map(quote, command)), "", "[Install]", "WantedBy=default.target", ""]
        temp = folder / (unit + ".new")
        temp.write_text("\n".join(body))
        temp.replace(folder / unit)
        units.append(unit)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", *units], check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("repo", "spool", "assets", "torch-python", "config"):
        p.add_argument("--" + option, type=lambda s: Path(os.path.abspath(s)), required=True)
    install(p.parse_args())
