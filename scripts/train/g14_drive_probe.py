"""Attempt one bounded, checksum-verified transfer of the released speech weights using the approved Drive remote."""

from pathlib import Path
import os
import subprocess
import time
from g9_storage import sha, write

ROOT = Path("/mnt/slwm-data/SLWM/g14")


def main():
    src = ROOT / "release/inference/speech.npz"
    digest = sha(src)
    remote = "slwm-drive:SLWM/g14/objects/" + digest
    env = dict(
        os.environ, RCLONE_CONFIG=str(Path.home() / ".config/slwm-colab/rclone.conf")
    )
    bin = str(Path.home() / ".local/bin/rclone")
    args = [
        bin,
        "--retries",
        "1",
        "--low-level-retries",
        "1",
        "--contimeout",
        "10s",
        "--timeout",
        "25s",
    ]
    phase = "verified"
    reason = None
    try:
        cp = subprocess.run(
            args + ["copyto", "--immutable", "--checksum", str(src), remote],
            env=env,
            text=True,
            capture_output=True,
            timeout=55,
        )
        if cp.returncode:
            raise RuntimeError(cp.stderr)
        result = subprocess.run(
            args + ["hashsum", "SHA-256", remote],
            env=env,
            text=True,
            capture_output=True,
            timeout=40,
        )
        if result.returncode:
            raise RuntimeError(result.stderr)
        if not result.stdout.startswith(digest):
            raise ValueError("Remote checksum differs")
    except (RuntimeError, subprocess.TimeoutExpired, OSError) as ex:
        phase = "pending"
        s = str(ex).lower()
        reason = (
            "Google Drive API project quota"
            if "quota" in s or "ratelimit" in s or "rate limit" in s
            else type(ex).__name__
        )
    write(
        ROOT / "work/drive-probe.json",
        dict(
            artifact=src.name,
            sha256=digest,
            status=phase,
            reason=reason,
            checked_unix=time.time(),
        ),
    )
    print("Current scoped Drive transfer", phase, reason, flush=True)


if __name__ == "__main__":
    main()
