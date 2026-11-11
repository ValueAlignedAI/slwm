"""Stream the exact G14 release revision to the training SSD, then certify both HDD archives."""

import argparse
from pathlib import Path
import re
import subprocess
from g7_integrate_release import git

HOST = "db@192.168.1.200"
ROOT = "/mnt/slwm-data/SLWM/g14"


def main(revision):
    if (
        not re.fullmatch(r"[0-9a-f]{40}", revision)
        or git("rev-parse", revision) != revision
    ):
        raise ValueError("Unverified Git object")
    names = git("ls-tree", "-r", "--name-only", revision).splitlines()
    selected = [
        name
        for name in names
        if name == "README.md"
        or "experiments/multimodal/t14/spoken-closed-loop-20261111/" in name
        or any(
            name.startswith(path)
            for path in (
                "data/g14_",
                "models/g14_",
                "evals/g14_",
                "scripts/train/g14_",
                "scripts/infer/g14_",
                "scripts/data/g14_",
                "tests/test_g14_",
            )
        )
    ]
    if not any(n.endswith("/metrics/release.json") for n in selected):
        raise ValueError("The study receipt is missing")
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", HOST]
    archive = ROOT + "/staging/g14-" + revision + ".tar.gz"
    action = (
        "set -eu; mkdir -p "
        + ROOT
        + "/staging; cat > "
        + archive
        + ".part; mv "
        + archive
        + ".part "
        + archive
    )
    with subprocess.Popen(
        ["git", "archive", "--format=tar.gz", revision, "--", *selected],
        stdout=subprocess.PIPE,
    ) as send:
        cp = subprocess.run(ssh + [action], stdin=send.stdout)
        send.stdout.close()
        if cp.returncode or send.wait():
            raise RuntimeError("Failed to transfer the exact release")
    subprocess.run(
        ssh
        + [
            "cd "
            + ROOT
            + "/code && tmux new-window -d -t slwm-g14 -n published-archive -c "
            + ROOT
            + "/code 'bash -lc \"set -eo pipefail; PYTHONPATH=. python3 -u scripts/train/g14_custody.py publish --revision "
            + revision
            + " 2>&1 | tee "
            + ROOT
            + "/logs/published-archive.log\"'"
        ],
        check=True,
    )
    import time

    for i in range(120):
        for mount in ("a", "b"):
            path = (
                "/mnt/slwm-checkpoints-" + mount + "/SLWM/g14/published/verified.json"
            )
            result = subprocess.run(
                ssh + ["cat " + path], capture_output=True, text=True
            )
            if result.returncode or "g14-" + revision + ".tar.gz" not in result.stdout:
                break
        else:
            print("Verified two archives of", revision, flush=True)
            return
        time.sleep(3)
    raise RuntimeError("Published archive receipts did not arrive")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("revision")
    main(p.parse_args().revision)
