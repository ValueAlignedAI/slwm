"""Restore exact G10 post-training parents and fully prepared public data on a replacement assignment."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from g7_inputs import FED, shell, stream
from g9_storage import write
from g10_dispatch import FILES


def main(ctl, name, seed, revision):
    wrapper = ["bash", "scripts/train/colab_session.sh"]
    subprocess.run(
        [
            *wrapper,
            "exec",
            name,
            "-f",
            "scripts/train/g4_colab_topology.py",
            "--timeout",
            "90",
        ],
        check=True,
        timeout=140,
    )
    source = sorted(
        set(
            FILES
            + [
                "models/g4_shared.py",
                "models/g5_shared.py",
                "models/g9_tabletop.py",
                "scripts/train/g9_fit.py",
            ]
        )
    )
    stream(
        name,
        ["git", "archive", revision, *source],
        "set -eu; mkdir -p /content/g10-code /content/g10-data /content/g10-out; cd /content/g10-code; tar -xf -",
    )
    for bank, origin, expected in (
        (
            "starss-train.npz",
            "/mnt/slwm-data/SLWM/g9/data/starss-train.npz",
            "5aa519fb7782bb6d6e3451d6c144619446c6541418d4e7a9bc39c1faebca3089",
        ),
        ("holo-train.npz", "/mnt/slwm-data/SLWM/g10/data/holo/holo-train.npz", None),
    ):
        if expected is None:
            expected = json.loads(
                subprocess.check_output(
                    [*FED, "cat /mnt/slwm-data/SLWM/g10/data/holo/holo-train.json"],
                    text=True,
                )
            )["sha256"]
        status = shell(
            name,
            "cd /content/g10-data; if test -f "
            + bank
            + "; then sha256sum "
            + bank
            + "; fi",
            capture_output=True,
            text=True,
            timeout=70,
        )
        if expected not in status.stdout:
            write(
                ctl / (name + "-restore.json"),
                dict(phase="sending", bank=bank, utc=time.time()),
            )
            stream(
                name,
                [*FED, "cat " + origin],
                "set -eu; cd /content/g10-data; cat > ."
                + bank
                + ".tmp; echo '"
                + expected
                + "  ."
                + bank
                + ".tmp' | sha256sum -c -; mv ."
                + bank
                + ".tmp "
                + bank,
            )
        if bank == "holo-train.npz":
            write(ctl / (name + "-bank.json"), dict(sha256=expected))
    files = []
    checks = []
    for core in ("hybrid", "gpt2"):
        for tag, step in (
            (f"context-{core}-variable-s{seed}", 5120),
            (f"align-{core}-rooms-s{seed}", 1536),
        ):
            stage = f"{tag}/step-{step:05d}/"
            remote = "/mnt/slwm-checkpoints-a/SLWM/g10/runs/"
            meta = json.loads(
                subprocess.check_output(
                    [*FED, "cat " + remote + stage + "verified.json"], text=True
                )
            )
            files += [stage + "weights.npz", stage + "receipt.json"]
            checks.append(
                "echo '"
                + meta["inference_sha256"]
                + "  "
                + stage
                + "weights.npz' | sha256sum -c -"
            )
    write(
        ctl / (name + "-restore.json"), dict(phase="restoring_parents", utc=time.time())
    )
    stream(
        name,
        [
            *FED,
            "tar -cf - -C /mnt/slwm-checkpoints-a/SLWM/g10/runs " + shlex.join(files),
        ],
        "set -eu; cd /content/g10-out; tar -xf -; " + "; ".join(checks),
    )
    command = "set -e; cd /content/g10-code; source /content/g4-topology.env; export PYTHONPATH=. JAX_PLATFORMS=tpu OMP_NUM_THREADS=2; "
    for op in ("mixed", "align-eval"):
        command += (
            "flock /content/g10-train.lock python -u scripts/train/g10_lane.py "
            + op
            + " /content/g10-out /content/g10-data /content/g10-eval "
            + str(seed)
            + " --cache /content/g10-cache; "
        )
    shell(
        name,
        "nohup bash -c "
        + shlex.quote(command)
        + " </dev/null >>/content/g10-out/recovery.log 2>&1 & echo $! >/content/g10-out/recovery.pid; echo G10_RESTORED",
        timeout=100,
    )
    write(
        ctl / (name + "-restore.json"),
        dict(phase="training", seed=seed, parents=4, utc=time.time()),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ctl", type=Path)
    p.add_argument("name", choices=("slwm-g10-ground-a", "slwm-g10-ground-b"))
    p.add_argument("seed", type=int)
    p.add_argument("revision")
    a = p.parse_args()
    main(a.ctl, a.name, a.seed, a.revision)
