"""Offer a real speech-codec replication to a metered native v6e, with an independent local lane."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from g7_inputs import FED, shell, stream
from g9_storage import write

NAME = "slwm-g14-speech-v6"


def main(ctl, cli, rev):
    os.environ.update(
        COLAB_CLI=str(cli),
        CLOUDSDK_CONFIG=str(Path.home() / ".config/slwm-colab/gcloud"),
        COLAB_CLI_REQUEST_TIMEOUT="50",
    )
    cmd = [str(cli), "--auth", "adc"]

    def assigned():
        out = subprocess.check_output(cmd + ["sessions"], text=True, timeout=85)
        return any(line.startswith("[" + NAME + "] ") for line in out.splitlines())

    attempts = []
    done = False
    try:
        for i in range(2):
            start = time.time()
            message = ""
            if not assigned():
                try:
                    r = subprocess.run(
                        cmd + ["new", "-s", NAME, "--tpu", "v6e1"],
                        capture_output=True,
                        text=True,
                        timeout=225,
                    )
                    message = (r.stdout + " " + r.stderr).lower()
                except subprocess.TimeoutExpired:
                    message = "timeout"
            active = assigned()
            reason = (
                "assigned"
                if active
                else (
                    "service unavailable"
                    if "service unavailable" in message
                    else "timeout" if "time" in message else "unassigned"
                )
            )
            attempts.append(dict(start=start, end=time.time(), result=reason))
            write(ctl / "attempts.json", attempts)
            if active:
                break
            if i == 0:
                time.sleep(100)
        if not active:
            return
        subprocess.run(
            [
                "bash",
                "scripts/train/colab_session.sh",
                "exec",
                NAME,
                "-f",
                "scripts/train/g4_colab_topology.py",
                "--timeout",
                "80",
            ],
            check=True,
            timeout=130,
        )
        stream(
            NAME,
            [
                "git",
                "archive",
                rev,
                "data/g14_grid.py",
                "models/g14_cue.py",
                "scripts/train/g14_cue_fit.py",
                "scripts/train/g13_optim.py",
                "scripts/train/g9_storage.py",
            ],
            "set -eu; mkdir -p /content/g14; cd /content/g14; tar -xf -",
        )
        stream(
            NAME,
            [
                *FED,
                "tar -C /mnt/slwm-data/SLWM/g14/data/utterances -cf - train.npz dev.npz",
            ],
            "set -eu; mkdir -p /content/g14/data/utterances; tar -xf - -C /content/g14/data/utterances",
        )
        shell(
            NAME,
            "set -eu; cd /content/g14; source /content/g4-topology.env; export PYTHONPATH=. JAX_PLATFORMS=tpu; python -u scripts/train/g14_cue_fit.py train --native --seed 1403",
            timeout=2400,
        )
        data = json.loads(
            shell(
                NAME,
                "cat /content/g14/work/cue/choices-1403.json",
                capture_output=True,
                text=True,
                timeout=100,
            ).stdout
        )
        if set(data) != {"unaligned-1403", "aligned-1403", "audiovisual-1403"} or not all(
            "v6" in v["accelerator"].lower() for v in data.values()
        ):
            raise ValueError("TPU replication incomplete")
        names = [
            f"{t}/step-{step:05d}/{file}"
            for t in data
            for step in (128, 640, 1536)
            for file in ("config.json", "verified.json", "weights.npz", "optimizer.npz")
        ]
        with subprocess.Popen(
            [
                "bash",
                "scripts/train/colab_session.sh",
                "ssh",
                NAME,
                "bash",
                "-c",
                "cd /content/g14/checkpoints && tar -cf - " + " ".join(names),
            ],
            stdout=subprocess.PIPE,
        ) as sender:
            try:
                subprocess.run(
                    [
                        *FED,
                        "set -eu; mountpoint -q /mnt/slwm-data; mkdir -p /mnt/slwm-data/SLWM/g14/native; tar -xf - -C /mnt/slwm-data/SLWM/g14/native",
                    ],
                    stdin=sender.stdout,
                    check=True,
                    timeout=900,
                )
            finally:
                sender.stdout.close()
            if sender.wait(timeout=50):
                raise ValueError("Native checkpoint transfer incomplete")
        write(ctl / "native-results.json", data)
        done = True
    finally:
        if assigned():
            subprocess.run(cmd + ["stop", "-s", NAME], check=True, timeout=130)
        if assigned():
            raise ValueError("Owned native assignment still active")
        write(
            ctl / "native.json",
            dict(verified_replication=done, released=True, attempts=attempts),
        )
        (ctl / "requests-done").touch()


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("control", type=Path)
    a.add_argument("cli", type=Path)
    a.add_argument("rev")
    p = a.parse_args()
    main(p.control, p.cli, p.rev)
