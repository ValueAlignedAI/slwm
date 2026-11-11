"""Train the registered width-256 replica on a named native v6e if the provider assigns one."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import time

from g7_inputs import FED, stream, shell
from g9_storage import write

REMOTE = "/mnt/slwm-data/SLWM/g12"
NAME = "slwm-g12-scale-v6"
CORES = ("gpt2", "gated", "oscillator", "associative")


def receive(step):
    files = [
        f"{fold}-{c}-paired-1202/step-{step:05d}/{k}"
        for fold in ("A", "B")
        for c in CORES
        for k in ("config.json", "weights.npz", "optimizer.npz", "verified.json")
    ]
    command = "set -eu; cd /content/g12/temporary-0; tar -cf - " + " ".join(
        map(shlex.quote, files)
    )
    source = [
        "bash",
        "scripts/train/colab_session.sh",
        "ssh",
        NAME,
        "bash",
        "-c",
        command,
    ]
    dest = (
        "set -eu; mkdir -p "
        + REMOTE
        + "/tpu-in; tar -xf - -C "
        + REMOTE
        + "/tpu-in; cd "
        + REMOTE
        + "/code; python3 scripts/train/g12_custody.py import-tpu --step "
        + str(step)
    )
    with subprocess.Popen(source, stdout=subprocess.PIPE) as sender:
        try:
            subprocess.run([*FED, dest], stdin=sender.stdout, check=True, timeout=1800)
        finally:
            sender.stdout.close()
        if sender.wait(timeout=60):
            raise ValueError("Native artifact transport failed")


def main(control, cli, revision):
    os.environ.update(
        CLOUDSDK_CONFIG=str(Path.home() / ".config/slwm-colab/gcloud"),
        COLAB_CLI=str(cli),
        COLAB_CLI_REQUEST_TIMEOUT="65",
    )
    cmd = [str(cli), "--auth", "adc"]

    def assigned():
        listing = subprocess.check_output(cmd + ["sessions"], text=True, timeout=95)
        return any(line.startswith("[" + NAME + "] ") for line in listing.splitlines())

    assigned_here = False
    steps = []
    reason = "unassigned"
    attempts = []
    try:
        for attempt in range(2):
            began = time.time()
            msg = ""
            if not assigned():
                try:
                    o = subprocess.run(
                        cmd + ["new", "-s", NAME, "--tpu", "v6e1"],
                        capture_output=True,
                        text=True,
                        timeout=265,
                    )
                    msg = (o.stdout + " " + o.stderr).lower()
                except subprocess.TimeoutExpired:
                    msg = "timeout"
            assigned_here = assigned()
            code = (
                "assigned"
                if assigned_here
                else (
                    "service unavailable"
                    if "service unavailable" in msg
                    else (
                        "timeout"
                        if "timeout" in msg or "timed out" in msg
                        else "unassigned"
                    )
                )
            )
            attempts.append(
                dict(started=began, ended=time.time(), result=code, hardware="v6e1")
            )
            write(control / "attempts.json", attempts)
            if assigned_here:
                break
            if attempt == 0:
                time.sleep(140)
        if not assigned_here:
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
                "90",
            ],
            check=True,
            timeout=145,
        )
        paths = [
            "models/g12_world.py",
            "scripts/train/g12_fit.py",
            "scripts/train/g9_storage.py",
        ]
        stream(
            NAME,
            ["git", "archive", revision, *paths],
            "set -eu; mkdir -p /content/g12; cd /content/g12; tar -xf -",
        )
        source_files = [
            "banks/" + f + "-" + r + ext
            for f in ("A", "B")
            for r in ("train", "dev")
            for ext in (".json", ".npz")
        ] + ["state-labels.json"]
        stream(
            NAME,
            [*FED, "tar -C " + REMOTE + "/data -cf - " + " ".join(source_files)],
            "set -eu; mkdir -p /content/g12/data; tar -xf - -C /content/g12/data",
        )
        for phase, rungs in (("warm", (256,)), ("joint", (512, 1024))):
            shell(
                NAME,
                "set -eu; cd /content/g12; source /content/g4-topology.env; export PYTHONPATH=. JAX_PLATFORMS=tpu OMP_NUM_THREADS=2; python -u scripts/train/g12_fit.py "
                + phase
                + " --scale --seed 1202 --v6",
                timeout=5400,
            )
            for step in rungs:
                receive(step)
                steps.append(step)
        reason = "verified_native_training"
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        reason = type(exc).__name__
        print("Native lane ended:", type(exc).__name__, flush=True)
    finally:
        if assigned():
            subprocess.run(cmd + ["stop", "-s", NAME], check=True, timeout=140)
        if assigned():
            raise ValueError("Owned native session still active")
        (control / "requests-done").touch()
        write(
            control / "lane.json",
            dict(reason=reason, steps=steps, assigned=assigned_here, released=True),
        )
        subprocess.run(
            [*FED, "touch " + REMOTE + "/scale-routing-complete"],
            check=True,
            timeout=35,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("control", type=Path)
    p.add_argument("cli", type=Path)
    p.add_argument("revision")
    a = p.parse_args()
    main(a.control, a.cli, a.revision)
