"""Retry the study's own v6e assignment and verify a real optimizer workload on granted hardware."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from g7_inputs import FED, shell, stream
from g9_storage import write


def main(ctl, cli):
    name = "slwm-g12-v6"
    os.environ.update(
        CLOUDSDK_CONFIG=str(Path.home() / ".config/slwm-colab/gcloud"),
        COLAB_CLI=str(cli),
        COLAB_CLI_REQUEST_TIMEOUT="50",
    )
    command = [str(cli), "--auth", "adc"]

    def assigned():
        listing = subprocess.check_output(command + ["sessions"], text=True, timeout=85)
        return any(line.startswith("[" + name + "] ") for line in listing.splitlines())

    try:
        for attempt in range(1, 5):
            if (ctl / "release").exists():
                break
            started = time.time()
            diagnostic = ""
            if not assigned():
                try:
                    r = subprocess.run(
                        command + ["new", "-s", name, "--tpu", "v6e1"],
                        timeout=240,
                        capture_output=True,
                        text=True,
                    )
                    diagnostic = (r.stdout + " " + r.stderr).lower()
                except subprocess.TimeoutExpired:
                    diagnostic = "timeout"
            active = assigned()
            outcome = (
                "assigned"
                if active
                else (
                    "service unavailable"
                    if "service unavailable" in diagnostic
                    else (
                        "timeout"
                        if "timeout" in diagnostic or "timed out" in diagnostic
                        else "unassigned"
                    )
                )
            )
            with (ctl / "attempts.jsonl").open("a") as out:
                out.write(
                    json.dumps(
                        dict(
                            attempt=attempt,
                            started=started,
                            ended=time.time(),
                            hardware="v6e1",
                            result=outcome,
                        )
                    )
                    + "\n"
                )
            if active:
                try:
                    subprocess.run(
                        [
                            "bash",
                            "scripts/train/colab_session.sh",
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
                    paths = [
                        "models/g4_shared.py",
                        "models/g5_shared.py",
                        "models/g9_shared.py",
                        "models/g9_tabletop.py",
                        "models/g10_context.py",
                        "models/g11_instruction.py",
                        "scripts/train/g9_fit.py",
                        "scripts/train/g10_fit.py",
                        "scripts/train/g11_fit.py",
                        "scripts/train/g11_workload.py",
                    ]
                    stream(
                        name,
                        ["git", "archive", "7bc8193", *paths],
                        "set -eu; mkdir -p /content/g12; cd /content/g12; tar -xf -",
                    )
                    source = "/mnt/slwm-data/SLWM/g11/workload.npz"
                    digest = subprocess.check_output(
                        [*FED, "sha256sum " + source], text=True
                    ).split()[0]
                    stream(
                        name,
                        [*FED, "cat " + source],
                        "set -eu; cd /content/g12; cat > workload.npz; echo '"
                        + digest
                        + "  workload.npz' | sha256sum -c -",
                    )
                    shell(
                        name,
                        "set -eu; cd /content/g12; source /content/g4-topology.env; export PYTHONPATH=. JAX_PLATFORMS=tpu; python scripts/train/g11_workload.py run workload.npz result.json",
                        timeout=1250,
                    )
                    result = json.loads(
                        shell(
                            name,
                            "cat /content/g12/result.json",
                            text=True,
                            capture_output=True,
                            timeout=60,
                        ).stdout
                    )
                    if (
                        "v6" not in result["device"].lower()
                        or result["data_sha256"] != digest
                    ):
                        raise ValueError("Wrong hardware or workload")
                    write(ctl / "v6-throughput.json", result)
                finally:
                    subprocess.run(
                        command + ["stop", "-s", name], timeout=120, check=True
                    )
                break
            if attempt < 4:
                for _ in range(36):
                    if (ctl / "release").exists():
                        return
                    time.sleep(10)
    finally:
        if assigned():
            subprocess.run(command + ["stop", "-s", name], timeout=120, check=True)
        (ctl / "requests-done").touch()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ctl", type=Path)
    p.add_argument("cli", type=Path)
    a = p.parse_args()
    main(a.ctl, a.cli)
