"""Retry the official v6e assignment with backoff and meter a real G11 workload if granted."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from g7_inputs import FED, shell, stream
from g9_storage import write


def main(ctl, repo, rev):
    name = "slwm-g11-v6-scale"
    env = dict(
        os.environ,
        CLOUDSDK_CONFIG=str(Path.home() / ".config/slwm-colab/gcloud"),
        COLAB_CLI_REQUEST_TIMEOUT="50",
    )
    current = Path("/private/tmp/slwm-g11-colab080-env/bin/colab")
    cli = str(current if current.exists() else Path.home() / ".local/bin/colab")
    os.environ.update(CLOUDSDK_CONFIG=env["CLOUDSDK_CONFIG"], COLAB_CLI=cli)
    log = ctl / "v6-attempts.jsonl"
    t0 = time.time()
    for attempt in range(3):
        if (ctl / "hardware-tests-done").exists() or time.time() > t0 + 65 * 60:
            return
        listing = subprocess.run(
            [cli, "--auth", "adc", "sessions"],
            env=env,
            capture_output=True,
            text=True,
            timeout=70,
        )
        if "[" + name + "]" not in listing.stdout:
            try:
                result = subprocess.run(
                    [cli, "--auth", "adc", "new", "-s", name, "--tpu", "v6e1"],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=270,
                )
                diagnostic = " ".join(
                    (result.stdout + " " + result.stderr).splitlines()[-3:]
                )[-360:]
            except subprocess.TimeoutExpired:
                diagnostic = "Official CLI allocation exceeded 270 seconds"
            listing = subprocess.run(
                [cli, "--auth", "adc", "sessions"],
                env=env,
                capture_output=True,
                text=True,
                timeout=70,
            )
            with log.open("a") as out:
                out.write(
                    json.dumps(
                        dict(
                            utc=time.time(),
                            attempt=attempt + 1,
                            assigned=("[" + name + "]" in listing.stdout),
                            provider=diagnostic,
                        )
                    )
                    + "\n"
                )
        if "[" + name + "]" in listing.stdout:
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
                        "75",
                    ],
                    check=True,
                    timeout=120,
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
                    ["git", "archive", rev, *paths],
                    "set -eu; mkdir -p /content/g11-code; cd /content/g11-code; tar -xf -",
                )
                origin = "/mnt/slwm-data/SLWM/g11/workload.npz"
                digest = subprocess.check_output(
                    [*FED, "sha256sum " + origin], text=True
                ).split()[0]
                stream(
                    name,
                    [*FED, "cat " + origin],
                    "set -eu; cd /content/g11-code; cat > workload.npz; echo '"
                    + digest
                    + "  workload.npz' | sha256sum -c -",
                )
                shell(
                    name,
                    "set -eu; cd /content/g11-code; source /content/g4-topology.env; export PYTHONPATH=. JAX_PLATFORMS=tpu; python -u scripts/train/g11_workload.py run workload.npz /content/g11-result.json",
                    timeout=1400,
                )
                d = json.loads(
                    shell(
                        name,
                        "cat /content/g11-result.json",
                        text=True,
                        capture_output=True,
                        timeout=90,
                    ).stdout
                )
                if "v6" not in d["device"].lower() or d["data_sha256"] != digest:
                    raise ValueError("Unverified v6 device or input")
                write(ctl / "v6-throughput.json", d)
                return
            finally:
                subprocess.run(
                    [cli, "--auth", "adc", "stop", "-s", name],
                    env=env,
                    timeout=120,
                    check=True,
                )
        # End promptly if the independent local study finishes while capacity is unavailable.
        for _ in range(60):
            if (ctl / "hardware-tests-done").exists():
                return
            time.sleep(10)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ctl", type=Path)
    p.add_argument("repo", type=Path)
    p.add_argument("revision")
    a = p.parse_args()
    main(a.ctl, a.repo, a.revision)
