"""Assemble the complete checked G11 release, including the 600-unit authorization and real hardware evidence."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from g7_inputs import FED
from g9_storage import write
from g11_writeup import build, FOLDER

BASE = "/mnt/slwm-data/SLWM/g11"


def fed(command, timeout=120):
    return subprocess.check_output([*FED, command], text=True, timeout=timeout)


def ready():
    code = """from pathlib import Path
import json
p=Path('/mnt/slwm-data/SLWM/g11');a=Path('/mnt/slwm-checkpoints-a/SLWM/g11')
print(json.dumps(dict(primary=len(list((a/'runs').glob('*/step-*/verified.json'))),scale=len(list((a/'scale').glob('*/step-*/verified.json'))),
  gated=len(list((a/'scale-parents').glob('*-gated-*/step-*/verified.json'))),tested=len(list((p/'eval/predictions').glob('*-1024.json'))),
  scale_tested=len(list((p/'eval/scale-predictions').glob('*-1024.json'))),probed=len(list((p/'eval/cue-probes').glob('*.json'))),
  gap=len(list((p/'eval/training-gap').glob('*.json'))),selected=len(list((p/'eval/development-selected').glob('*.json'))),
  input_copies=(a/'inputs/verified.json').exists(),cli=(p/'release/cli-verified.json').exists())))"""
    return json.loads(fed("python3 -c " + shlex.quote(code)))


def main(repo, ctl):
    folder = repo / FOLDER
    m = folder / "metrics"
    end = json.loads((ctl / "bounds.json").read_text())["deadline"] + 1800
    expected = dict(
        primary=162,
        scale=72,
        gated=16,
        tested=54,
        scale_tested=24,
        probed=18,
        gap=18,
        selected=54,
        input_copies=True,
        cli=True,
    )
    while time.time() < end:
        try:
            snapshot = ready()
            budget = json.loads((ctl / "budget.json").read_text())
            write(
                ctl / "release-progress.json",
                dict(
                    counts=snapshot,
                    active=budget["active"],
                    project_spend=budget["project_spend"],
                ),
            )
            if (
                snapshot == expected
                and budget["phase"] == "complete"
                and not budget["active"]
            ):
                break
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        time.sleep(12)
    else:
        raise RuntimeError("G11 release readiness timeout")
    command = "cd /home/db/dev/slwm-g11 && export PYTHONPATH=. OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu && "
    fed(
        command
        + "/home/db/dev/slwm-g4/.jax-cpu/bin/python scripts/train/g11_scientific_release.py "
        + BASE
        + " "
        + BASE
        + "/public",
        timeout=900,
    )
    fed(
        command
        + "MPLCONFIGDIR="
        + BASE
        + "/mpl /home/db/dev/slwm/.research-venv/bin/python scripts/train/g11_figure.py "
        + BASE
        + "/public "
        + BASE
        + "/g11-comparisons.png",
        timeout=120,
    )

    def pull(source, dst):
        subprocess.run(
            ["rsync", "-a", "db@192.168.1.200:" + source, str(dst)],
            check=True,
            timeout=150,
        )

    pull(BASE + "/public/", m)
    pull(BASE + "/g11-comparisons.png", folder / "g11-comparisons.png")
    pull(BASE + "/eval/rtx3090-throughput.json", m / "rtx3090-throughput.json")
    cuda = json.loads((m / "rtx3090-throughput.json").read_text())
    native = ctl / "v6-throughput.json"
    attempts = [
        dict(
            client="0.7.5",
            hardware="v6e1",
            result="official allocation read timeout",
            when="request initiated before upgraded balance confirmation",
        ),
        dict(
            client="0.7.5",
            hardware="v6e1",
            result="provider assignment POST returned Service Unavailable",
            when="after upgrade",
        ),
        dict(
            client="0.7.5",
            hardware="v5e1",
            result="provider assignment POST returned Service Unavailable",
            when="after upgrade",
        ),
        dict(
            client="0.8.0",
            hardware="v6e1",
            result="provider assignment POST returned Service Unavailable",
            when="after upgrade, isolated verified distribution",
        ),
    ]
    log = ctl / "v6-attempts.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            d = json.loads(line)
            msg = d["provider"].lower()
            summary = (
                "assigned"
                if d["assigned"]
                else "Service Unavailable"
                if "service unavailable" in msg
                else "allocation timeout"
                if "270 seconds" in msg or "timeout" in msg
                else "no verified assignment"
            )
            attempts.append(
                dict(client="0.8.0", hardware="v6e1", result=summary, utc_unix=d["utc"])
            )
    hardware = dict(attempts=attempts, rtx3090=cuda, v6_verified=native.is_file())
    if native.is_file():
        v = json.loads(native.read_text())
        if v["data_sha256"] != cuda["data_sha256"]:
            raise ValueError("Hardware fixtures differ")
        hardware.update(
            v6=v,
            throughput_ratios={
                k: v["cases"][k]["positions_per_second"] / r["positions_per_second"]
                for k, r in cuda["cases"].items()
            },
            interpretation="A named v6e assignment ran and returned a checksum-matched measured workload.",
        )
    else:
        hardware["interpretation"] = (
            "Google returned assignment timeouts or Service Unavailable on v6e after the upgrade. No v6e speed ratio was observed."
        )
    ledger = [
        {k: r[k] for k in ("name", "hardware", "units", "seconds", "rate", "active")}
        for r in budget["project_endpoints"].values()
    ]
    used = budget["project_spend"]
    prior = 46.15995692341765
    if (
        any(v["active"] for v in ledger)
        or abs(sum(v["units"] for v in ledger) - used) > 1e-6
        or used + prior > 600
    ):
        raise ValueError("Invalid owned budget ledger")
    drv = json.loads(fed("cat /mnt/slwm-checkpoints-a/SLWM/g11/drive-state.json"))
    source = json.loads((m / "scientific-verdict.json").read_text())
    rel = {
        **source,
        "complete": True,
        "all_owned_released": True,
        "owned_units": used,
        "prior_owned_units": prior,
        "cumulative_owned_units": prior + used,
        "authorized_project_allowance": 600,
        "remaining_project_allowance": 600 - prior - used,
        "project_endpoints": ledger,
        "hardware": hardware,
        "drive_at_release": {
            k: v
            for k, v in drv.items()
            if k in ("phase", "reason", "pending", "verified_files")
        },
        "release_scope": "named G11 endpoints only",
    }
    write(m / "release.json", rel)
    write(m / "hardware.json", hardware)
    build(repo)
    write(ctl / "assembled.json", dict(ready=True, owned_units=used))
    print(json.dumps(source), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=Path)
    ap.add_argument("control", type=Path)
    a = ap.parse_args()
    main(a.repo.resolve(), a.control.resolve())
