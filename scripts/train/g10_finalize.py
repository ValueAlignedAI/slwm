"""Assemble the checked research release after both named Colab waves and all Fedora panels complete."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from g7_inputs import FED
from g9_storage import write
from g10_writeup import build, FOLDER

BASE = "/mnt/slwm-data/SLWM/g10"


def remote(command, timeout=120):
    return subprocess.check_output([*FED, command], text=True, timeout=timeout)


def counts():
    code = """from pathlib import Path
import json
r=Path('/mnt/slwm-data/SLWM/g10');a=Path('/mnt/slwm-checkpoints-a/SLWM/g10')
print(json.dumps(dict(verified=len(list((a/'runs').glob('*/step-*/verified.json'))),control=len(list((r/'eval/control').glob('*.json'))),real=len(list((r/'eval/real').glob('*.json'))),
 forecast=len(list((r/'eval/forecast').glob('*.json'))),latched=len(list((r/'eval/latched').glob('*.json'))),cue=len(list((r/'eval/commands').glob('cue-*.json'))),
 compact=(a/'public-inference/verified.json').exists(),inputs=(a/'inputs/verified.json').exists())))"""
    return json.loads(remote("python3 -c " + shlex.quote(code)))


def run(repo, ctl):
    dest = repo / FOLDER
    m = dest / "metrics"
    deadline = json.loads((ctl / "bounds.json").read_text())["deadline"] + 1800
    goal = dict(
        verified=90,
        control=78,
        real=24,
        forecast=12,
        latched=9,
        cue=12,
        compact=True,
        inputs=True,
    )
    while time.time() < deadline:
        try:
            state = counts()
            budgets = [
                json.loads((p / "budget.json").read_text())
                for p in (ctl, ctl / "recovery")
            ]
            active = [name for b in budgets for name in b["active"]]
            spend = sum(b["project_spend"] for b in budgets)
            write(
                ctl / "release-progress.json",
                dict(counts=state, active=active, owned_units=spend),
            )
            if (
                state == goal
                and not active
                and all(b["phase"] == "complete" for b in budgets)
            ):
                break
        except (ValueError, OSError, subprocess.SubprocessError):
            pass
        time.sleep(15)
    else:
        raise RuntimeError("Final audit deadline")
    prefix = "cd /home/db/dev/slwm-g10 && export PYTHONPATH=. JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 && "
    remote(
        prefix
        + "/home/db/dev/slwm-g4/.jax-cpu/bin/python scripts/train/g10_scientific_release.py "
        + BASE
        + "/eval "
        + BASE
        + "/public",
        timeout=900,
    )
    remote(
        prefix
        + "MPLCONFIGDIR="
        + BASE
        + "/mpl /home/db/dev/slwm/.research-venv/bin/python scripts/train/g10_figure.py "
        + BASE
        + "/public "
        + BASE
        + "/g10-comparisons.png",
        timeout=120,
    )

    def pull(source, target):
        subprocess.run(
            ["rsync", "-a", "db@192.168.1.200:" + source, str(target)],
            check=True,
            timeout=120,
        )

    pull(BASE + "/public/", m)
    pull(BASE + "/g10-comparisons.png", dest / "g10-comparisons.png")
    heads = dest / "inference/risk-heads"
    heads.mkdir(exist_ok=True)
    for c in ("hybrid", "gpt2"):
        f = f"align-{c}-paired-s1003-risk-heads.npz"
        pull("/mnt/slwm-checkpoints-a/SLWM/g10/risk-heads/" + f, heads / f)
    pull(BASE + "/release/inference/risk-heads/receipts.json", heads / "receipts.json")
    pull(
        BASE + "/release/inference/risk-heads/packaging-check.json",
        heads / "packaging-check.json",
    )
    ledger = []
    for budget in budgets:
        for endpoint in budget["project_endpoints"].values():
            ledger.append(
                {
                    k: endpoint[k]
                    for k in ("name", "hardware", "units", "seconds", "rate", "active")
                }
            )
    prior = 37.501280916544204
    if (
        any(r["active"] for r in ledger)
        or abs(sum(r["units"] for r in ledger) - spend) > 1e-6
        or spend > 28
        or prior + spend > 100
    ):
        raise ValueError("Owned compute ledger mismatch")
    drv = json.loads(remote("cat /mnt/slwm-checkpoints-a/SLWM/g10/drive-state.json"))
    release = dict(
        **json.loads((m / "scientific-verdict.json").read_text()),
        complete=True,
        all_owned_released=True,
        owned_units=spend,
        prior_owned_units=prior,
        cumulative_owned_units=prior + spend,
        waves=ledger,
        drive_at_release={
            k: v
            for k, v in drv.items()
            if k in ("phase", "reason", "verified_files", "pending")
        },
        source_and_control_base="4273b75d952e027775ef5f340d55dec77235ae60",
        public_only=True,
        source_transfer_recovery="Original assignment disappeared before training-bank delivery. Replacement assignments verified prepared training data and four frozen parent identities per seed.",
    )
    write(m / "release.json", release)
    build(repo)
    write(ctl / "assembled.json", dict(complete=True, owned_units=spend))
    print(json.dumps(release, sort_keys=True), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("repo", type=Path)
    p.add_argument("control", type=Path)
    a = p.parse_args()
    run(a.repo.resolve(), a.control.resolve())
