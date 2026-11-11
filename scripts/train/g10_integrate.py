"""Integrate a G10 scientific milestone after the exact Git merge tree passes the Fedora suite."""

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
from g7_integrate_release import git, integrate
from g9_storage import write

BASE = "experiments/multimodal/t10/context-alignment-20261111"
BRANCH = "training/g10-context-alignment-20261111"


def record(repo, ctl, event, receipt, finished=False):
    root = Path.home() / ".local/state/slwm"
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (root / "research-state.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "research-state.json"
        d = json.loads(path.read_text())
        g = d["studies"]["G10"]
        g.setdefault("merged_milestones", {})[str(receipt["number"])] = receipt
        d.setdefault("integration", {}).setdefault("completed_merges", {})[
            str(receipt["number"])
        ] = receipt
        d["integration"].update(
            last_verified_main=receipt["main_sha"],
            phase="completed" if finished else "integrating_verified_studies",
        )
        g["phase"] = "integrated" if finished else "real_grounding_and_validation"
        d["updated_utc"] = now
        if finished:
            (ctl / "finalized").touch()
            g["release"] = receipt
            g["owned_session_names"] = []
            budget = json.loads((ctl / "budget.json").read_text())
            spent = budget["project_spend"]
            if (ctl / "recovery/budget.json").is_file():
                spent += json.loads((ctl / "recovery/budget.json").read_text())[
                    "project_spend"
                ]
            counts = json.loads((repo / BASE / "metrics/release.json").read_text())
            g.update(
                completed_fits=counts["total_fits"],
                two_disk_stages=counts["verified_generations"],
                owned_credits_this_study=round(spent, 5),
                cumulative_owned_credits=round(spent + g["prior_project_units"], 5),
                remaining_under_original_allowance=round(
                    100 - spent - g["prior_project_units"], 5
                ),
            )
            g["next_decision"] = (
                "Retain explicit live goals, keep dense and gated matched-context controls, and ground spoken instructions in subsequent measured actions and outcomes on new task families."
            )
            d["active_goal"] = (
                "G10 context, public multimodal alignment and decision results integrated."
            )
        item = dict(utc=now, study="G10", event=event)
        d["last_event"] = item
        with (root / "research-journal.jsonl").open("a") as f:
            f.write(json.dumps(item) + "\n")
        write(path, d)
        os.chmod(path, 0o600)
    write(ctl / (("final" if finished else "context") + "-merge.json"), receipt)


def main(a):
    os.chdir(a.repo)
    final = a.stage == "final"
    if final:
        r = json.loads(git("show", a.head + ":" + BASE + "/metrics/release.json"))
        if not r["complete"] or not r["all_owned_released"]:
            raise ValueError("G10 release is incomplete")
        title = (
            "Publish verified G10 public multimodal grounding and live decision results"
        )
        body = "Releases the matched real-alignment study, recording-disjoint human task outcomes, independent instruction and context controls, runnable artifacts, custody hashes, compute ledger and updated scientific conclusions."
    else:
        r = json.loads(git("show", a.head + ":" + BASE + "/metrics/context.json"))
        if r["fits"] != 27 or r["verified_generations"] != 54:
            raise ValueError("Context factorial incomplete")
        title = "Publish G10 equal-exposure context training and recurrent control comparison"
        body = "Reports all 27 matched context fits, three training seeds, uninterrupted held controls, actuator changes, delayed histories and paired uncertainty. The public HoloAssist grounding study continues separately."
    print(
        json.dumps(
            integrate(
                a.repo,
                a.control,
                BRANCH,
                a.head,
                title,
                body,
                final=final,
                recorder=record,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("context", "final"))
    ap.add_argument("repo", type=Path)
    ap.add_argument("control", type=Path)
    ap.add_argument("head")
    a = ap.parse_args()
    main(a)
