"""Integrate verified G11 milestones using the exact-tree Fedora review and private project journal."""

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
from g7_integrate_release import git, integrate
from g9_storage import write

BASE = "experiments/multimodal/t11/instructions-physical-effects-20261111"
BRANCH = "training/g11-instruction-effects-20261111"


def record(repo, ctl, event, receipt, finished=False):
    root = Path.home() / ".local/state/slwm"
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (root / "research-state.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "research-state.json"
        d = json.loads(path.read_text())
        g = d["studies"]["G11"]
        g.setdefault("merged_milestones", {})[str(receipt["number"])] = receipt
        d.setdefault("integration", {}).setdefault("completed_merges", {})[
            str(receipt["number"])
        ] = receipt
        d["integration"].update(
            last_verified_main=receipt["main_sha"],
            phase="completed" if finished else "integrating_verified_studies",
        )
        g["phase"] = "integrated" if finished else "scaling_and_validation"
        d["updated_utc"] = now
        d["project_credit_allowance"] = 600
        g.update(
            primary_adapter_fits=54,
            verified_primary_generations=162,
            held_device_families=4,
            new_recordings=96,
            visual_audit_examples=259,
        )
        if finished:
            (ctl / "finalized").touch()
            g["release"] = receipt
            g["owned_session_names"] = []
            summary = json.loads((repo / BASE / "metrics/release.json").read_text())
            spent = summary["owned_units"]
            g.update(
                total_g11_fits=summary["total_fits"],
                two_disk_generations=summary["verified_generations"],
                owned_credits_this_study=round(spent, 5),
                cumulative_owned_credits=round(spent + g["prior_project_units"], 5),
                remaining_under_authorized_allowance=round(
                    600 - spent - g["prior_project_units"], 5
                ),
            )
            g["next_decision"] = (
                "Measure device-specific state transitions with independent instrumentation and distinguish the requested manipulation from preparatory actions."
            )
            d["active_goal"] = (
                "G11 verified and integrated, including true/wrong spoken interventions, physical evidence, matched scale and hardware checks."
            )
        item = dict(utc=now, study="G11", event=event)
        d["last_event"] = item
        with (root / "research-journal.jsonl").open("a") as f:
            f.write(json.dumps(item) + "\n")
        write(path, d)
        os.chmod(path, 0o600)
    write(ctl / (("final" if finished else "primary") + "-merge.json"), receipt)


def main(a):
    os.chdir(a.repo)
    final = a.stage == "final"
    if final:
        d = json.loads(git("show", a.head + ":" + BASE + "/metrics/release.json"))
        if not d["complete"] or not d["all_owned_released"]:
            raise ValueError("G11 incomplete")
        title = "Publish the G11 equal-exposure scale study, spoken decisions and checked physical measurements"
        body = "Releases primary and scale comparisons, independently reviewed device-centered sensitivity, matched recurrent and Transformer controls, public inference artifacts, the project-only compute receipt and the revised research conclusions."
    else:
        d = json.loads(git("show", a.head + ":" + BASE + "/metrics/primary.json"))
        if d["fits"] != 54 or len(d["gates"]) != 18:
            raise ValueError("Missing matched primary cells")
        title = "Publish G11 spoken-instruction tests on four held device families"
        body = "Reports all 54 bounded transfer fits, 838 prospective instruction decisions, genuine microphone audio and earlier same-recording instructions, independent 3D surface changes, object-center review, baseline comparisons and calibrated claim limits. The approved scale extension is tracked separately."
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
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("primary", "final"))
    p.add_argument("repo", type=Path)
    p.add_argument("control", type=Path)
    p.add_argument("head")
    a = p.parse_args()
    main(a)
