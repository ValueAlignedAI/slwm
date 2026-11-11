"""Publish exact-tree-tested G14 milestones and retain the receipts in the external research ledger."""

import argparse
from datetime import datetime, timezone
from functools import partial
import fcntl
import json
import os
from pathlib import Path
from g7_integrate_release import integrate, git
from g9_storage import write

BASE = Path("experiments/multimodal/t14/spoken-closed-loop-20261111")
BRANCH = "training/g14-spoken-closed-loop-20261111"


def record(repo, ctl, event, receipt, finished=False, milestone="speech"):
    root = Path.home() / ".local/state/slwm"
    now = datetime.now(timezone.utc).isoformat()
    with (root / "research-state.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        path = root / "research-state.json"
        doc = json.loads(path.read_bytes())
        study = doc["studies"]["G14"]
        study.setdefault("merged_milestones", {})[str(receipt["number"])] = receipt
        study.update(
            phase="integrated" if finished else "held_physical_evaluation",
            latest_verified_main=receipt["main_sha"],
            owned_session_names=[],
        )
        if finished:
            release = json.loads((repo / BASE / "metrics/release.json").read_bytes())
            study.update(
                release=receipt,
                neural_fits=release["fits"],
                primary_held_worlds=release["total_held_worlds"],
                primary_held_branches=release["total_held_branches"],
                physical_model_files=release["published_physical_seeds"],
                two_disk_generations=release["generations"],
                next_decision=release["next"],
                owned_credits_this_study=release["owned_credits"],
                cumulative_owned_credits=release["total_owned_credits"],
            )
        entry = dict(utc=now, study="G14", event=event)
        doc.update(
            current_study="G14",
            updated_utc=now,
            last_event=entry,
            active_goal=(
                "G14 completed and merged."
                if finished
                else "Complete the 320 paired held physical worlds, continuous missions and verified artifact release."
            ),
        )
        control = doc.setdefault("integration", {})
        control.setdefault("completed_merges", {})[str(receipt["number"])] = receipt
        control.update(
            last_verified_main=receipt["main_sha"],
            phase="completed" if finished else "integrating_verified_studies",
        )
        with (root / "research-journal.jsonl").open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
        write(path, doc)
        os.chmod(path, 0o600)
    write(ctl / (milestone + "-merge.json"), receipt)
    if finished:
        (ctl / "finalized").touch()


def main(stage, repo, ctl, head):
    os.chdir(repo)
    final = stage == "final"
    target = (
        BASE
        / "metrics"
        / {
            "speech": "speech-replay.json",
            "fields": "field-replay.json",
            "final": "release.json",
        }[stage]
    )
    d = json.loads(git("show", head + ":" + str(target)))
    if final:
        if not d["complete"]:
            raise ValueError("Unverified physical results")
        title = (
            "Release G14 executed spoken-goal comparisons and live tabletop missions"
        )
        physics = json.loads(
            git("show", head + ":" + str(BASE / "metrics/physical-results.json"))
        )
        fresh = physics["cohorts"]["fresh"]["by_policy_family"]
        paired = physics["preregistered_positive_seeds"]
        body = (
            f"Executed {d['total_held_branches']:,} paired physical branches on {d['total_held_worlds']} held tabletop starts. "
            f"On two fresh speakers the spatial field achieved {100*fresh['transport']['success']['value']:.1f}% selective placement, dense {100*fresh['gpt2']['success']['value']:.1f}%, and geometry {100*fresh['geometry']['success']['value']:.1f}%. "
            f"The spatial-minus-dense lower bound was positive at {paired['held']}/3 seeds in the eight-speaker cohort and {paired['fresh']}/3 in the new-speaker cohort. "
            "Includes the original speech substitutions, constant-field interventions, 16 continuous missions, independent native replays, real actuator calibration, all six deployable physical weights, and verified two-disk custody."
        )
    elif stage == "fields":
        if len(d["models"]) != 6 or any(len(x) != 5 for x in d["models"].values()):
            raise ValueError("Unverified world inference package")
        title = "Publish six matched G14 physical fields and the frozen spoken control panel"
        body = "Releases the independently checked inference weights, three original spoken alternatives, the public station map, the verified training and solver audits, and the frozen controls. The 320 held state comparisons and sustained missions are executing."
    else:
        if set(d["results"]) != {"features", "cpu", "cuda"}:
            raise ValueError("Unverified speech artifact")
        title = "Publish G14 fresh-speaker grounding and real actuator transfer"
        body = "Releases the prospective physical protocol, development interventions, original-audio inference, held speaker recognition, and a disjoint real-robot calibration study. The main closed-loop comparison is still in progress."
    print(
        json.dumps(
            integrate(
                repo,
                ctl,
                BRANCH,
                head,
                title,
                body,
                final=final,
                recorder=partial(record, milestone=stage),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("stage", choices=("speech", "fields", "final"))
    a.add_argument("repo", type=Path)
    a.add_argument("ctl", type=Path)
    a.add_argument("head")
    p = a.parse_args()
    main(p.stage, p.repo, p.ctl, p.head)
