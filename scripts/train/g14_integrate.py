"""Publish exact-tree-tested G14 milestones and retain the receipts in the external research ledger."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from g7_integrate_release import integrate, git
from g9_storage import write

BASE = Path("experiments/multimodal/t14/spoken-closed-loop-20261111")
BRANCH = "training/g14-spoken-closed-loop-20261111"


def record(repo, ctl, event, receipt, finished=False):
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
    write(ctl / (("final" if finished else "speech") + "-merge.json"), receipt)
    if finished:
        (ctl / "finalized").touch()


def main(stage, repo, ctl, head):
    os.chdir(repo)
    final = stage == "final"
    target = BASE / "metrics" / ("release.json" if final else "speech-replay.json")
    d = json.loads(git("show", head + ":" + str(target)))
    if final:
        if not d["complete"]:
            raise ValueError("Unverified physical results")
        title = (
            "Release G14 executed spoken-goal comparisons and live tabletop missions"
        )
        body = "Publishes all frozen physical outcomes, same-scene counterfactuals, matched dense and spatial fields, strong control baselines, original-waveform inference, continuous mission videos, independent HDD receipts, and the updated research conclusions."
    else:
        if set(d["results"]) != {"features", "cpu", "cuda"}:
            raise ValueError("Unverified speech artifact")
        title = "Publish G14 fresh-speaker grounding and real actuator transfer"
        body = "Releases the prospective physical protocol, development interventions, original-audio inference, held speaker recognition, and a disjoint real-robot calibration study. The main closed-loop comparison is still in progress."
    print(
        json.dumps(
            integrate(
                repo, ctl, BRANCH, head, title, body, final=final, recorder=record
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("stage", choices=("speech", "final"))
    a.add_argument("repo", type=Path)
    a.add_argument("ctl", type=Path)
    a.add_argument("head")
    p = a.parse_args()
    main(p.stage, p.repo, p.ctl, p.head)
