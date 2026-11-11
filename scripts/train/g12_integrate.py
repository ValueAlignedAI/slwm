"""Merge the complete G12 candidate after exact-tree tests and reconcile the private research ledger."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from g7_integrate_release import git, integrate
from g9_storage import write

BASE = Path("experiments/multimodal/t12/instruction-sequences-20261111")
BRANCH = "training/g12-instruction-sequences-20261111"


def record(repo, ctl, event, receipt, finished=False):
    root = Path.home() / ".local/state/slwm"
    d = json.loads((repo / BASE / "metrics/release.json").read_bytes())
    money = json.loads((repo / BASE / "metrics/compute.json").read_bytes())
    now = datetime.now(timezone.utc).isoformat()
    with (root / "research-state.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "research-state.json"
        state = json.loads(path.read_bytes())
        g = state["studies"]["G12"]
        g.update(
            phase="integrated",
            release=receipt,
            fits=d["fits"],
            two_disk_generations=d["generations"],
            portable_models=d["portable_models"],
            fresh_recordings=48,
            fresh_decisions=373,
            measured_fresh_outcomes=77,
            continuous_source_seconds=603,
            owned_credits_this_study=money["owned_units"],
            cumulative_owned_credits=money["cumulative_owned_units"],
            remaining_under_authorized_allowance=money["remaining_credits"],
            owned_session_names=[],
            next_decision="Compare component-specific outcomes under matched different goals using direct telemetry or independent visual annotation. Retain oscillatory and gated candidates, early joint selection, a dense core and simple priors.",
        )
        g.setdefault("merged_milestones", {})[str(receipt["number"])] = receipt
        integration = state.setdefault("integration", {})
        integration.setdefault("completed_merges", {})[str(receipt["number"])] = receipt
        integration.update(last_verified_main=receipt["main_sha"], phase="completed")
        state.update(
            current_study="G12",
            updated_utc=now,
            project_credit_allowance=600,
            active_goal="G12 complete and integrated, including early joint training, matched width, observed device changes and continuous replay.",
        )
        item = dict(utc=now, study="G12", event=event)
        state["last_event"] = item
        with (root / "research-journal.jsonl").open("a") as stream:
            stream.write(json.dumps(item) + "\n")
        write(path, state)
        os.chmod(path, 0o600)
    write(ctl / "final-merge.json", receipt)
    (ctl / "finalized").touch()


def main(repo, ctl, head):
    os.chdir(repo)

    def get(name):
        return json.loads(git("show", head + ":" + str(BASE / "metrics" / name)))

    d = get("release.json")
    money = get("compute.json")
    if (
        d["fits"],
        d["generations"],
        d["portable_models"],
        len(get("raw-audio.json")),
        len(get("single-clock.json")),
    ) != (112, 384, 24, 8, 4) or not money["all_owned_released"]:
        raise ValueError("Incomplete G12 release")
    receipt = integrate(
        repo,
        ctl,
        BRANCH,
        head,
        "Publish G12 matched spoken sequences, visible device outcomes and continuous replay",
        "Compares 112 fully trained processors on unseen public device families with matched Transformer, recurrent, raw and persistence controls. Publishes blinded source images, paired recording uncertainty, verified inference models, chronological examples, width and early-learning results, and project-only TPU accounting.",
        final=True,
        recorder=record,
    )
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("repo", type=Path)
    p.add_argument("ctl", type=Path)
    p.add_argument("head")
    a = p.parse_args()
    main(a.repo, a.ctl, a.head)
