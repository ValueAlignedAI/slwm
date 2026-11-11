"""Review the exact proposed trunk and keep the private study state current after verified milestones."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from g7_integrate_release import integrate, git
from g9_storage import write

BASE = Path("experiments/multimodal/t13/grounded-transfer-20261111")
BRANCH = "training/g13-grounded-transfer-20261111"


def record(repo, ctl, event, receipt, finished=False):
    state_root = Path.home() / ".local/state/slwm"
    now = datetime.now(timezone.utc).isoformat()
    with (state_root / "research-state.lock").open("a") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX)
        path = state_root / "research-state.json"
        doc = json.loads(path.read_bytes())
        g = doc["studies"]["G13"]
        g.setdefault("merged_milestones", {})[str(receipt["number"])] = receipt
        g.update(
            phase="integrated" if finished else "direct_readout_and_release",
            latest_verified_main=receipt["main_sha"],
            prospective_action_gain_mm=4.06756089210529,
            prospective_parent_groups=196,
        )
        if finished:
            d = json.loads((repo / BASE / "metrics/release.json").read_bytes())
            g.update(
                release=receipt,
                neural_fits=d["neural_fits"],
                two_disk_generations=d["generations"],
                portable_models=d["portable_models"],
                owned_session_names=[],
                owned_credits_this_study=d["owned_units"],
                cumulative_owned_credits=d["cumulative_owned_units"],
                remaining_under_authorized_allowance=d["remaining_units"],
                next_decision=d["next_experiment"],
            )
        item = dict(utc=now, study="G13", event=event)
        doc.update(
            current_study="G13",
            updated_utc=now,
            last_event=item,
            active_goal=(
                "G13 series completed and merged."
                if finished
                else "Finish the preregistered G13 action readout, natural components and verified inference release."
            ),
        )
        integration = doc.setdefault("integration", {})
        integration.setdefault("completed_merges", {})[str(receipt["number"])] = receipt
        integration.update(
            last_verified_main=receipt["main_sha"],
            phase="completed" if finished else "integrating_verified_studies",
        )
        with (state_root / "research-journal.jsonl").open("a") as f:
            f.write(json.dumps(item) + "\n")
        write(path, doc)
        os.chmod(path, 0o600)
    write(ctl / (("final" if finished else "prospective") + "-merge.json"), receipt)
    if finished:
        (ctl / "finalized").touch()


def main(stage, repo, ctl, head):
    os.chdir(repo)
    final = stage == "final"
    target = "release.json" if final else "prospective.json"
    r = json.loads(git("show", head + ":" + str(BASE / "metrics" / target)))
    if final:
        if not r["complete"] or not r["all_owned_released"]:
            raise ValueError("Incomplete G13")
        title = "Release the verified G13 physical, speech and shared-field transfer studies"
        body = "Publishes all matched arms and controls, open-vocabulary natural speech, independently reproduced physical outcomes, the final action readout, original-waveform and chronological inference artifacts, multiple audits and the updated project README."
    else:
        if not isinstance(r["passed"], bool) or len(r["seed_results"]) != 5:
            raise ValueError("Missing prospective result")
        title = (
            "Publish the prospective G13 action forecast against matched dense GPT-2"
        )
        body = "Reports the frozen five-seed action comparison on four independent real-robot shards, exact predictions and paired source uncertainty. Includes the stronger raw reference and explicitly tracks the follow-up that matches its training target."
    print(
        json.dumps(
            integrate(
                repo, ctl, BRANCH, head, title, body, final=final, recorder=record
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prospective", "final"))
    p.add_argument("repo", type=Path)
    p.add_argument("control", type=Path)
    p.add_argument("head")
    a = p.parse_args()
    main(a.stage, a.repo, a.control, a.head)
