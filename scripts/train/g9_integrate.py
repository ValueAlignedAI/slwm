"""Publish an exact-tree-reviewed G9 milestone and journal the verified main commit."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
from g7_integrate_release import git,integrate
from g9_storage import write

BASE="experiments/multimodal/t9/concurrent-memory-20261110"
BRANCH="training/g9-concurrent-memory-20261110"


def record(repo,ctl,event,receipt,finished=False):
    root=Path.home()/".local/state/slwm";now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (root/"research-state.lock").open("a") as fd:
        fcntl.flock(fd,fcntl.LOCK_EX);p=root/"research-state.json";d=json.loads(p.read_text());g=d["studies"]["G9"]
        g.setdefault("merged_milestones",{})[str(receipt["number"])]=receipt
        d.setdefault("integration",{}).setdefault("completed_merges",{})[str(receipt["number"])]=receipt
        d["integration"].update(last_verified_main=receipt["main_sha"],phase="completed" if finished else "integrating_verified_studies")
        g["phase"]="integrated" if finished else "training_and_validation";d["updated_utc"]=now
        if finished:
            (ctl/"finalized").touch();g["release"]=receipt;g["owned_session_names"]=[]
            budget=json.loads((ctl/"budget.json").read_text());spent=budget["project_spend"]
            g.update(owned_credits_this_study=round(spent,5),cumulative_owned_credits=round(spent+g["prior_project_units"],5),
                     remaining_under_original_allowance=round(100-spent-g["prior_project_units"],5))
            d["active_goal"]="G9 integrated with verified matched controls, inference artifacts and new-cohort follow-ups."
        item=dict(utc=now,study="G9",event=event);d["last_event"]=item
        with (root/"research-journal.jsonl").open("a") as f:f.write(json.dumps(item)+"\n")
        write(p,d);os.chmod(p,0o600)
    write(ctl/("final-merge.json" if finished else "pilot-merge.json"),receipt)


def main(a):
    os.chdir(a.repo)
    if a.stage=="pilot":
        report=json.loads(git("show",a.head+":"+BASE+"/metrics/interim.json"))
        pilots={key.split("/")[0] for key in report["rows"] if key.startswith("pilot-")}
        if len(pilots)!=22 or "confirm-hybrid-z128-gelu-s912-fresh/4096/test" not in report["comparisons"]["hybrid_minus_dense"]:raise ValueError("Missing matched interim milestone")
        title="Publish G9 concurrent sensing screen and first matched sustained-control result"
        body="Releases the 22-fit small-screen milestone and first independently scored 116.70M pair. Includes sensor-identical action histories, a physical positive control, a real four-stream recording inventory, source masks, paired figures and a clearly labeled one-seed result. Full-history, equal-exposure, remaining seeds and robot transfer are tracked in the same study."
    else:
        report=json.loads(git("show",a.head+":"+BASE+"/metrics/release.json"))
        if not report["complete"] or not report["all_owned_released"] or not report["two_disk_checkpoints"]==54:raise ValueError("Missing complete stage custody")
        title="Publish verified G9 shared-state, full-history and equal-exposure experiments"
        body="Releases all matched training seeds, real cross-sensor scores, 51.2-second held interactive missions, history interventions, matched data exposure, independent robot contact transfer, decision adaptation and reproducible inference artifacts. Includes a project-only compute ledger, independently verified checkpoint hashes and revised research priorities."
    print(json.dumps(integrate(a.repo,a.control,BRANCH,a.head,title,body,final=a.stage=="final",recorder=record)),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("stage",choices=["pilot","final"]);ap.add_argument("--repo",required=True,type=Path)
    ap.add_argument("--control",required=True,type=Path);ap.add_argument("--head",required=True);main(ap.parse_args())
