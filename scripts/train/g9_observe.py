"""Separate deadline-limited status polling from large transfers and journal only owned G9 usage."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from g9_storage import write
from g7_inputs import FED


def journal(ctl,phase,counts,receipts):
    base=Path.home()/".local/state/slwm";now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (base/"research-state.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);path=base/"research-state.json";doc=json.loads(path.read_text())
        if (ctl/"finalized").exists():return
        g=doc["studies"]["G9"];b=json.loads((ctl/"budget.json").read_text())
        spent=b["project_spend"];g.update(phase=phase,owned_session_names=b["active"],owned_credits_this_study=round(spent,5),
            cumulative_owned_credits=round(spent+g["prior_project_units"],5),remaining_under_original_allowance=round(100-spent-g["prior_project_units"],5),
            training=counts,two_disk_confirm_receipts=receipts,updated_utc=now)
        doc["updated_utc"]=now;write(path,doc);os.chmod(path,0o600)


def main(ctl):
    deadline=json.loads((ctl/"bounds.json").read_text())["deadline"]
    while time.time()<deadline and not (ctl/"finalized").exists():
        try:
            b=json.loads((ctl/"budget.json").read_text())
            counts={}
            for name in b["active"]:
                out=subprocess.check_output(["bash","scripts/train/colab_session.sh","exec",name,"-f","scripts/train/g9_inventory.py","--timeout","55"],stderr=subprocess.DEVNULL,text=True,timeout=80)
                status=json.loads(next(s[8:] for s in out.splitlines() if s.startswith("SLWM_G9 ")))
                write(ctl/(name+"-live.json"),status);counts[name]=dict(job=status["progress"].get("job"),step=status["progress"].get("step"),rungs=len(status["completed"]))
            prog=subprocess.check_output([*FED,"cat /mnt/slwm-checkpoints-a/SLWM/g9/runs/progress.json"],text=True,timeout=30)
            counts["fedora"]=json.loads(prog);count=len(list((ctl/"receipts").glob("*.json")))
            journal(ctl,"training_and_validation",counts,count)
        except (OSError,ValueError,StopIteration,subprocess.SubprocessError) as e:
            write(ctl/"observer-retry.json",dict(time=time.time(),error=str(e)[:180]))
        time.sleep(35)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("control",type=Path);main(p.parse_args().control)
