"""Update the shared private research journal and cumulative allowance without changing older studies."""
import datetime
import fcntl
import json
import os
from pathlib import Path


def update(ctl,event=None,phase=None,receipt=None):
    root=Path.home()/".local/state/slwm";path=root/"research-state.json";now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (root/"research-state.lock").open("a") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        d=json.loads(path.read_bytes());g=d["studies"]["G8"];b=json.loads((ctl/"budget.json").read_bytes())
        g.update(owned_session_names=b["active"],owned_credits_this_study=round(b["project_spend"],5),cumulative_owned_credits=round(g["prior_project_units"]+b["project_spend"],5))
        g["remaining_under_original_allowance"]=round(100-g["cumulative_owned_credits"],5)
        vals=[json.loads(f.read_bytes()) for f in ctl.glob("*-runtime.json")]
        g["training_progress"]={"bridges":sum(len(v["bridges"]) for v in vals),"frozen_native_rungs":sum(len(v["stages"]) for v in vals),
            "two_disk_receipts":len(list((ctl/"receipts").glob("*.json"))),"lanes":[v["lane"] for v in vals]}
        if phase:g["phase"]=phase
        elif g.get("phase")=="registered" and vals:g["phase"]="training_and_custody"
        d["current_study"]="G8";d["updated_utc"]=now
        if receipt:
            d["integration"].setdefault("completed_merges",{})[str(receipt["number"])]=receipt
            d["integration"].update(phase="completed",last_verified_main=receipt["main_sha"]);g["in_flight"]=[];g["release"]=receipt
            d["active_goal"]="G8 factorial study complete, artifacts verified, README and revised research priorities integrated."
        if event:
            item={"utc":now,"study":"G8","event":event};d["last_event"]=item
            with (root/"research-journal.jsonl").open("a") as f:f.write(json.dumps(item,sort_keys=True)+"\n")
        tmp=path.with_suffix(".new");tmp.write_text(json.dumps(d,sort_keys=True,indent=2)+"\n");os.chmod(tmp,0o600);tmp.replace(path)
