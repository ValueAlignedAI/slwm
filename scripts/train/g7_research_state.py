"""Keep the user-owned research goal, active evidence and verified main-branch history outside the checkout."""
import datetime
import fcntl
import json
from pathlib import Path
import os


def record(repo,ctl,event=None,merge=None,finished=False):
    base=Path.home()/".local/state/slwm";base.mkdir(exist_ok=True,parents=True,mode=0o700)
    path=base/"research-state.json"
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (base/"research-state.lock").open("a") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        data=json.loads(path.read_bytes()) if path.exists() else {}
        study=data.setdefault("studies",{}).setdefault("G7",{})
        history=data.setdefault("integration",{}).setdefault("completed_merges",{})
        if merge:
            history[str(merge["number"])]=merge;data["integration"]["last_verified_main"]=merge["main_sha"];data["integration"]["phase"]="integrating_verified_studies"
        status=json.loads((ctl/"publication.json").read_bytes()) if (ctl/"publication.json").is_file() else {}
        meter=json.loads((ctl/"budget.json").read_bytes())
        study.update(owned_credits_as_of_update=round(meter["project_spend"],5),owned_sessions=meter["active"],
                     verified_counts=status.get("scientific",{}),publication_phase=status.get("phase","validating"))
        study["checkout"]=str(repo);study["project_credit_ceiling"]=100
        if finished:
            study["phase"]="integrated";study["in_flight"]=[];study["pending_decision"]="Select the next prospective study using the completed scale, waveform, timing and planning controls."
            data["integration"]["phase"]="completed";data["active_goal"]="G7 study verified, README current, published artifacts merged into main."
        if event:
            data["last_event"]={"utc":now,"description":event}
            with (base/"research-journal.jsonl").open("a") as log:log.write(json.dumps({"utc":now,"study":"G7","event":event,"merge":merge},sort_keys=True)+"\n")
        data["current_study"]="G7";data["updated_utc"]=now
        tmp=path.with_suffix(".new");tmp.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n");os.chmod(tmp,0o600);tmp.replace(path)
