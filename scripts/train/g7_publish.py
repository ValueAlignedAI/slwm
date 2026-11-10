"""Publish the completed G7 study once, after the scientific gates, physical verification and session release."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from g7_release_report import make

REMOTE="db@192.168.1.200"
CHECKS="/mnt/slwm-checkpoints-a/SLWM/g7"
REPORT=Path("experiments/multimodal/t7/continuation-alignment-control-20261110")
BRANCH="training/g7-continuation-alignment-control-20261110"

def run(args,**kw):return subprocess.run(args,check=True,**kw)
def remote(c,timeout=120):
    return run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=10",REMOTE,c],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=timeout).stdout
def store(file,data):
    tmp=file.with_suffix(".partial");tmp.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n");tmp.replace(file)
def compact(v):return datetime.datetime.fromtimestamp(v,datetime.timezone.utc).isoformat()
def sync(source,target):
    run(["rsync","-a",REMOTE+":"+source,str(target)],timeout=240)


def publish(repo,ctl):
    os.chdir(repo);status=ctl/"publication.json"
    if subprocess.check_output(["git","branch","--show-current"],text=True).strip()!=BRANCH:raise ValueError("Research branch changed")
    tag="cd /home/db/dev/slwm-g7 && OPENBLAS_NUM_THREADS=1 /home/db/dev/slwm-g4/.jax-cpu/bin/python -m scripts.train.g7_release_data"
    deadline=time.time()+11*3600
    while time.time()<deadline:
        try:
            panel=json.loads(remote(tag+" --status"))
            meter=json.loads((ctl/"budget.json").read_bytes())
            ready=panel=={"original_rungs":48,"physical_readback":48,"fresh_evaluations":80,"alignment_heads":8,"supervised_tests":20,"policy_tests":4,
                "camera_tests":6,"planner_tests":4,"physical_timing_tests":12,"live_examples":12}
            store(status,{"phase":"validating","scientific":panel,"meter_phase":meter["phase"],"project_units":meter["project_spend"],"utc":compact(time.time())})
            if ready and meter["phase"]=="complete":break
        except (subprocess.SubprocessError,OSError,ValueError) as exc:
            store(status,{"phase":"reconnecting","reason":str(exc)[:180],"utc":compact(time.time())})
        time.sleep(45)
    else:raise TimeoutError("Scientific release gates not all reached")
    store(status,{"phase":"assembling","utc":compact(time.time())})
    remote(tag+" --final",1500)
    remote("cd /home/db/dev/slwm-g7 && /home/db/dev/slwm/.research-venv/bin/python scripts/train/g7_figure.py "+CHECKS+"/public/scale-curves.json "+CHECKS+"/public/scale-curves.png",240)
    remote("cd /home/db/dev/slwm-g7 && python3 -c 'from scripts.train.g7_mirror import mirror; print(mirror())'",900)
    (REPORT/"metrics").mkdir(exist_ok=True)
    names=("scale-curves","matched-contrasts","physical-timing","camera-observability","checkpoint-inventory","continuous-generation",
           "release-status","alignment","camera","closed-loop","data","planning","retrieval","supervised","timing-audit")
    for name in names:sync(CHECKS+"/public/"+name+".json",REPORT/"metrics"/(name+".json"))
    sync(CHECKS+"/public/scale-curves.png",REPORT/"scale-curves.png")
    for core in ("hybrid","gpt2"):
        folder=REPORT/"examples"/core;folder.mkdir(exist_ok=True,parents=True)
        tag_run=f"g7-v5e-anchor-{core}-511-fresh"
        receipt=json.loads((REPORT/"metrics/continuous-generation.json").read_bytes())[tag_run]
        for name in ("audio-to-video.webm","video-to-audio.ogg","text-to-audio-video.webm"):
            sync(f"{CHECKS}/live/{tag_run}/previews/"+name,folder/name)
            import hashlib
            if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=receipt["previews"][name]["sha256"]:raise ValueError("Published preview hash differs")
    owned=[{k:d[k] for k in ("name","hardware","seconds","rate","units")} for d in meter["project_endpoints"].values()]
    if any(v["name"] not in ("slwm-g7-a-1110","slwm-g7-b5-1110","slwm-g7-b-1110") for v in owned):raise ValueError("Unrelated compute in project ledger")
    archive=json.loads(remote("cat "+CHECKS+"/readouts-archive.json"))
    drive=json.loads(remote("cat "+CHECKS+"/drive-state.json"))
    expense={"project_compute_units":round(meter["project_spend"],5),"limit":meter["credit_limit"],"total_runtime_hours":sum(d["seconds"] for d in owned)/3600,
        "native_assignments":owned,"attribution":"Owned endpoint duration multiplied by observed generation rate; polling boundaries are conservative.",
        "all_owned_assignments_released":not meter["active"],"readout_archive_verified_files":archive["verified_files"],
        "readout_archive_bytes":archive["bytes"],"private_drive_status":{k:v for k,v in drive.items() if k in ("phase","reason","verified_files","pending_files")}}
    store(REPORT/"metrics/project-compute.json",expense);make(REPORT)
    paths=[str(REPORT/"metrics"/(n+".json")) for n in (*names,"project-compute")]
    paths+=[str(REPORT/p) for p in ("report.md","scale-curves.png","examples/hybrid","examples/gpt2")]
    run(["git","add","--",*paths]);has=subprocess.run(["git","diff","--cached","--quiet","--",*paths]).returncode
    if has==1:run(["git","commit","--only","-m","Publish verified 16,384-update continuations, equal-time comparisons and sustained inference artifacts","--",*paths])
    elif has!=0:raise RuntimeError("Could not verify release difference")
    env=dict(os.environ,GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/Users/db/dev/slwm-g2/artifacts/predictive_world/g2/github_known_hosts")
    run(["git","push","origin","HEAD:refs/heads/"+BRANCH],env=env)
    ident=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    head=subprocess.check_output(["git","ls-remote","origin","refs/heads/"+BRANCH],env=env,text=True).split()[0]
    if head!=ident:raise ValueError("GitHub head differs from local release")
    store(status,{"phase":"published","commit":ident,"project_units":expense["project_compute_units"],"scientific":panel,"utc":compact(time.time())})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--repo",type=Path,required=True);p.add_argument("--ctl",type=Path,required=True);a=p.parse_args()
    with (a.ctl/"publication.lock").open("a") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);publish(a.repo,a.ctl)
