"""Publish the G8 factorial release once its complete scientific, hardware and custody checks pass."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from g8_state import update
from g8_writeup import report,readme

HOST="db@192.168.1.200"
SOURCE="/mnt/slwm-checkpoints-a/SLWM/g8"
REPORT=Path("experiments/multimodal/t8/factorial-transfer-20261110")
BRANCH="training/g8-factorial-transfer-20261110"
TARGETS={"models":20,"pushing_results":42,"reacher_results":23,"live_controllers":48,"retention":20,"initial_speech_panels":10,"rate_speech_panels":10}
ENV=dict(os.environ,GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/Users/db/dev/slwm-g2/artifacts/predictive_world/g2/github_known_hosts")


def store(p,d):
    q=p.with_suffix(".new");q.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n");q.replace(p)


def remote(script,timeout=120):
    return subprocess.check_output(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=9",HOST,script],stderr=subprocess.PIPE,text=True,timeout=timeout)


def git(*cmd):return subprocess.check_output(["git",*cmd],env=ENV,text=True).strip()


def main(repo,ctl):
    os.chdir(repo);status=ctl/"publication.json";deadline=time.monotonic()+10*3600
    python="/home/db/dev/slwm-g4/.jax-cpu/bin/python"
    command="cd /home/db/dev/slwm-g8 && OPENBLAS_NUM_THREADS=1 "+python+" -m scripts.train.g8_release "
    if git("branch","--show-current")!=BRANCH:raise ValueError("Not on G8 research branch")
    until_release=False
    while time.monotonic()<deadline:
        try:
            checks=json.loads(remote(command+"status"));meter=json.loads((ctl/"budget.json").read_bytes())
            store(status,{"phase":"validating","scientific":checks,"owned_units":meter["project_spend"],"utc_unix":time.time()})
            update(ctl)
            ready=checks["analysis"] and checks["archive_complete"] and all(checks[k]==v for k,v in TARGETS.items())
            if ready and meter["phase"]=="complete" and not meter["active"]:until_release=True;break
        except (subprocess.SubprocessError,OSError,ValueError) as e:
            store(status,{"phase":"reconnecting","diagnostic":str(e)[:155],"utc_unix":time.time()})
        time.sleep(40)
    if not until_release:raise TimeoutError("G8 release gates remain incomplete")
    store(status,{"phase":"rendering_and_verifying","utc_unix":time.time()})
    remote("cd /home/db/dev/slwm-g8 && OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy /home/db/dev/slwm-g5/.sim/bin/python -m scripts.train.g8_rollout_media",900)
    remote("cd /home/db/dev/slwm-g8 && /home/db/dev/slwm/.research-venv/bin/python scripts/train/g8_plot.py "+SOURCE+"/public/factorial.json "+SOURCE+"/public/decision-effects.png",200)
    remote("cd /home/db/dev/slwm-g8 && OPENBLAS_NUM_THREADS=1 "+python+" -m scripts.train.g8_archive",1100)
    remote(command+"assemble",500)
    receipt=json.loads(remote("cat "+SOURCE+"/public/release-ready.json"))
    for source,pin in receipt["public_sha256"].items():
        if source in ("factorial.json","raw-and-live.json","retention.json","paired-scenes.npz","artifact-inventory.json","rollout-examples.json"):
            target=REPORT/"metrics"/source
        elif source in ("decision-effects.png","examples/pusht.webm","examples/puck_barrier.webm"):target=REPORT/source
        else:raise ValueError("Unexpected scientific release artifact")
        target.parent.mkdir(exist_ok=True,parents=True)
        subprocess.run(["rsync","-a",HOST+":"+SOURCE+"/public/"+source,str(target)],check=True,timeout=200)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=pin:raise ValueError("Release payload checksum mismatch")
    prior=json.loads((ctl/"bounds.json").read_bytes())["prior_project_units"];units=meter["project_spend"]
    owned=[{k:v[k] for k in ("name","hardware","seconds","rate","units")} for v in meter["project_endpoints"].values()]
    names=set(json.loads((ctl/"assignments.json").read_bytes()))
    if any(i["name"] not in names for i in owned) or prior+units>100 or abs(sum(d["units"] for d in owned)-units)>1e-5:raise ValueError("Invalid owned hardware ledger")
    drive=json.loads(remote("cat "+SOURCE+"/drive-state.json"));inv=json.loads(remote("cat "+SOURCE+"/data/complete.json"))
    final={"complete":True,"all_owned_released":True,"study_units":round(units,5),"prior_project_units":prior,"cumulative_owned_units":round(prior+units,5),"ceiling":100,
           "remaining_under_original_allowance":round(100-prior-units,5),"owned_assignments":owned,"single_chip_hours":sum(item["seconds"] for item in owned)/3600,
           "scope":"Time of G8-named endpoints times their measured generation rates. Other processes excluded.","scientific":receipt["gates"],
           "private_drive_state":{k:v for k,v in drive.items() if k in ("phase","reason","verified_files","pending_files")},"new_data_mirrors":inv}
    store(REPORT/"metrics/release.json",final);summary=report(REPORT);readme(repo,summary,final)
    paths=[str(REPORT/"metrics"/n) for n in ("factorial.json","raw-and-live.json","retention.json","paired-scenes.npz","artifact-inventory.json","rollout-examples.json","release.json")]
    paths+=["README.md",str(REPORT/"report.md"),str(REPORT/"decision-effects.png"),str(REPORT/"examples/pusht.webm"),str(REPORT/"examples/puck_barrier.webm")]
    git("add","-f","--",*paths);changed=subprocess.run(["git","diff","--cached","--quiet","--",*paths]).returncode
    if changed==1:git("commit","--only","-m","Publish complete matched G8 transfer, new-task policies, verified weights and live inference artifacts","--",*paths)
    elif changed!=0:raise RuntimeError("Cannot verify scientific release diff")
    git("push","origin","HEAD:refs/heads/"+BRANCH);head=git("rev-parse","HEAD")
    if git("ls-remote","origin","refs/heads/"+BRANCH).split()[0]!=head:raise ValueError("Scientific release is not the registered remote head")
    store(status,{"phase":"published","commit":head,"study_units":final["study_units"],"cumulative_units":final["cumulative_owned_units"],"scientific":receipt["gates"],"utc_unix":time.time()})
    update(ctl,event="Full factorial evidence and physical decision replays verified and pushed.",phase="reviewing_for_main")
    subprocess.run([sys.executable,"-u","scripts/train/g8_integrate.py","factorial","--repo",str(repo),"--control",str(ctl),"--branch",BRANCH,"--head",head],check=True)
    store(status,{"phase":"integrated","commit":head,"main":git("rev-parse","origin/main"),"scientific":receipt["gates"],"study_units":final["study_units"],"cumulative_units":final["cumulative_owned_units"],"utc_unix":time.time()})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--repo",type=Path,required=True);p.add_argument("--control",type=Path,required=True);a=p.parse_args()
    with (a.control/"publication.lock").open("a") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);main(a.repo.resolve(),a.control.resolve())
