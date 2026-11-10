"""Assemble the audited G9 release from the completed Fedora and named Colab workers."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from g7_inputs import FED
from g9_storage import write
from g9_writeup import main as writeup

REMOTE="/mnt/slwm-data/SLWM/g9"
RESULT=Path("experiments/multimodal/t9/concurrent-memory-20261110")


def ssh(cmd,timeout=120):
    return subprocess.check_output([*FED,cmd],text=True,timeout=timeout)


def ready():
    code="""import json,pathlib
s=pathlib.Path('/mnt/slwm-data/SLWM/g9/scores')
a=pathlib.Path('/mnt/slwm-checkpoints-a/SLWM/g9')
def count(role,keys):
 return sum(all(k in json.loads(p.read_bytes()) for k in keys) for p in s.glob('*-'+role+'.json'))
print(json.dumps(dict(primary=len(list((a/'runs').glob('*/step-*/verified.json'))),dose=len(list((a/'dose').glob('*/step-*/verified.json'))),
 held=count('test',('room','sim','live','history','contact')),development=count('dev',('room','sim','live','history')),
 source_invariance=len(list((s/'invariance').glob('*.json'))),uncapped=len(list((s/'uncapped').glob('*.json'))))))"""
    return json.loads(ssh("python3 -c "+shlex.quote(code)))


def run(repo,ctl):
    dest=repo/RESULT;metrics=dest/"metrics"
    deadline=json.loads((ctl/"bounds.json").read_text())["deadline"]+1800
    while time.time()<deadline:
        try:
            counts=ready();budget=json.loads((ctl/"budget.json").read_text())
            write(ctl/"release-progress.json",dict(counts=counts,phase=budget["phase"],active=budget["active"]))
            if counts==dict(primary=54,dose=4,held=38,development=54,source_invariance=8,uncapped=8) and not budget["active"] and budget["phase"]=="complete":break
        except (OSError,ValueError,subprocess.SubprocessError):pass
        time.sleep(15)
    else:raise RuntimeError("Release readiness deadline")
    prefix="cd /home/db/dev/slwm-g9 && export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 PYTHONPATH=. && "
    ssh(prefix+"/home/db/dev/slwm-g4/.jax-cpu/bin/python scripts/train/g9_analysis.py "+REMOTE+"/scores "+REMOTE+"/final --final",timeout=400)
    ssh(prefix+"MPLCONFIGDIR="+REMOTE+"/mpl /home/db/dev/slwm/.research-venv/bin/python scripts/train/g9_figure.py "+REMOTE+"/final/results.json "+REMOTE+"/final/g9-comparisons.png",timeout=190)
    def pull(remote,local):
        subprocess.run(["rsync","-a",f"db@192.168.1.200:{remote}",str(local)],check=True,timeout=140)
    for f in ("results.json","artifacts.json","paired-observations.npz","data-card.json","raw-oracle.json"):
        pull(REMOTE+"/final/"+f,metrics/f)
    pull(REMOTE+"/final/g9-comparisons.png",dest/"g9-comparisons.png")
    (metrics/"invariance").mkdir(exist_ok=True)
    pull(REMOTE+"/scores/invariance/",metrics/"invariance")
    pull(REMOTE+"/data/starss-invariance.json",metrics/"additional-rooms.json")
    pull("/mnt/slwm-checkpoints-a/SLWM/g9/dose/scores.json",metrics/"independent-dose.json")
    (metrics/"native-inference.json").write_bytes((ctl/"native-parity.json").read_bytes())
    drive=json.loads(ssh("cat /mnt/slwm-checkpoints-a/SLWM/g9/drive-state.json"))
    own=[{k:stage[k] for k in ("name","hardware","seconds","units","rate","active")} for stage in budget["project_endpoints"].values()]
    result=json.loads((metrics/"results.json").read_text());prior=33.37924
    if any(x["active"] for x in own) or sum(x["units"] for x in own)>100-prior or result["two_disk_checkpoints"]!=58:raise ValueError("Outstanding owned session, allowance or missing custody")
    release=dict(complete=True,primary_fits=38,additional_dose_fits=4,total_fits=42,primary_frozen_stages=54,
        supplemental_frozen_stages=4,two_disk_checkpoints=result["two_disk_checkpoints"],score_panels=counts,
        all_owned_released=True,owned_units=budget["project_spend"],prior_owned_units=prior,cumulative_owned_units=prior+budget["project_spend"],
        scope="Named G9 endpoints only; other account activity excluded",owned_endpoints=own,
        tpu_environment=json.loads((ctl/"tpu-environment.json").read_text()),cuda_environment=dict(hardware="RTX 3090",driver="615.78.08",jax="0.12.0.1",jaxlib="0.12.0.1",numpy="2.5.4",python="3.12.15"),
        drive_at_release={k:drive[k] for k in ("phase","reason","pending","verified_files") if k in drive},
        verified_portable_models=2,verified_adapter_files=8,media_durations_seconds={"cross_modal":40,"closed_loop":51.2})
    write(metrics/"release.json",release);writeup(repo)
    write(ctl/"assembled.json",{"ready":True,"two_disk_checkpoints":58,"owned_units":budget["project_spend"]})
    print(json.dumps(release,sort_keys=True),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("repo",type=Path);ap.add_argument("control",type=Path)
    a=ap.parse_args();run(a.repo.resolve(),a.control.resolve())
