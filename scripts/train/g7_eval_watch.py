"""Finish fresh held-set evaluation for both root sizes, both cores, seeds, and interventions."""
import json
from pathlib import Path
import subprocess
import sys
import time
from scripts.train.g6_artifacts import roots
from scripts.train.g7_scale_lock import cases,freeze


def main():
    root=Path("/mnt/slwm-data/SLWM/g7"); old=root.parent/"g5"
    a,b=roots("g7"); parents=roots("g6")
    results=a/"scale-eval";results.mkdir(exist_ok=True)
    until=time.time()+11*3600
    jobs=[]
    for name,terminal in cases():
        for step in ([8192] if terminal==8192 else [10240,12288,16384]):
            jobs.append((name,step,"dev"))
    jobs+= [(name,step,"test") for name,step in cases()]
    while time.time()<until:
        terminal_lock=a/"scale-test-lock.json"
        if not terminal_lock.exists():
            if all(all((disk/tag/f"step-{step:06d}"/"verified.json").is_file() for disk in ((a,b) if step==16384 else parents)) for tag,step in cases()):
                freeze(root,old)
        remaining=0;chosen=None
        for tag,step,role in jobs:
            stage=f"step-{step:06d}";out=results/tag/stage
            if (out/(role+"-signals.json")).is_file() and (out/(role+"-metrics.json")).is_file():continue
            remaining+=1
            disks=(a,b) if tag.startswith("g7-") else parents
            if (role=="test" and not terminal_lock.is_file()) or not all((d/tag/stage/"verified.json").is_file() for d in disks):continue
            # Newly sealed test comparisons take priority over unsealed intermediate dev panels.
            priority=(role!="test",step not in (8192,16384),step)
            if chosen is None or priority<chosen[0]:
                chosen=(priority,disks[0]/tag/stage,role)
        if not remaining:
            print("Complete",len(jobs),"fresh evaluations",flush=True);return
        if chosen is None:
            time.sleep(25);continue
        _,stage,role=chosen
        print("Evaluating",stage,role,remaining,"remaining",flush=True)
        cmd=["flock","/tmp/slwm_research_gpu.lock","bash","scripts/train/g4_cuda.sh","-m","training.g7_fresh_eval",
             "--root",str(root),"--old",str(old),"--run",str(stage),"--out",str(results),"--role",role]
        if role=="test":cmd+=["--lock",str(terminal_lock)]
        subprocess.run(cmd,check=True)
    raise TimeoutError("Evaluation deadline")


if __name__=="__main__":main()
