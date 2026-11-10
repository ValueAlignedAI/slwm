"""Publish only complete same-chip comparisons backed by two physical checkpoint receipts."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

BASE=Path("experiments/multimodal/t6/tpu-scale-20261110")
BRANCH="training/g5-world-dynamics-v6e-20261110"
FEDORA=["ssh","-o","BatchMode=yes","-o","ConnectTimeout=8","db@192.168.1.200"]


def index(directory):
    entries=[json.loads(p.read_bytes()) for p in directory.glob("*.json")]
    rows=sorted((r for r in entries if r.get("comparison")=="architecture"),key=lambda r:(r["seed"],r["step"],r["arms"][0]["parameters"]))
    lines=["# Verified matched TPU checkpoints","",
           "Every row uses the same chip, data order, updates, and parameter budget for both cores. Both full optimizer states and inference weights have been verified on two Fedora disks. Lower action error and higher contact average precision are better.","",
           "| TPU | Seed | Parameters | Updates | Hybrid action error, px | Dense action error, px | Hybrid contact AP | Dense contact AP |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        h,d=r["arms"]
        name=h["run"].split("-")[2]; gen=h["tpu"]
        link=f"metrics/{gen}-{r['seed']}-{name}-{r['step']}.json"
        lines.append(f"| {r['device']} | {r['seed']} | {h['parameters']:,} | [{r['step']:,}]({link}) | {h['action']['joint']['endpoint_px']:.3f} | {d['action']['joint']['endpoint_px']:.3f} | {h['contact']['average_precision']['correct']:.5f} | {d['contact']['average_precision']['correct']:.5f} |")
    sizes=sorted((r for r in entries if r.get("comparison")=="size"),key=lambda r:(r["seed"],r["step"],r["arms"][0]["run"]))
    if sizes:
        lines+=["","## Parameter effect at the same exposure","",
                "Positive values favor the 202.35M model in both columns. The linked records include paired 95% intervals.","",
                "| TPU | Seed | Core | Updates | Action error reduction, px | Contact AP gain |",
                "| --- | ---: | --- | ---: | ---: | ---: |"]
        for r in sizes:
            arm=r["arms"][0]; core=arm["run"].split("-")[3]
            path=f"metrics/{arm['tpu']}-{r['seed']}-size-{core}-{r['step']}.json"
            diff=r["first_minus_second"]
            lines.append(f"| {r['device']} | {r['seed']} | {core} | [{r['step']:,}]({path}) | {diff['action_error_reduction_px']:.3f} | {diff['contact_ap_gain']:.5f} |")
    lines+=["","These are development measurements. An architecture claim requires the two-seed, two-endpoint gate. [Study protocol](protocol.md)"]
    return "\n".join(lines)+"\n"


def run(a):
    os.chdir(a.repo); directory=BASE/"metrics"; directory.mkdir(exist_ok=True,parents=True)
    while time.time()<a.deadline:
        try:
            receipts={}
            for p in (a.control/"receipts").glob("*.json"):
                r=json.loads(p.read_bytes())
                m=re.fullmatch(r"g6-(v5e|v6e)-(larger|anchor)-(hybrid|gpt2)-(511|512)",r["run"])
                if not m or len(set(r["disk_uuids"]))!=2:
                    raise ValueError("Unexpected local artifact receipt")
                gen,scale,core,seed=m.groups()
                receipts[(gen,seed,r["step"],scale,core)]=r
            cases=[]
            for gen,seed,step in sorted({key[:3] for key in receipts}):
                for scale in ("larger","anchor"):
                    keys=[(gen,seed,step,scale,c) for c in ("hybrid","gpt2")]
                    if all(k in receipts for k in keys):
                        cases.append((f"{gen}-{seed}-{scale}-{step}","architecture",step,[receipts[k] for k in keys]))
                for core in ("hybrid","gpt2"):
                    keys=[(gen,seed,step,s,core) for s in ("larger","anchor")]
                    if all(k in receipts for k in keys):
                        cases.append((f"{gen}-{seed}-size-{core}-{step}","size",step,[receipts[k] for k in keys]))
            files=[]
            for stem,kind,step,pair in cases:
                path=directory/(stem+".json")
                if path.exists(): continue
                args=["/home/db/dev/slwm-g4/.jax-cpu/bin/python","-m","training.g6_pair_report",
                      "--root","/mnt/slwm-checkpoints-a/SLWM/g6","--first",pair[0]["run"],"--second",pair[1]["run"],
                      "--kind",kind,"--step",str(step)]
                command="cd /home/db/dev/slwm-g5 && OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu "+shlex.join(args)
                out=json.loads(subprocess.check_output([*FEDORA,command],text=True,timeout=100))
                if [v["inference_sha256"] for v in out["arms"]]!=[v["sha256"]["inference.npz"] for v in pair]:
                    raise ValueError("Published rows differ from the two registered weights")
                tmp=path.with_suffix(".pending"); tmp.write_text(json.dumps(out,sort_keys=True,indent=2)+"\n"); tmp.replace(path)
                files.append(str(path))
            idx=BASE/"matched.md"; idx.write_text(index(directory))
            # A previous interrupted publication can have valid new evidence still unstaged.
            valid=[str(directory/(c[0]+".json")) for c in cases if (directory/(c[0]+".json")).exists()]+[str(idx)]
            if subprocess.check_output(["git","branch","--show-current"],text=True).strip()!=BRANCH:
                raise ValueError("Study worktree changed branch")
            if subprocess.check_output(["git","status","--porcelain","--",*valid],text=True).strip():
                subprocess.run(["git","add","--",*valid],check=True)
                subprocess.run(["git","commit","--only","-m","Publish verified same-exposure G6 TPU comparisons","--",*valid],check=True)
            env=dict(os.environ,GIT_SSH_COMMAND="ssh -o BatchMode=yes -o UserKnownHostsFile=/Users/db/dev/slwm-g2/artifacts/predictive_world/g2/github_known_hosts -o StrictHostKeyChecking=yes")
            if subprocess.check_output(["git","rev-list","--count","@{upstream}..HEAD"],text=True).strip()!="0":
                subprocess.run(["git","push","origin",BRANCH],env=env,check=True,timeout=100)
            if a.done.is_file():
                return
        except (OSError,ValueError,subprocess.SubprocessError) as e:
            print(type(e).__name__,str(e)[:180],flush=True)
        time.sleep(55)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--repo",required=True,type=Path)
    p.add_argument("--control",required=True,type=Path); p.add_argument("--deadline",required=True,type=float)
    p.add_argument("--done",required=True,type=Path); run(p.parse_args())
