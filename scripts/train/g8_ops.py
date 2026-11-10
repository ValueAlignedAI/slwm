"""Drain completed checkpoints to both HDDs over one official SSH connection per owned assignment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from g6_transfer import transfer
from g7_inputs import stream
from g7_ops import save


def supervise(a,name):
    ctl=a.control;wrapper=["bash","scripts/train/colab_session.sh"];receipts=ctl/"receipts"; receipts.mkdir(exist_ok=True)
    def note(phase,**values):save(ctl/(name+"-ops.json"),dict(phase=phase,utc_unix=time.time(),**values))
    installed=False;errors=0
    while True:
        budget=json.loads((ctl/"budget.json").read_bytes())
        if budget["phase"] in ("releasing","complete") or time.time()>=budget["deadline_unix"]-90:break
        try:
            result=subprocess.check_output([*wrapper,"exec",name,"-f","scripts/train/g8_inventory.py","--timeout","55"],text=True,stderr=subprocess.PIPE,timeout=90)
            line=next(s for s in result.splitlines() if s.startswith("SLWM_G8_STATUS "))
            status=json.loads(line.split(" ",1)[1]);save(ctl/(name+"-runtime.json"),status)
            for b in status["bridges"]:save(ctl/(f"bridge-{b['core']}-{b['seed']}.json"),b)
            with (ctl/(name+".transport")).open("a") as fd:
                try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:note("staging_inputs",lane=status["lane"]);time.sleep(12);continue
                if not installed:
                    stream(name,["git","archive",a.revision,"scripts/train/g6_artifacts.py"],"set -eu; cd /content/g8-code; tar -xf -");installed=True
                jobs=[v for v in status["stages"] if v["evaluated"] and not (receipts/f"{v['run']}-{v['step']}.json").is_file()]
                if jobs:
                    job=min(jobs,key=lambda x:(x["step"]!=16384,x["step"],x["run"]));note("archiving",run=job["run"],step=job["step"])
                    transfer(a.repo,name,job["run"],job["step"],job["sha"],receipts)
                else:
                    phase=status["lane"].get("phase","staging")
                    note(phase,stages=len(status["stages"]),seeds=status["registered_seeds"])
                    if phase=="draining" and all(v["evaluated"] for v in status["stages"]):
                        if set(status["registered_seeds"])!=set(a.expected[name]):raise RuntimeError("Assigned seed set not yet complete")
                        mandatory=sum(3*(2 if b["exact"] else 4) for b in status["bridges"])
                        if len(status["bridges"])!=2*len(a.expected[name]) or len(status["stages"])!=mandatory:raise RuntimeError("Missing registered generations")
                        break
            errors=0
            if not jobs:time.sleep(22)
        except (OSError,ValueError,RuntimeError,StopIteration,subprocess.SubprocessError) as exc:
            errors+=1;note("retry",count=errors,diagnostic=str(exc)[:230]);time.sleep(min(90,10*errors))
    # Release only the study-owned, explicitly assigned endpoint.
    if name in json.loads((ctl/"budget.json").read_bytes())["active"]:
        subprocess.run([*wrapper,"stop",name],timeout=120,check=True)
    note("released")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--repo",type=Path,required=True);p.add_argument("--control",type=Path,required=True);p.add_argument("--revision",required=True)
    a=p.parse_args();os.chdir(a.repo);a.expected=json.loads((a.control/"assignments.json").read_bytes())
    allowed=set(json.loads((a.control/"bounds.json").read_bytes())["sessions"])
    if set(a.expected)-allowed or sorted(s for seeds in a.expected.values() for s in seeds)!=[511,512]:raise ValueError("Unregistered G8 allocation plan")
    with ThreadPoolExecutor(max_workers=len(a.expected)) as pool:
        for f in [pool.submit(supervise,a,n) for n in a.expected]:f.result()
    (a.control/"requests-done").touch()
