"""One-shot Mac recovery and Git publication worker for the locked G5 terminal comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

HOST="db@192.168.1.200"
REMOTE="/mnt/slwm-checkpoints-a/SLWM/g5"
BASE=Path("experiments/multimodal/t5/interventional-world-20261110")
BRANCH="training/g5-world-dynamics-v6e-20261110"


def main(a):
    os.chdir(a.repo)
    deadline=time.time()+a.hours*3600
    ssh=["ssh","-o","BatchMode=yes","-o","ConnectTimeout=6",HOST]
    def remote(path):
        buff=subprocess.check_output(ssh+["cat",path],timeout=35)
        if len(buff)>2_000_000:
            raise ValueError("Refusing an unexpectedly large evidence file")
        return json.loads(buff)
    def save(path,obj):
        path.parent.mkdir(exist_ok=True,parents=True)
        temp=path.with_suffix(".tmp"); temp.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n"); temp.replace(path)
    failures=0
    while time.time()<deadline:
        try:
            with Path("scripts/train/g5_reconcile_fedora.sh").open("rb") as fd:
                check=subprocess.run([*ssh,"bash","-s"],stdin=fd,capture_output=True,timeout=40)
            if check.returncode:
                raise RuntimeError("volumes unavailable" if check.returncode==10 else "Fedora not reachable")
            status=json.loads(check.stdout.decode().splitlines()[-1])
            failures=0
            save(a.state,dict(utc_unix=time.time(),**status))
            if not status["terminal"]:
                time.sleep(100)
                continue
            lock=remote(REMOTE+"/terminal-held-lock.json"); result=remote(REMOTE+"/terminal-test.json")
            if len(lock["sha256"])!=4 or result["sha256"]!=lock["sha256"] or result["split"]!="test":
                raise ValueError("Terminal metrics or four-endpoint membership differ from lock")
            receipts={}
            for tag,expected in lock["sha256"].items():
                if not tag.startswith("g5-"+lock["scale"]+"-") or not tag.endswith("-residual-gpu"):
                    raise ValueError("Unregistered terminal name")
                rec=remote(REMOTE+"/"+tag+"/disk-receipt.json")
                if not rec["independent_filesystems"] or rec["sha256"]["inference.npz"]!=expected:
                    raise ValueError("Terminal inference file lacks two matching physical copies")
                receipts[tag]=rec["sha256"]
            files=[]
            for name,content in (("terminal-test",result),("terminal-lock",lock),("terminal-storage",receipts),
                                 ("terminal-development",remote(REMOTE+"/"+lock["scale"]+"-residual-dev.json"))):
                path=BASE/"metrics"/(name+".json"); save(path,content); files.append(str(path))
            report=BASE/"terminal.md"
            text=[f"# Terminal {lock['scale']} comparison","",f"All four inference weights and optimizer states were verified on two independent Fedora checkpoint volumes before opening the held targets.",
                  "", "| Seed | Hybrid endpoint error, px | Dense endpoint error, px | Raw fitted, px | Hybrid contact AP | Dense contact AP | Raw motion AP |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for seed,row in result["seeds"].items():
                h=row["controls"]["hybrid"]; d=row["controls"]["gpt2"]
                text.append(f"| {seed} | {h['push']['joint']['endpoint_px']:.2f} | {d['push']['joint']['endpoint_px']:.2f} | {h['push']['ridge']['endpoint_px']:.2f} | {h['impact']['correct']:.4f} | {d['impact']['correct']:.4f} | {h['impact']['raw_motion']:.4f} |")
            text+=["",f"The two-seed, two-endpoint architecture gate {'passed' if result['joint_architecture_win'] else 'was unmet'}. The hybrid's separate comparison against both fitted raw controls {'passed' if result['beats_fitted_raw_and_motion'] else 'was unmet'}.",
                   "", "The held simulator measures the consequences of issued actions. The real recordings measure visual evidence of contact timing. [Paired intervals, input ablations and controls](metrics/terminal-test.json) · [Frozen endpoints](metrics/terminal-lock.json) · [Storage verification](metrics/terminal-storage.json)"]
            report.write_text("\n".join(text)+"\n"); files.append(str(report))
            if subprocess.check_output(["git","branch","--show-current"],text=True).strip()!=BRANCH:
                raise ValueError("Research worktree changed branch")
            subprocess.run(["git","add","--",*files],check=True)
            changed=subprocess.run(["git","diff","--cached","--quiet","--",*files]).returncode
            if changed:
                subprocess.run(["git","commit","--only","-m","Publish sealed G5 matched terminal world-state evidence","--",*files],check=True)
            env=dict(os.environ,GIT_SSH_COMMAND="ssh -o BatchMode=yes -o UserKnownHostsFile=/Users/db/dev/slwm-g2/artifacts/predictive_world/g2/github_known_hosts -o StrictHostKeyChecking=yes")
            subprocess.run(["git","push","origin",BRANCH],check=True,env=env,timeout=90)
            save(a.state,{"phase":"published","sha":subprocess.check_output(["git","log","-1","--format=%H","--",str(report)],text=True).strip()})
            return
        except (OSError,RuntimeError,ValueError,subprocess.SubprocessError) as e:
            failures+=1
            save(a.state,{"phase":"retry","reason":str(e)[:155],"consecutive":failures,"utc_unix":time.time()})
            time.sleep(min(180,25*failures))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--repo",required=True,type=Path)
    p.add_argument("--state",required=True,type=Path); p.add_argument("--hours",type=float,default=10)
    main(p.parse_args())
