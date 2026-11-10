"""Stream between two authenticated SSH transports, keeping only a lock and receipt on the Mac."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess
import time


def transfer(repo,session,tag,step,digest,control):
    study=tag[:2]
    patterns={"g6":r"g6-(?:v5e|v6e)-(?:larger|anchor)-(?:hybrid|gpt2)-(?:511|512)",
              "g7":r"g7-(?:v5e|v6e)-(?:larger|anchor)-(?:hybrid|gpt2)-(?:511|512)-(?:replay|fresh)"}
    patterns["g8"]=r"g8-v5e-anchor-(?:hybrid|gpt2)-(?:511|512)-(?:sim|other|replay|fresh)"
    stages={"g8":(10240,12288,16384),"g6":(128,512,2048,8192),"g7":(10240,12288,16384)}
    if study not in patterns or not re.fullmatch(rf"slwm-{study}-[a-z0-9-]+",session) or not re.fullmatch(patterns[study],tag) or not re.fullmatch("[a-f0-9]{64}",digest):
        raise ValueError("Invalid registered artifact")
    if step not in stages[study]: raise ValueError("Unregistered milestone")
    control.mkdir(exist_ok=True,parents=True)
    meta=control/f"{tag}-{step}.json"
    bridge=["ssh","-o","BatchMode=yes","-o","ConnectTimeout=12","-o","ServerAliveInterval=30","db@192.168.1.200"]
    opts=f" --run {tag} --step {step} --sha {digest}"
    checkout={"g6":"slwm-g5","g7":"slwm-g7","g8":"slwm-g8"}[study]
    receive=f"cd /home/db/dev/{checkout} && python3 -m scripts.train.g6_artifacts "
    with meta.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        info=json.loads(subprocess.check_output([*bridge,receive+"offset"+opts],text=True,timeout=30))
        if not info["verified"]:
            offset=int(info["offset"])
            send=f"cd /content/{study}-code && python -m scripts.train.g6_artifacts send{opts} --offset {offset} --output /content/{study}-out"
            command=["bash","scripts/train/colab_session.sh","ssh",session,"bash","-c",send]
            with subprocess.Popen(command,cwd=repo,stdout=subprocess.PIPE) as sender:
                try:
                    result=subprocess.run([*bridge,receive+"receive"+opts],stdin=sender.stdout,timeout=2200)
                finally:
                    sender.stdout.close()
                try:
                    rc=sender.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    sender.terminate(); sender.wait(timeout=15); raise
                if rc or result.returncode:
                    raise RuntimeError(f"Transport interrupted; verified offset will be reused ({rc}, {result.returncode})")
        receipt=json.loads(subprocess.check_output([*bridge,"cat",f"/mnt/slwm-checkpoints-a/SLWM/{study}/{tag}/step-{step:06d}/verified.json"],text=True,timeout=30))
        if receipt["sha256"]["resume.npz"]!=digest or len(set(receipt["disk_uuids"]))!=2:
            raise ValueError("Incomplete replica receipt")
        temp=meta.with_suffix(".tmp"); temp.write_text(json.dumps(receipt,sort_keys=True)+"\n"); temp.replace(meta)
        print(json.dumps({"archived":tag,"step":step,"optimizer_bytes":receipt["optimizer_bytes"]}),flush=True)
        return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,required=True); p.add_argument("--session",required=True)
    p.add_argument("--run",required=True); p.add_argument("--step",required=True,type=int)
    p.add_argument("--sha",required=True); p.add_argument("--control",type=Path,required=True)
    a=p.parse_args(); transfer(a.repo,a.session,a.run,a.step,a.sha,a.control)
