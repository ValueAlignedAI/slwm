"""Publish completed project artifacts from Fedora to authorized private Drive, with verified objects and quota backoff."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from colab_artifacts import put_verified
from g7_lock_heads import sha


def write(p,obj):
    tmp=p.with_suffix(".partial");tmp.write_text(json.dumps(obj,sort_keys=True)+"\n");tmp.replace(p)


def publish(root,data,until):
    base="slwm-drive:SLWM/g7"
    remote=root/"drive-verified.json";sent=json.loads(remote.read_bytes()) if remote.exists() else {}
    retry=300
    while time.time()<until:
        if subprocess.check_output(["findmnt","-nro","TARGET","-T",str(root)],text=True).strip()!="/mnt/slwm-checkpoints-a":raise ValueError("Checkpoint disk unavailable")
        files=[]
        for family in ("test","dev","train"):
            marker=data/"packets"/(family+".verified.json")
            if marker.exists():
                d=json.loads(marker.read_bytes());files.append((data/"packets"/d["file"],"inputs/"+d["file"],d["sha256"]))
        index=root/"readouts-archive.json"
        if index.is_file():
            for name,d in json.loads(index.read_bytes())["files"].items():files.append((root/name,"readouts/"+name,d["sha256"]))
        stages=sorted(root.glob("g7-*/step-*/verified.json"),key=lambda f:-int(f.parent.name[5:]))
        for mark in stages:
            d=json.loads(mark.read_bytes())
            for k,h in d["sha256"].items():
                if Path(k).name!=k:raise ValueError("Invalid registered filename")
                files.append((mark.parent/k,str(mark.parent.relative_to(root)/k),h))
            files.append((mark,str(mark.relative_to(root)),sha(mark)))
        remaining=[(p,n,h) for p,n,h in files if sent.get(n)!=h]
        if not remaining:
            write(root/"drive-state.json",{"phase":"verified_through_inventory","verified_files":len(sent),"current_stages":len(stages)})
            if len(stages)==48 and index.exists() and len(json.loads(index.read_bytes())["finished_markers"])>=128:return
            time.sleep(90);continue
        try:
            subprocess.run(["rclone","lsf","--dirs-only","--max-depth","1","--retries","1","--low-level-retries","1",
                            "--contimeout","7s","--timeout","12s","slwm-drive:SLWM"],text=True,capture_output=True,check=True,timeout=35)
            path,name,h=remaining[0]
            put_verified(path,base+"/objects/"+h,h)
            sent[name]=h;write(remote,sent)
            put_verified(remote,base+"/inventories/"+sha(remote)+".json",sha(remote))
            write(root/"drive-state.json",{"phase":"publishing","verified_files":len(sent),"pending_files":len(remaining)-1})
            retry=300
        except (OSError,RuntimeError,subprocess.SubprocessError) as exc:
            text=str(getattr(exc,"stderr","")).lower()
            cause="drive_project_query_limit" if "quota" in text or "ratelimit" in text else type(exc).__name__
            write(root/"drive-state.json",{"phase":"backoff","reason":cause,"verified_files":len(sent),"pending_files":len(remaining),"retry_unix":time.time()+retry})
            time.sleep(retry);retry=min(1800,2*retry)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--data",type=Path,required=True)
    p.add_argument("--hours",type=float,default=24);a=p.parse_args()
    os.environ["PATH"]=str(Path.home()/".local/bin")+":"+os.environ.get("PATH","")
    os.environ.setdefault("RCLONE_CONFIG",str(Path.home()/".config/slwm-colab/rclone.conf"))
    publish(a.root,a.data,time.time()+3600*a.hours)
