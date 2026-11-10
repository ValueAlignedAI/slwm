"""Publish immutable G5 media and completed models to the existing private Drive, with bounded backoff."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from colab_artifacts import put_verified
from g5_promote import promote


def publish(root,mirror,deadline):
    pending={}
    media=("video-to-audio.mp4","audio-to-video.mp4","text-to-audio-video.mp4","live.json")
    while time.time()<deadline:
        todo=[]
        for folder in sorted(root.glob("g5-*/live")):
            if all((folder/k).is_file() for k in media) and not (folder/"drive-receipt.json").is_file():
                todo.append(("media",folder))
        for run in sorted(root.glob("g5-*-residual-gpu")):
            if (run/"complete.json").is_file() and not (run/"verified.json").is_file():
                todo.append(("model",run))
        worked=False
        for kind,p in todo:
            identity=kind+"-"+p.parent.name+"-"+p.name
            fails,next_check=pending.get(identity,(0,0))
            if time.time()<next_check:
                continue
            try:
                if kind=="model":
                    promote(p,mirror)
                else:
                    dest=f"slwm-drive:SLWM/g5/media/{p.parent.name}"
                    checks={}
                    for k in media:
                        with (p/k).open("rb") as f:
                            checks[k]=hashlib.file_digest(f,"sha256").hexdigest()
                        put_verified(p/k,dest+"/"+k,checks[k])
                    listing=json.loads(subprocess.check_output(["rclone","lsjson",dest,"--files-only"],text=True))
                    ids={r["Name"]:r.get("ID") for r in listing if r["Name"] in media}
                    if any(not ids.get(k) for k in media):
                        raise ValueError("Drive file IDs unavailable")
                    receipt={"private":True,"run":p.parent.name,"sha256":checks,"file_ids":ids}
                    tmp=p/"receipt.tmp"; tmp.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
                    tmp.replace(p/"drive-receipt.json")
                    print("Verified private media",p.parent.name,flush=True)
                worked=True
            except (OSError,ValueError,subprocess.SubprocessError) as err:
                fails+=1
                delay=min(1800,60*2**min(fails,5))
                pending[identity]=(fails,time.time()+delay)
                print("Transfer deferred",kind,p.name,type(err).__name__,"seconds",delay,flush=True)
                # Respect a shared account or client rate limit before spending more queries.
                time.sleep(min(delay,120))
        if (root/"terminal-test.json").is_file() and not todo:
            return
        if not worked:
            time.sleep(45)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--root",required=True,type=Path)
    p.add_argument("--mirror",required=True,type=Path); p.add_argument("--hours",type=float,default=10)
    a=p.parse_args(); publish(a.root,a.mirror,time.time()+3600*a.hours)
