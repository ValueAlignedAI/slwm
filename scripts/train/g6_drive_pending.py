"""Retry private Drive independently of paid compute; upload a manifest after verifying its blobs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from colab_artifacts import put_verified


def write(p,d):
    tmp=p.with_suffix(".new"); tmp.write_text(json.dumps(d,sort_keys=True,indent=2)+"\n"); tmp.replace(p)


def publish(root,until):
    remote="slwm-drive:SLWM/g6"
    state=root/"drive-state.json"; delay=300
    while time.time()<until:
        if subprocess.check_output(["findmnt","-nro","TARGET","-T",root],text=True).strip()!="/mnt/slwm-checkpoints-a":
            raise RuntimeError("Checkpoint disk is unavailable")
        records=sorted(root.glob("g6-*/step-*/verified.json"),key=lambda p:-int(p.parent.name[5:]))
        pending=[p for p in records if not (p.parent/"drive-receipt.json").exists()]
        if not pending:
            if len(records)>=28: return
            time.sleep(90); continue
        try:
            subprocess.run(["rclone","lsf","--dirs-only","--max-depth","1","--low-level-retries","1","--retries","1",
                            "--contimeout","7s","--timeout","12s","slwm-drive:SLWM"],
                           capture_output=True,text=True,timeout=35,check=True)
            source=pending[0]; folder=source.parent; receipt=json.loads(source.read_bytes())
            for k,expected in receipt["sha256"].items():
                if k not in {"inference.npz","resume.npz","config.json","progress.json","complete.json","dev-metrics.json","dev-predictions.npz"}:
                    raise ValueError("Unexpected artifact")
                put_verified(folder/k,remote+"/objects/"+expected,expected)
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            locator=f"{remote}/runs/{receipt['run']}/{receipt['step']:06d}-{digest}.json"
            put_verified(source,locator,digest)
            write(folder/"drive-receipt.json",{"manifest":locator,"sha256":digest,"verified":receipt["sha256"]})
            write(state,{"phase":"publishing","remaining_stages":len(pending)-1,"last":locator,"utc_unix":time.time()})
            delay=300
        except (OSError,ValueError,subprocess.SubprocessError) as exc:
            diagnostics=str(getattr(exc,"stderr","")).lower()
            cause="drive_project_query_limit" if ("quota" in diagnostics or "ratelimitexceeded" in diagnostics) else type(exc).__name__
            write(state,{"phase":"backoff","reason":cause,"pending_stages":len(pending),"retry_unix":time.time()+delay})
            print("Drive transfer deferred",cause,"seconds",delay,flush=True)
            time.sleep(delay); delay=min(1800,delay*2)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("root",type=Path); p.add_argument("--until",required=True,type=float)
    a=p.parse_args(); publish(a.root,a.until)
