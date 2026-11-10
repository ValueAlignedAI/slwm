"""Mirror local G9 terminals and independently queue verified data and weights for private Drive."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from g9_storage import disks,mirror,sha,write


def local():
    a,_=disks()
    while True:
        items=sorted(a.glob("pilot-*/step-00384/receipt.json"))
        for p in items:
            meta=json.loads(p.read_text())
            if not (p.parent/"verified.json").exists():mirror(meta["tag"],384)
        if len(items)==22:return
        time.sleep(10)


def drive():
    from colab_artifacts import put_verified
    a,_=disks();root=a.parent;data=Path("/mnt/slwm-data/SLWM/g9/data")
    os.environ["PATH"]=str(Path.home()/".local/bin")+":"+os.environ["PATH"]
    os.environ.setdefault("RCLONE_CONFIG",str(Path.home()/".config/slwm-colab/rclone.conf"))
    dest="slwm-drive:SLWM/g9";inventory=root/"drive-verified.json";sent=json.loads(inventory.read_text()) if inventory.exists() else {}
    end=time.time()+26*3600;delay=300
    while time.time()<end:
        files=[]
        for name in ("starss-train.npz","starss-dev.npz","starss-test.npz","starss-invariance.npz","contact/roma-cal.npz","contact/roma-held.npz","contact/force-cal-uncapped.npz","contact/force-held-uncapped.npz"):
            p=data/name
            if p.is_file() and (p.stat().st_size>100000 or "uncapped" in name):files.append((p,"data/"+name,None))
        aux=root/"auxiliary/verified.json"
        if aux.exists():
            for name,key in json.loads(aux.read_text()).items():
                src=aux.parent/name;files.append((src,str(src.relative_to(root)),key))
        portable=root/"portable/portable.json"
        if portable.exists():
            for tag,meta in json.loads(portable.read_text()).items():
                src=portable.parent/(tag+".npz");files.append((src,str(src.relative_to(root)),meta["sha256"]))
        for p in sorted([*a.glob("*/step-*/verified.json"),*(root/"dose").glob("*/step-*/verified.json")],key=lambda f:("confirm" not in f.as_posix(),"04096" not in f.as_posix())):
            d=json.loads(p.read_text())
            for n,k in (("weights.npz","inference_sha256"),("optimizer.npz","optimizer_sha256")):
                path=p.parent/n;files.append((path,str(path.relative_to(root)),d[k]))
        pending=[v for v in files if v[1] not in sent]
        if not pending:
            write(root/"drive-state.json",dict(phase="verified_current_inventory",verified_files=len(sent),pending=0))
            if (root/"finalized").exists():return
            time.sleep(45);continue
        defer=0
        for old in ("g7","g8"):
            f=root.parent/old/"drive-state.json"
            if f.is_file():
                st=json.loads(f.read_text())
                if "query_limit" in st.get("reason",""):defer=max(defer,st.get("retry_unix",0))
        if defer>time.time():
            write(root/"drive-state.json",dict(phase="backoff",reason="shared_drive_project_query_limit",retry_unix=defer,verified_files=len(sent),pending=len(pending)))
            time.sleep(min(120,max(1,defer-time.time())));continue
        try:
            subprocess.run(["rclone","lsf","--dirs-only","--max-depth","1","--retries","1","--low-level-retries","1","--timeout","15s","slwm-drive:SLWM"],check=True,capture_output=True,text=True,timeout=35)
            path,name,key=pending[0];key=key or sha(path)
            put_verified(path,dest+"/objects/"+key,key);sent[name]=key;write(inventory,sent)
            pin=sha(inventory);put_verified(inventory,dest+"/inventories/"+pin+".json",pin)
            write(root/"drive-state.json",dict(phase="publishing",verified_files=len(sent),pending=len(pending)-1));delay=300
        except (OSError,RuntimeError,subprocess.SubprocessError) as exc:
            msg=str(getattr(exc,"stderr","")).lower();reason="drive_project_query_limit" if "quota" in msg or "ratelimit" in msg else type(exc).__name__
            write(root/"drive-state.json",dict(phase="backoff",reason=reason,verified_files=len(sent),pending=len(pending),retry_unix=time.time()+delay))
            time.sleep(delay);delay=min(1800,2*delay)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("role",choices=["local","drive"]);a=p.parse_args()
    local() if a.role=="local" else drive()
