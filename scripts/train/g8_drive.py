"""Transfer bounded, verified G8 content to private Drive independently of training and respect quota backoff."""
import json
import os
from pathlib import Path
import subprocess
import time
from colab_artifacts import put_verified
from g7_lock_heads import sha

def write(file,d):
    tmp=file.with_suffix(".partial");tmp.write_text(json.dumps(d,sort_keys=True)+"\n");tmp.replace(file)


def main():
    root=Path("/mnt/slwm-checkpoints-a/SLWM/g8");remote="slwm-drive:SLWM/g8"
    os.environ["PATH"]=str(Path.home()/".local/bin")+":"+os.environ["PATH"]
    os.environ.setdefault("RCLONE_CONFIG",str(Path.home()/".config/slwm-colab/rclone.conf"))
    record=root/"drive-verified.json";sent=json.loads(record.read_bytes()) if record.exists() else {};delay=300;end=time.time()+26*3600
    while time.time()<end:
        if subprocess.check_output(["findmnt","-nro","TARGET","-T",str(root)],text=True).strip()!="/mnt/slwm-checkpoints-a":raise ValueError("Registered checkpoint volume is unavailable")
        objects=[];aux=root/"readouts-archive.json"
        if aux.exists():
            for name,entry in json.loads(aux.read_bytes())["files"].items():objects.append((root/name,name,entry["sha256"]))
        for p in sorted(root.glob("g8-*/step-*/verified.json"),key=lambda f:-int(f.parent.name[5:])):
            d=json.loads(p.read_bytes())
            for k,h in d["sha256"].items():
                if k!=Path(k).name:raise ValueError("Unsafe manifest path")
                f=p.parent/k;objects.append((f,str(f.relative_to(root)),h))
            objects.append((p,str(p.relative_to(root)),sha(p)))
        awaiting=[v for v in objects if sent.get(v[1])!=v[2]]
        if not awaiting:
            write(root/"drive-state.json",{"phase":"verified_current_inventory","verified_files":len(sent),"pending_files":0})
            if (root/"public/release-ready.json").exists():return
            time.sleep(60);continue
        neighbor=root.parent/"g7/drive-state.json"
        if neighbor.is_file():
            previous=json.loads(neighbor.read_bytes());resume=previous.get("retry_unix",0)
            if previous.get("reason")=="drive_project_query_limit" and resume>time.time():
                write(root/"drive-state.json",{"phase":"backoff","reason":"shared_drive_project_query_limit","verified_files":len(sent),"pending_files":len(awaiting),"retry_unix":resume})
                time.sleep(min(300,max(1,resume-time.time())));continue
        try:
            subprocess.run(["rclone","lsf","--dirs-only","--max-depth","1","--retries","1","--low-level-retries","1","--timeout","15s","slwm-drive:SLWM"],capture_output=True,text=True,timeout=35,check=True)
            f,name,h=awaiting[0];put_verified(f,remote+"/objects/"+h,h);sent[name]=h;write(record,sent)
            pin=sha(record);put_verified(record,remote+"/inventories/"+pin+".json",pin)
            write(root/"drive-state.json",{"phase":"publishing","verified_files":len(sent),"pending_files":len(awaiting)-1});delay=300
        except (OSError,RuntimeError,subprocess.SubprocessError) as exc:
            text=str(getattr(exc,"stderr","")).lower();cause="drive_project_query_limit" if "quota" in text or "ratelimit" in text else type(exc).__name__
            write(root/"drive-state.json",{"phase":"backoff","reason":cause,"verified_files":len(sent),"pending_files":len(awaiting),"retry_unix":time.time()+delay})
            time.sleep(delay);delay=min(1800,2*delay)


if __name__=="__main__":main()
