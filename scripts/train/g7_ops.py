"""Monitor only owned G7 sessions, install verified fresh inputs, drain optimizer generations, and release."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from g6_transfer import transfer
from g7_inputs import FED,shell,stream


def save(p,doc):
    q=p.with_suffix(".updating");q.write_text(json.dumps(doc,sort_keys=True)+"\n");q.replace(p)


def main(a):
    ctl=a.ctl; wrapper=["bash","scripts/train/colab_session.sh"]
    def monitor(name,seed):
        status=ctl/(name+"-ops.json")
        def note(phase,**kw):save(status,dict(phase=phase,utc_unix=time.time(),**kw))
        lockfile=ctl/(name+".transport"); receipts=ctl/"receipts";receipts.mkdir(exist_ok=True)
        fresh=set(); failures=0
        while True:
            budget=json.loads((ctl/"budget.json").read_bytes())
            if budget["phase"] in ("releasing","complete") or time.time()>=budget["deadline_unix"]-120:break
            try:
                result=subprocess.check_output([*wrapper,"exec",name,"-f","scripts/train/g7_inventory.py","--timeout","55"],text=True,timeout=90,stderr=subprocess.PIPE)
                info=json.loads(next(s[15:] for s in result.splitlines() if s.startswith("SLWM_G7_STATUS ")))
                save(ctl/(name+"-runtime.json"),info)
                with lockfile.open("a") as fd:
                    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        note("preparing_or_transferring",lane=info["lane"]);time.sleep(12);continue
                    ready=ctl/(name+"-sender-installed")
                    if not ready.exists():
                        stream(name,["git","archive",a.revision,"scripts/train/g6_artifacts.py","scripts/train/g7_extract_fresh.py"],
                               "cd /content/g7-code && tar -xf -")
                        ready.touch()
                    for role in ("train","dev"):
                        if role in fresh:continue
                        record=subprocess.run([*FED,"cat",f"/mnt/slwm-data/SLWM/g7/packets/{role}.verified.json"],text=True,capture_output=True)
                        if record.returncode:continue
                        meta=json.loads(record.stdout);sha=meta["sha256"];file=f"g7-fresh-{role}.tar.zst"
                        if meta["file"]!=file or len(sha)!=64 or any(c not in "0123456789abcdef" for c in sha):raise ValueError("Invalid fresh packet metadata")
                        marker=".verified" if role=="train" else ".dev-verified"
                        ok=shell(name,f"test -f /content/g7-fresh/{marker} && cat /content/g7-fresh/{marker} || echo pending",text=True,capture_output=True).stdout.strip()
                        if ok!=sha:
                            note("installing_fresh",role=role,bytes=meta["bytes"])
                            extract=(f"set -eu; cd /content/g7-code; cat > /content/{file}; "
                                     "mkdir -p /dev/shm/g7-fresh; test -e /content/g7-fresh || ln -s /dev/shm/g7-fresh /content/g7-fresh; "
                                     f"python scripts/train/g7_extract_fresh.py /content/{file} {sha} /content/g7-fresh {role}")
                            stream(name,[*FED,"cat","/mnt/slwm-data/SLWM/g7/packets/"+file],extract)
                        fresh.add(role)
                    job=next((v for v in sorted(info["rungs"],key=lambda r:(r["step"],"fresh" not in r["run"],r["run"]))
                              if v["evaluated"] and not (receipts/(v["run"]+"-"+str(v["step"])+".json")).exists()),None)
                    if job:
                        note("archiving",run=job["run"],step=job["step"])
                        transfer(a.repo,name,job["run"],job["step"],job["sha"],receipts)
                    elif info["lane"].get("phase") in ("failed","draining"):
                        note("durable" if info["lane"]["phase"]=="draining" else "failed",lane=info["lane"],rungs=len(info["rungs"]));break
                    else:note("active",lane=info["lane"],rungs=len(info["rungs"]))
                failures=0
                if not job:time.sleep(30)
            except (OSError,ValueError,RuntimeError,StopIteration,subprocess.SubprocessError) as e:
                failures+=1;note("retry",count=failures,reason=str(e)[:190])
                if failures>=3:
                    live=subprocess.run([str(Path.home()/".local/bin/colab"),"--auth","adc","sessions"],env=dict(os.environ,CLOUDSDK_CONFIG=str(Path.home()/".config/slwm-colab/gcloud")),text=True,capture_output=True)
                    if live.returncode==0 and ("["+name+"] ") not in live.stdout:note("lost",reason="native assignment unavailable");break
                time.sleep(min(120,20*failures))
        if name in json.loads((ctl/"budget.json").read_bytes())["active"]:
            subprocess.run([*wrapper,"stop",name],timeout=100)
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(monitor,n,seed) for n,seed in ((a.first,511),(a.second,512))]
        for task in tasks:task.result()
    (ctl/"requests-done").touch()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--repo",type=Path,required=True);p.add_argument("--ctl",type=Path,required=True)
    p.add_argument("--revision",required=True);p.add_argument("--first",required=True);p.add_argument("--second",required=True)
    a=p.parse_args();os.chdir(a.repo);main(a)
