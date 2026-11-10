"""Bounded native Colab dispatcher for two paid G5 lanes. No chat notifications or recurring schedule."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from g5_colab_budget import named, usage


def main(a):
    credentials=Path.home()/".config/slwm-colab"
    cli=[str(Path.home()/".local/bin/colab"),"--auth","adc"]
    env=dict(os.environ,CLOUDSDK_CONFIG=str(credentials/"gcloud"),COLAB_CLI_REQUEST_TIMEOUT="420",
             COLAB_CLI_ASSIGN_RETRIES="0")
    wrapper=["bash",str(a.repo/"scripts/train/colab_session.sh")]
    roots=a.state.parent
    names={"hybrid":"slwm-g5-v6e-signal-1110","gpt2":"slwm-g5-v6e-dense-1110"}
    def call(cmd,timeout=80,**kwargs):
        return subprocess.run(cmd,cwd=a.repo,env=env,timeout=timeout,check=True,**kwargs)
    def report(core,phase,**extra):
        path=roots/(core+".json")
        part=path.with_suffix(".tmp")
        part.write_text(json.dumps(dict(core=core,phase=phase,utc_unix=time.time(),**extra),sort_keys=True)+"\n")
        part.replace(path)
        print(core,phase,extra,flush=True)
    def bounds():
        budget=json.loads(a.state.read_bytes())
        if time.time()-budget["checked_unix"]>130 or budget["phase"]!="watching" or (roots/"release").exists():
            return False
        return time.time()+3600 < budget["deadline_unix"] and budget["account_spend"]+budget["reserve"]<budget["credit_limit"]
    def live():
        return named(call([*cli,"sessions"],text=True,capture_output=True).stdout,names.values())
    def ready():
        for role in ("train","dev"):
            p=a.pins/(role+".sha256")
            if not p.exists():
                return False
            if len(p.read_text().strip().splitlines())!=3:
                return False
        return (a.pins/"prior.npz").is_file() and (a.pins/"prior.sha256").is_file()
    def run(core,name):
        path=roots/(core+".json")
        old=json.loads(path.read_bytes()) if path.is_file() else {}
        if old.get("phase") in ("complete","capacity_unavailable","refused","stopped","failed"):
            return
        attempts=int(old.get("attempts",0))
        while bounds() and not ready():
            report(core,"waiting_for_verified_data",attempts=attempts)
            time.sleep(60)
        scale=a.scale
        if not bounds():
            report(core,"stopped",reason="budget, heartbeat or archive reserve")
            return
        endpoint=None
        try:
            available=live()
            endpoint=available.get(name)
            while not endpoint and attempts<3 and bounds():
                if attempts:
                    # Supply errors back off across a single account; do not switch users or regions.
                    until=time.time()+[0,1800,3600][attempts]
                    while time.time()<until and bounds():
                        time.sleep(min(45,max(.1,until-time.time())))
                if not bounds():
                    break
                endpoint=live().get(name)
                if endpoint:
                    break
                acct=usage(call([*cli,"usage"],text=True,capture_output=True).stdout)
                if acct[0]<json.loads(a.state.read_bytes())["starting_balance"]-96:
                    break
                attempts+=1
                report(core,"requesting",attempts=attempts)
                try:
                    call([*wrapper,"new",name,"v6e1"],timeout=530,capture_output=True,text=True)
                except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
                    # Reconcile ambiguous results before a second POST.
                    output=str(getattr(exc,"stdout",""))+str(getattr(exc,"stderr",""))
                    if "precondition failed" in output.lower() or "too many assignments" in output.lower():
                        endpoint=live().get(name)
                        if not endpoint:
                            report(core,"refused",attempts=attempts)
                            return
                endpoint=live().get(name)
            if not endpoint:
                report(core,"capacity_unavailable",attempts=attempts)
                return
            report(core,"preparing",attempts=attempts,endpoint=endpoint)
            call([*wrapper,"prepare",name],timeout=300,stdout=subprocess.DEVNULL)
            call([*wrapper,"exec",name,"-f","scripts/train/g4_colab_topology.py","--timeout","90"],timeout=150)
            with tempfile.TemporaryDirectory(dir=roots) as temp:
                tmp=Path(temp); tree=tmp/"code.tar"
                selected=["models/g4_shared.py","models/g5_shared.py","models/g5_residual.py","training/g4_data.py",
                          "training/g4_train.py","training/g5_fit.py","training/g5_batches.py","training/g5_priors.py",
                          "training/g5_residual_fit.py","training/g5_eval.py","configs/t5/joint_world.json",
                          "scripts/train/g5_cloud_data.py","scripts/train/g5_cloud_lane.sh",
                          "scripts/train/colab_artifacts.py","scripts/train/g5_promote.py"]
                with tree.open("wb") as fd:
                    call(["git","archive",a.revision,*selected],stdout=fd)
                call(["tar","-rf",str(tree),"-C",str(a.pins.parent),a.pins.name])
                with tree.open("rb") as fd:
                    call([*wrapper,"ssh",name,"bash","-c","mkdir -p /content/g5-code && tar -xf - -C /content/g5-code"],timeout=160,stdin=fd)
            deadline=int(json.loads(a.state.read_bytes())["deadline_unix"])
            job=f"cd /content/g5-code && bash scripts/train/g5_cloud_lane.sh {core} {deadline} {scale}; echo $? > /content/g5-exit"
            call([*wrapper,"ssh",name,"bash","-c",'nohup bash -c "$1" </dev/null >/content/g5-lane.log 2>&1 &',"launch",job],timeout=100)
            report(core,"training",endpoint=endpoint,attempts=attempts)
            fails=0
            while time.time()<deadline-180 and not (roots/"release").exists():
                time.sleep(60)
                try:
                    result=call([*wrapper,"ssh",name,"bash","-c",
                        "if test -f /content/g5-exit; then cat /content/g5-exit; else echo active; fi"],timeout=90,text=True,capture_output=True)
                    state=result.stdout.strip().splitlines()[-1]
                    fails=0
                    if state!="active":
                        report(core,"complete" if state=="0" else "failed",exit=state,attempts=attempts)
                        return
                except (subprocess.SubprocessError,IndexError,OSError):
                    fails+=1
                    if fails>=3:
                        report(core,"failed",reason="lost runtime after three status checks",attempts=attempts)
                        return
            report(core,"stopped",reason="deadline or explicit release",attempts=attempts)
        except (OSError,ValueError,IndexError,subprocess.SubprocessError) as exc:
            report(core,"failed",reason=type(exc).__name__,attempts=attempts)
        finally:
            if endpoint:
                try:
                    call([*wrapper,"stop",name],timeout=80)
                except subprocess.SubprocessError:
                    pass
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda pair:run(*pair),names.items()))
    (roots/"requests-done").touch()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",required=True,type=Path); p.add_argument("--state",required=True,type=Path)
    p.add_argument("--pins",required=True,type=Path); p.add_argument("--revision",required=True)
    p.add_argument("--scale",choices=["medium","anchor"],default="medium")
    main(p.parse_args())
