"""Two native Colab lanes sharing one account budget, independent same-chip pairs and physical archives."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from g6_transfer import transfer

FEDORA=["ssh","-o","BatchMode=yes","-o","ConnectTimeout=10","db@192.168.1.200"]
PRIMARY="slwm-g6-v5e-1110"
SECONDARY=("slwm-g6-v6e-1110","slwm-g6-v5e-b-1110")
FILES=[f"{f}-{s}.tar.zst" for s in ("dev","train") for f in ("paired","impact","pusht")]
CODE=["models/g4_shared.py","models/g5_shared.py","models/g5_residual.py","training/g4_data.py","training/g4_train.py",
      "training/g5_batches.py","training/g5_priors.py","training/g5_eval.py","training/g6_train.py","training/g6_checkpoint.py",
      "configs/t5/joint_world.json","configs/t6/tpu_scale.json","scripts/train/g5_cloud_data.py",
      "scripts/train/g6_artifacts.py","scripts/train/g6_resume.py","scripts/train/g6_lane.py","scripts/train/g6_ram_inputs.py"]


def write(path,value):
    tmp=path.with_suffix(".new"); tmp.write_text(json.dumps(value,sort_keys=True)+"\n"); tmp.replace(path)


def operations(a):
    credentials=Path.home()/".config/slwm-colab"
    a.control.mkdir(exist_ok=True,parents=True)
    cli=[str(Path.home()/".local/bin/colab"),"--auth","adc"]
    env=dict(os.environ,CLOUDSDK_CONFIG=str(credentials/"gcloud"),COLAB_CLI_ASSIGN_RETRIES="0",COLAB_CLI_REQUEST_TIMEOUT="420")
    wrapper=["bash","scripts/train/colab_session.sh"]
    def cmd(args,timeout=120,**kwargs):
        return subprocess.run(args,env=env,cwd=a.repo,timeout=timeout,check=True,**kwargs)
    def bound(margin=0):
        data=json.loads(a.budget.read_bytes())
        return (data["phase"]=="watching" and time.time()-data["checked_unix"]<140 and
                data["account_spend"]+data["reserve"]<data["credit_limit"] and
                time.time()+margin<data["deadline_unix"] and not (a.budget.parent/"release").exists())
    def active_allowed():
        b=json.loads(a.budget.read_bytes())
        return (b["phase"] in ("watching","accounting_retry") and time.time()-b["checked_unix"]<150
                and time.time()<b["deadline_unix"]-120 and b["account_spend"]+b["reserve"]<b["credit_limit"]
                and not (a.budget.parent/"release").exists())
    def report(key,phase,**values):
        write(a.control/(key+".json"),dict(phase=phase,utc_unix=time.time(),**values))
        print(key,phase,values,flush=True)
    def live():
        return cmd([*cli,"sessions"],capture_output=True,text=True,timeout=32).stdout
    def assigned(text,name):
        return any(line.startswith("["+name+"] ") for line in text.splitlines())
    def inventory(name):
        r=cmd([*wrapper,"exec",name,"-f","scripts/train/g6_inventory.py","--timeout","50"],capture_output=True,text=True,timeout=90)
        return json.loads(next(line[15:] for line in r.stdout.splitlines() if line.startswith("SLWM_G6_STATUS ")))
    def shell(name,script,timeout=120,**kwargs):
        return cmd([*wrapper,"ssh",name,"bash","-c",script],timeout=timeout,**kwargs)
    def launch(name,generation,seed):
        deadline=int(json.loads(a.budget.read_bytes())["deadline_unix"])
        script=("set -eu; cd /content/g6-code; test -f /content/g6-data-verified; mkdir -p /content/g6-out; "
            "source /content/g4-topology.env; export JAX_PLATFORMS=tpu JAX_COMPILATION_CACHE_DIR=/content/g6-xla "
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2; "
            f"nohup flock -n /content/g6-lane.lock python -u scripts/train/g6_lane.py --generation {generation} --seed {seed} "
            f"--deadline {deadline} --data /content/g6-data --output /content/g6-out </dev/null >>/content/g6-lane.log 2>&1 &")
        shell(name,script)
    def prepare(name,generation):
        cmd([*wrapper,"exec",name,"-f","scripts/train/g4_colab_topology.py","--timeout","100"],timeout=155)
        cmd([*wrapper,"prepare",name],timeout=300)
        command=["git","archive",a.revision,*CODE]
        with subprocess.Popen(command,cwd=a.repo,stdout=subprocess.PIPE) as archive:
            try: shell(name,"mkdir -p /content/g6-code && tar -xf - -C /content/g6-code",stdin=archive.stdout)
            finally: archive.stdout.close()
            if archive.wait(timeout=15): raise OSError("Code stream failed")
        # Use only the six pre-registered train/dev archives plus the 3.5 KB calibration.
        sender=[*FEDORA,"tar","-cf","-","-C","/mnt/slwm-data/SLWM/g5/local-packs","dev.sha256","train.sha256","prior.sha256",
                *FILES,"-C","/mnt/slwm-data/SLWM/g5","prior.npz"]
        with subprocess.Popen(sender,stdout=subprocess.PIPE) as stream:
            try:
                shell(name,"set -eu; mkdir -p /content/g5-landing; cd /content/g5-landing; tar -xf -; sha256sum -c dev.sha256 train.sha256 prior.sha256",
                      timeout=1600,stdin=stream.stdout)
            finally: stream.stdout.close()
            if stream.wait(timeout=20): raise OSError("Data stream failed")
        shell(name,"set -eu; cd /content/g6-code; python -c 'import optax,scipy,zstandard' || python -m pip -q install optax scipy zstandard; "
              "mkdir -p /content/g6-data; python scripts/train/g5_cloud_data.py dev --root /content/g6-data --pins /content/g5-landing & devpid=$!; "
              "python scripts/train/g5_cloud_data.py train --root /content/g6-data --pins /content/g5-landing; wait \"$devpid\"; "
              "cp /content/g5-landing/prior.npz /content/g6-data/prior.npz; python -u scripts/train/g6_ram_inputs.py /content/g6-data; "
              "touch /content/g6-data-verified",timeout=700)
        if list((a.control/"receipts").glob(f"g6-{generation}-*-512-*.json")):
            restore=f" -m scripts.train.g6_resume {{action}} --generation {generation} --seed 512"
            sender=[*FEDORA,"cd /home/db/dev/slwm-g5 && python3"+restore.format(action="send")]
            with subprocess.Popen(sender,stdout=subprocess.PIPE) as stream:
                try: shell(name,"cd /content/g6-code && python"+restore.format(action="receive"),timeout=2500,stdin=stream.stdout)
                finally: stream.stdout.close()
                if stream.wait(timeout=30): raise OSError("Cross-session recovery transport failed")
        launch(name,generation,512)
    def watch(name,generation,key,seed):
        failures=0; second_reused=False; receipt_dir=a.control/"receipts"
        while active_allowed():
            job=None
            try:
                info=inventory(name)
                write(a.control/(name+"-runtime.json"),info)
                phase=info["lane"].get("phase","starting")
                report(key,phase,session=name,generation=generation,seed=info["lane"].get("seed",seed),
                       target=info["lane"].get("target"),scale=info["lane"].get("scale"))
                # Finish each same-exposure pair in order. Transfer uses a disk-side offset after interruption.
                ranks={("larger",128):0,("larger",512):1,("anchor",512):2,("larger",2048):3,("anchor",2048):4,
                       ("larger",8192):5,("anchor",8192):6}
                ordered=sorted(info["rungs"],key=lambda v:(ranks[(v["run"].split("-")[2],v["step"])],v["run"].split("-")[3]!="hybrid"))
                job=next((v for v in ordered if v["evaluated"] and not (receipt_dir/(v["run"]+"-"+str(v["step"])+".json")).is_file()),None)
                if job:
                    report(key,"archiving",session=name,run=job["run"],step=job["step"])
                    transfer(a.repo,name,job["run"],job["step"],job["sha"],receipt_dir)
                failures=0
                if phase=="failed" and job is None:
                    report(key,"failed",diagnostic=info.get("diagnostic"),session=name)
                    return
                if phase=="draining" and not job:
                    secondary=a.control/"pair512.json"
                    fallback=secondary.is_file() and json.loads(secondary.read_bytes()).get("phase") in ("capacity_unavailable","quota_refused","setup_failed")
                    if key=="pair511" and fallback and not second_reused and info["lane"].get("seed")==511 and bound(4200):
                        launch(name,generation,512); second_reused=True; time.sleep(15); continue
                    report(key,"durable",session=name,rungs=len(info["rungs"])); return
            except (OSError,ValueError,StopIteration,RuntimeError,subprocess.SubprocessError) as error:
                failures+=1
                report(key,"retrying",session=name,attempt=failures,reason=str(error)[:180])
                if failures>=3 and not assigned(live(),name):
                    report(key,"lost",session=name); return
            if job is None or failures:
                time.sleep(35 if not failures else min(180,20*failures))
    def stop(name):
        try: cmd([*wrapper,"stop",name],timeout=100)
        except subprocess.SubprocessError: pass
    def secondary():
        key="pair512"; report(key,"waiting_for_allocation",earliest_unix=a.not_before)
        while time.time()<a.not_before and bound(2400):
            time.sleep(min(30,max(.2,a.not_before-time.time())))
        if not bound(2400):
            report(key,"budget_bound"); return
        names=list(zip(SECONDARY,("v6e","v5e")))
        current=live()
        active=[(name,gen) for name,gen in names if assigned(current,name)]
        if active:
            name,generation=active[0]
            try:
                info=inventory(name)
                if info["lane"]:
                    watch(name,generation,key,512)
                else:
                    prepare(name,generation); watch(name,generation,key,512)
            finally:
                stop(name)
            return
        for name,generation in names:
            listing=live()
            if not assigned(listing,name):
                if "[?]" in listing and any("TPU" in line for line in listing.splitlines() if line.startswith("[?]")):
                    report(key,"capacity_unavailable",reason="unmapped account TPU"); return
                if sum("Variant: TPU" in line for line in listing.splitlines())>=2:
                    report(key,"quota_refused",reason="two native TPU assignments already active"); return
                report(key,"requesting",name=name,generation=generation)
                try: cmd([*wrapper,"new",name,generation+"1"],timeout=530,capture_output=True,text=True)
                except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
                    diag=(str(getattr(error,"stdout",""))+str(getattr(error,"stderr",""))).lower()
                    if not assigned(live(),name):
                        if "precondition failed" in diag or "too many assignments" in diag:
                            report(key,"quota_refused",generation=generation); return
                        report(key,"allocation_failed",generation=generation,reason="provider returned no matching assignment"); continue
            if not assigned(live(),name): continue
            try:
                report(key,"preparing",session=name,generation=generation)
                prepare(name,generation); watch(name,generation,key,512)
            except (OSError,RuntimeError,ValueError,subprocess.SubprocessError) as e:
                report(key,"setup_failed",reason=str(e)[:160])
            finally: stop(name)
            return
        report(key,"capacity_unavailable")
    def primary():
        if not assigned(live(),PRIMARY):
            report("pair511","lost",session=PRIMARY); return
        try: watch(PRIMARY,"v5e","pair511",511)
        finally: stop(PRIMARY)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(primary),pool.submit(secondary)]
        for f in futures: f.result()
    (a.budget.parent/"requests-done").touch()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--repo",required=True,type=Path)
    p.add_argument("--control",required=True,type=Path); p.add_argument("--budget",required=True,type=Path)
    p.add_argument("--revision",required=True); p.add_argument("--not-before",required=True,type=float)
    operations(p.parse_args())
