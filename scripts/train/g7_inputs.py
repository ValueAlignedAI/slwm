"""Send code, verified media and exact parents through the official CLI without Mac-side data files."""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import tempfile

FED=["ssh","-o","BatchMode=yes","-o","ConnectTimeout=10","-o","ServerAliveInterval=30","db@192.168.1.200"]
DATA="/mnt/slwm-data/SLWM/g5"; G7="/mnt/slwm-data/SLWM/g7"; HDD="/mnt/slwm-checkpoints-a/SLWM/g6"
BASE_FILES=[f"{family}-{s}.tar.zst" for s in ("dev","train") for family in ("pusht","impact","paired")]


def shell(name,cmd,**kw):
    return subprocess.run(["bash","scripts/train/colab_session.sh","ssh",name,"bash","-c",cmd],check=True,**kw)


def stream(name,src,dst):
    with subprocess.Popen(src,stdout=subprocess.PIPE) as source:
        try:shell(name,dst,stdin=source.stdout,timeout=2400)
        finally:source.stdout.close()
        if source.wait(timeout=50):raise RuntimeError("Input source did not finish")


def code(name):
    stream(name,["git","archive","HEAD","data","models","training","configs/t5","configs/t6","configs/t7","scripts/train/g5_cloud_data.py",
                 "scripts/train/g6_ram_inputs.py","scripts/train/g7_lane.py"],
           "set -eu; mkdir -p /content/g7-code; cd /content/g7-code; tar -xf -; python -c 'import importlib.util; assert importlib.util.find_spec(\"training.g7_continue\")'")


def parent(name,seed,size,core):
    tag=f"g6-v5e-{size}-{core}-{seed}/step-008192"
    base="/content/g5-landing/"+tag
    test=shell(name,f"if test -s {base}/resume.npz && test -s {base}/verified.json; then stat -c %s {base}/resume.npz; else echo 0; fi",
               text=True,capture_output=True).stdout.strip()
    expected=int(subprocess.check_output([*FED,"stat","-c","%s",HDD+"/"+tag+"/resume.npz"],text=True))
    if int(test)!=expected:
        files=[tag+"/"+k for k in ("config.json","verified.json","resume.npz")]
        stream(name,[*FED,"tar","-cf","-","-C",HDD,*files],"set -eu; mkdir -p /content/g5-landing; cd /content/g5-landing; tar -xf -")
    # The trainer also verifies the parent content and full config digest before first use.
    shell(name,f"cd {base} && python -c 'import hashlib,json,pathlib; x=json.loads(pathlib.Path(\"verified.json\").read_bytes())[\"sha256\"][\"resume.npz\"]; assert hashlib.file_digest(open(\"resume.npz\",\"rb\"),\"sha256\").hexdigest()==x; pathlib.Path(\".sha-checked\").write_text(x); pathlib.Path(\".received\").touch()'")
    print("Verified parent",name,size,core,seed,flush=True)


def bootstrap(name,gen,seed,deadline):
    code(name)
    shell(name,"set -eu; cd /content/g7-code; python -c 'import optax,scipy,zstandard' || python -m pip -q install optax scipy zstandard; "
          "if ! test -f /content/g7-base-ready; then mkdir -p /content/g7-data; "
          "python scripts/train/g5_cloud_data.py dev --root /content/g7-data --pins /content/g5-landing & pid=$!; "
          "python scripts/train/g5_cloud_data.py train --root /content/g7-data --pins /content/g5-landing; wait \"$pid\"; "
          "cp /content/g5-landing/prior.npz /content/g7-data/prior.npz; python scripts/train/g6_ram_inputs.py /content/g7-data; touch /content/g7-base-ready; fi",
          timeout=650)
    core="hybrid" if seed==511 else "gpt2"
    parent(name,seed,"anchor",core)
    # Queue behind a possibly active first manual rung. The exclusive lane lock prevents duplicate supervisors.
    shell(name,"set -eu; cd /content/g7-code; mkdir -p /content/g7-out; source /content/g4-topology.env; "
          "export JAX_PLATFORMS=tpu JAX_COMPILATION_CACHE_DIR=/content/g7-xla JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2; "
          f"nohup flock -n /content/g7-lane.lock flock /content/g7-train.lock python -u scripts/train/g7_lane.py --generation {gen} --seed {seed} --deadline {deadline} "
          "</dev/null >>/content/g7-lane.log 2>&1 & echo lane-supervisor-queued")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("bootstrap","parents"));p.add_argument("--session",required=True)
    p.add_argument("--generation",choices=("v5e","v6e"),required=True);p.add_argument("--seed",type=int,choices=(511,512),required=True);a=p.parse_args()
    ctl=Path.home()/".config/slwm-colab/g7-20261110"
    with (ctl/(a.session+".transport")).open("a") as fd:
        fcntl.flock(fd,fcntl.LOCK_EX)
        if a.action=="bootstrap":bootstrap(a.session,a.generation,a.seed,int(json.loads((ctl/"bounds.json").read_bytes())["deadline"]))
        else:
            first="hybrid" if a.seed==511 else "gpt2";other="gpt2" if first=="hybrid" else "hybrid"
            for size,core in (("anchor",other),("larger",first),("larger",other)):parent(a.session,a.seed,size,core)
