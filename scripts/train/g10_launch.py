"""Stage a named G10 Colab assignment and run a disjoint seed through Google's native transport."""
import argparse
import json
import shlex
import subprocess
from pathlib import Path
from g7_inputs import FED,shell,stream

NAMES=("slwm-g10-context-a","slwm-g10-context-b")
FILES=["models/g4_shared.py","models/g5_shared.py","models/g9_shared.py","models/g9_tabletop.py","models/g10_context.py",
       "scripts/train/g9_fit.py","scripts/train/g10_fit.py","configs/t10/study.json"]


def main(name,rev,seeds):
    cmd=["bash","scripts/train/colab_session.sh"]
    subprocess.run([*cmd,"exec",name,"-f","scripts/train/g4_colab_topology.py","--timeout","90"],check=True,timeout=130)
    stream(name,["git","archive",rev,*FILES],"set -eu; mkdir -p /content/g10-code /content/g10-data /content/g10-out; cd /content/g10-code; tar -xf -")
    path="/mnt/slwm-data/SLWM/g9/data/starss-train.npz"
    digest=subprocess.check_output([*FED,"sha256sum "+path],text=True,timeout=25).split()[0]
    stream(name,[*FED,"cat "+path],"set -eu; cd /content/g10-data; cat > .input; echo '"+digest+"  .input' | sha256sum -c -; mv .input starss-train.npz")
    run="set -e; cd /content/g10-code; source /content/g4-topology.env; export PYTHONPATH=. JAX_PLATFORMS=tpu OMP_NUM_THREADS=2; exec python -u scripts/train/g10_fit.py /content/g10-data/starss-train.npz /content/g10-out --cache /content/g10-xla --seeds "+" ".join(map(str,seeds))
    shell(name,"nohup flock -n /content/g10-train.lock bash -c "+shlex.quote(run)+" </dev/null >>/content/g10-out/context.log 2>&1 & echo $! >/content/g10-out/context.pid; echo G10_LAUNCHED",timeout=85)
    print(name,seeds,rev,digest,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("name",choices=NAMES);p.add_argument("rev");p.add_argument("seeds",nargs="+",type=int,choices=(1001,1002,1003))
    a=p.parse_args();main(a.name,a.rev,a.seeds)
