"""Install reviewed source through the official notebook API and launch a finite named stage."""
import argparse
import base64
import json
from pathlib import Path
import shlex
import subprocess
from g9_cloud_archive import notebook

FILES=["evals/g9_live.py","evals/g10_control.py","models/g9_shared.py","models/g10_context.py","models/g10_alignment.py","scripts/train/g10_fit.py","scripts/train/g10_align_fit.py",
       "scripts/train/g10_lane.py","scripts/train/g10_store.py","scripts/train/g9_storage.py"]


def dispatch(name,seed,rev,operations):
    src={p:base64.b64encode(subprocess.check_output(["git","show",f"{rev}:{p}"])).decode() for p in FILES}
    setup="import pathlib,base64\nfor n,data in %r.items():\n p=pathlib.Path('/content/g10-code')/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(data))\nprint('G10_CODE_READY')"%src
    if "G10_CODE_READY" not in notebook(name,setup,90).splitlines():raise RuntimeError("Native source installation failed")
    run="set -eu; cd /content/g10-code; source /content/g4-topology.env; export PYTHONPATH=. JAX_PLATFORMS=tpu OMP_NUM_THREADS=2;"
    for op in operations:
        if op=="mixed":run+=" until test -s /content/g10-data/holo-ready; do sleep 10; done;"
        run+=" flock /content/g10-train.lock python -u scripts/train/g10_lane.py "+op+" /content/g10-out /content/g10-data /content/g10-eval "+str(seed)+" --cache /content/g10-xla;"
    label="-".join(operations)
    command="import subprocess,pathlib\nf=open('/content/g10-out/"+label+".log','ab'); proc=subprocess.Popen(['bash','-c',%r],stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True);print('G10_STAGE_PID',proc.pid)"%run
    result=notebook(name,command,70)
    if "G10_STAGE_PID" not in result:raise RuntimeError(result[-460:])
    print(name,seed,label,result.split("G10_STAGE_PID")[-1].strip(),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("name",choices=["slwm-g10-context-a","slwm-g10-context-b"]);p.add_argument("seed",type=int);p.add_argument("rev")
    p.add_argument("operations",nargs="+",choices=["context-eval","align-eval","rooms","mixed"]);a=p.parse_args();dispatch(a.name,a.seed,a.rev,a.operations)
