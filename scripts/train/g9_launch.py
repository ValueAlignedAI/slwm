"""Stage the exact source revision and one verified real bank on a named, already owned Colab TPU."""
import argparse
import shlex
import subprocess
import time
from g7_inputs import FED, shell, stream

NAMES={"slwm-g9-aurora","slwm-g9-boreal"}
DATA="/mnt/slwm-data/SLWM/g9/data/"


def launch(name,revision,group):
    if name not in NAMES:raise ValueError(name)
    wrapper=["bash","scripts/train/colab_session.sh"]
    subprocess.run([*wrapper,"exec",name,"-f","scripts/train/g4_colab_topology.py","--timeout","75"],check=True,timeout=120)
    paths=["models/g4_shared.py","models/g5_shared.py","models/g9_tabletop.py","models/g9_shared.py",
           "scripts/train/g9_fit.py","configs/t9/study.json"]
    stream(name,["git","archive",revision,*paths],"set -eu; mkdir -p /content/g9-code /content/g9-data /content/g9-out; cd /content/g9-code; tar -xf -")
    check="test $(/usr/bin/stat -c %s "+DATA+"starss-train.npz) -gt 100000 && sha256sum "+DATA+"starss-train.npz"
    for _ in range(150):
        b=subprocess.run([*FED,check],text=True,capture_output=True,timeout=28)
        if b.returncode==0:break
        time.sleep(6)
    else:raise RuntimeError("No verified training bank")
    expected=b.stdout.split()[0]
    stream(name,[*FED,"cat "+DATA+"starss-train.npz"],"set -eu; cd /content/g9-data; cat > .input; echo '"+expected+"  .input' | sha256sum -c -; mv .input starss-train.npz")
    command="cd /content/g9-code; source /content/g4-topology.env; export JAX_PLATFORMS=tpu PYTHONPATH=. OMP_NUM_THREADS=3; exec python -u scripts/train/g9_fit.py --group "+shlex.quote(group)+" --data /content/g9-data --out /content/g9-out --cache /content/g9-xla"
    shell(name,"set -eu; cd /content/g9-code; nohup flock -n /content/g9-training.lock bash -c "+shlex.quote(command)+" </dev/null >>/content/g9-out/lane.log 2>&1 & echo $! > /content/g9-out/lane.pid; echo 'G9 training submitted'",timeout=120)
    print(name,group,expected,revision,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("name",choices=sorted(NAMES));p.add_argument("revision");p.add_argument("group",choices=["confirm","confirm-911","confirm-912"])
    a=p.parse_args();launch(a.name,a.revision,a.group)
