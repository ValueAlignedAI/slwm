"""Test the exact proposed Git merge tree on Fedora with the verified project data."""
import json
import hashlib
from pathlib import Path
import re
import shlex
import subprocess
import time

FED=["ssh","-o","BatchMode=yes","-o","ConnectTimeout=10","db@192.168.1.200"]
ROOT="/mnt/slwm-data/SLWM/g7/integration"


def review(tree):
    if not re.fullmatch("[a-f0-9]{40}",tree):raise ValueError("A concrete Git tree is required")
    actual=subprocess.check_output(["git","cat-file","-t",tree],text=True).strip()
    if actual!="tree":raise ValueError("Expected proposed merge tree")
    folder=ROOT+"/"+tree;receipt=ROOT+"/logs/"+tree+".review.json"
    def ssh(command,**kw):return subprocess.run([*FED,command],**kw)
    def status():
        r=ssh("cat "+receipt,capture_output=True,text=True)
        return json.loads(r.stdout) if not r.returncode else None
    previous=status()
    signature=hashlib.sha256(Path(__file__).with_name("g7_review_suite.sh").read_bytes()).hexdigest()
    if previous and previous.get("runner_sha256")!=signature:
        stamp=str(int(time.time()))
        ssh("mv "+receipt+" "+receipt+".prior-"+stamp+" && cp "+ROOT+"/logs/"+tree+".review.log "+ROOT+"/logs/"+tree+".review.log.prior-"+stamp,check=True)
        previous=None
    if previous and not previous["exit"] and previous["passed"]>0:return previous
    ssh("mkdir -p "+folder+" "+ROOT+"/logs && test \"$(findmnt -nro TARGET -T "+folder+")\" = /mnt/slwm-data",check=True)
    if ssh("test -f "+folder+"/.slwm-export-complete").returncode:
        with subprocess.Popen(["git","archive","--format=tar",tree],stdout=subprocess.PIPE) as sender:
            result=ssh("tar -xf - -C "+folder,stdin=sender.stdout)
            sender.stdout.close()
            if sender.wait() or result.returncode:raise RuntimeError("Source tree transfer failed")
        ssh("touch "+folder+"/.slwm-export-complete",check=True)
    window="review-"+tree[:9]; command="bash /home/db/dev/slwm-g7/scripts/train/g7_review_suite.sh "+tree
    current=ssh("tmux list-windows -t slwm-g7 -F '#{window_name}'",text=True,capture_output=True,check=True).stdout.splitlines()
    if window not in current:
        if previous:raise RuntimeError("Prior exact-tree suite failed: "+previous["summary"])
        ssh("tmux new-window -d -t slwm-g7 -n "+window+" "+shlex.quote(command),check=True)
    until=time.monotonic()+1800
    while time.monotonic()<until:
        doc=status()
        if doc:
            if doc["tree"]!=tree or doc["exit"] or doc["passed"]<=0 or doc.get("runner_sha256")!=signature:raise RuntimeError("Scientific integration suite failed: "+str(doc))
            return doc
        time.sleep(4)
    raise TimeoutError("Exact-tree integration review timed out")


if __name__=="__main__":
    import sys
    print(json.dumps(review(sys.argv[1])),flush=True)
