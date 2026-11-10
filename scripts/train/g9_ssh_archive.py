"""Drain intermediate recovery generations over the independent official SSH transport."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from g7_inputs import FED
from g9_cloud_archive import FED_CMD,WRAP,notebook
from g9_storage import FILES,location,write


def copy(name,tag,step,expected):
    rel=location(tag,step);stage="/content/g9-out/"+str(rel)
    code="import json,pathlib\np=pathlib.Path(%r)\nr=json.loads((p/'receipt.json').read_text())\nprint('G9_META '+json.dumps(dict(receipt=r,logs=(p.parent/'train.jsonl').read_text(),size={k:(p/k).stat().st_size for k in ('weights.npz','optimizer.npz')})))"%stage
    out=notebook(name,code,60);doc=json.loads(next(s[8:] for s in out.splitlines() if s.startswith("G9_META ")))
    record=doc["receipt"]
    if record["tag"]!=tag or record["step"]!=step or record["inference_sha256"]!=expected:raise ValueError("Intermediate generation mismatch")
    for filename,field in FILES.items():
        digest=record[field];args=shlex.join([tag,str(step),filename,digest]);size=doc["size"][filename]
        offset=int(subprocess.check_output([*FED,FED_CMD+"offset "+args],text=True,timeout=110))
        if offset<0:continue
        source_code="import sys,shutil\nf=open(sys.argv[1],'rb');f.seek(int(sys.argv[2]));shutil.copyfileobj(f,sys.stdout.buffer,4*1024*1024)"
        with subprocess.Popen([*WRAP,"ssh",name,"python","-c",source_code,stage+"/"+filename,str(offset)],stdout=subprocess.PIPE) as source:
            target=subprocess.Popen([*FED,FED_CMD+"receive "+args+" "+str(size)+" "+str(offset)],stdin=source.stdout)
            source.stdout.close()
            try:
                if target.wait(timeout=1600) or source.wait(timeout=45):raise RuntimeError("Intermediate direct stream interrupted")
            finally:
                if target.poll() is None:target.terminate()
                if source.poll() is None:source.terminate()
    subprocess.run([*FED,FED_CMD+"finish "+shlex.join([tag,str(step)])],input=json.dumps(dict(receipt=record,logs=doc["logs"])).encode(),check=True,timeout=600)
    return record


def main(ctl):
    name="slwm-g9-boreal";receipts=ctl/"receipts"
    while time.time()<json.loads((ctl/"bounds.json").read_text())["deadline"]:
        try:
            stages=json.loads((ctl/(name+"-live.json")).read_text())["completed"]
            todo=[v for v in stages if v[1]==1024 and not (receipts/f"{v[0]}-1024.json").exists()]
            if not todo:
                if len([v for v in stages if v[1]==1024])==16:return
                time.sleep(15);continue
            # Small intermediate checkpoints first, freeing the equal-exposure analysis quickly.
            # The native drain takes ascending tags. Work from the other end of each tier.
            small=[v for v in todo if v[0].startswith("exposure")]
            if len(todo)==1 and sum(v[1]==1024 for v in stages)==16:return
            tag,step,expected=max(small or todo,key=lambda v:v[0])
            write(ctl/"intermediate-transfer.json",dict(phase="copying",tag=tag,step=step))
            record=copy(name,tag,step,expected);write(receipts/f"{tag}-{step}.json",record)
        except (OSError,ValueError,StopIteration,subprocess.SubprocessError,RuntimeError) as exc:
            write(ctl/"intermediate-transfer.json",dict(phase="retry",error=str(exc)[:160]));time.sleep(12)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("control",type=Path);main(ap.parse_args().control)
