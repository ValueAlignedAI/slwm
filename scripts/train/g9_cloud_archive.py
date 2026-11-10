"""Drain named G9 runtime milestones without storing model or optimizer bytes on the Mac."""
import argparse
import json
import shlex
import subprocess
import tempfile
import time
import zlib
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from g7_inputs import FED,stream
from g8_file_transfer import file_data
from g9_storage import CHUNK,FILES,location,write

WRAP=["bash","scripts/train/colab_session.sh"]
FED_CMD="cd /home/db/dev/slwm-g9 && python3 scripts/train/g9_storage.py "


def notebook(name,script,timeout=100):
    with tempfile.NamedTemporaryFile("w",suffix=".py") as f:
        f.write(script);f.flush()
        return subprocess.check_output([*WRAP,"exec",name,"-f",f.name,"--timeout",str(timeout)],stderr=subprocess.DEVNULL,text=True,timeout=timeout+40)


def archive(name,tag,step,expected):
    relative=location(tag,step)
    notebook(name,"import sys\nsys.path.insert(0,'/content/g9-code/scripts/train')\nfrom g9_storage import bundle\nbundle(%r,%d)"%(tag,step),240)
    path="/content/g9-packets/"+str(relative)
    doc=json.loads(file_data(name,path+"/manifest.json",2**21))
    receipt=doc["receipt"]
    if receipt["inference_sha256"]!=expected or receipt["tag"]!=tag or receipt["step"]!=step:raise ValueError("Generation mismatch")
    for filename,field in FILES.items():
        f=doc["files"][filename];size=f["bytes"];digest=receipt[field]
        args=shlex.join([tag,str(step),filename,digest])
        offset=int(subprocess.check_output([*FED,FED_CMD+"offset "+args],text=True,timeout=120))
        if offset<0:continue
        def block(part):
            data=file_data(name,path+"/"+part["name"],part["wire"])
            if hashlib.sha256(data).hexdigest()!=part["sha"]:raise ValueError("Block checksum mismatch")
            dec=zlib.decompressobj();raw=dec.decompress(data,part["raw"]+1)
            if not dec.eof or dec.unused_data or len(raw)!=part["raw"] or len(raw)>CHUNK:raise ValueError("Invalid checkpoint packet")
            return raw
        pieces=f["parts"][offset//CHUNK:];skip=offset%CHUNK
        proc=subprocess.Popen([*FED,FED_CMD+"receive "+args+" "+str(size)+" "+str(offset)],stdin=subprocess.PIPE)
        try:
            with ThreadPoolExecutor(4) as pool:
                waiting=[pool.submit(block,b) for b in pieces[:4]]
                for i,part in enumerate(pieces):
                    raw=waiting.pop(0).result();proc.stdin.write(memoryview(raw)[skip:]);skip=0
                    if i+4<len(pieces):waiting.append(pool.submit(block,pieces[i+4]))
            proc.stdin.close()
            if proc.wait(timeout=200):raise RuntimeError("HDD receiver rejected a checkpoint")
        finally:
            if proc.poll() is None:proc.terminate();proc.wait(timeout=30)
    subprocess.run([*FED,FED_CMD+"finish "+shlex.join([tag,str(step)])],input=json.dumps({"receipt":receipt,"logs":doc["logs"]}).encode(),check=True,timeout=600)
    return receipt


def monitor(name,ctl,revision,expected):
    for attempt in range(10):
        try:
            stream(name,["git","archive",revision,"scripts/train/g9_storage.py"],"set -eu; cd /content/g9-code; tar -xf -")
            break
        except (OSError,RuntimeError,subprocess.SubprocessError):
            if attempt==9:raise
            time.sleep(min(30,4+3*attempt))
    receipts=ctl/"receipts";receipts.mkdir(exist_ok=True)
    while True:
        b=json.loads((ctl/"budget.json").read_text())
        if b["phase"] in ("complete","releasing"):return
        try:
            output=subprocess.check_output([*WRAP,"exec",name,"-f","scripts/train/g9_inventory.py","--timeout","60"],text=True,stderr=subprocess.DEVNULL,timeout=95)
            status=json.loads(next(s[8:] for s in output.splitlines() if s.startswith("SLWM_G9 ")))
            write(ctl/(name+"-runtime.json"),status)
            pending=[x for x in status["completed"] if not (receipts/f"{x[0]}-{x[1]}.json").exists()]
            if pending:
                tag,step,sha=min(pending,key=lambda x:(x[1]!=4096,x[0]))
                write(ctl/(name+"-transfer.json"),dict(run=tag,step=step,phase="copying"))
                receipt=archive(name,tag,step,sha);write(receipts/f"{tag}-{step}.json",receipt)
                continue
            if len(status["completed"])==expected and all(sum(x[1]==t for x in status["completed"])==expected//2 for t in (1024,4096)):
                subprocess.run([*WRAP,"stop",name],check=True,timeout=120)
                write(ctl/(name+"-transfer.json"),dict(phase="released",verified=expected));return
            time.sleep(20)
        except (OSError,ValueError,StopIteration,RuntimeError,subprocess.SubprocessError) as exc:
            write(ctl/(name+"-transfer.json"),dict(phase="retry",error=str(exc)[:210]));time.sleep(16)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("name",choices=["slwm-g9-boreal","slwm-g9-aurora"]);p.add_argument("control",type=Path)
    p.add_argument("revision");p.add_argument("--generations",type=int,default=8)
    a=p.parse_args();monitor(a.name,a.control,a.revision,a.generations)
