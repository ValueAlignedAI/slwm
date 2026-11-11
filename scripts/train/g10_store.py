"""Asynchronous native Colab packets and two-disk, hash-verified G10 stage custody."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import zlib
from g9_storage import FILES,disks,sha,write

CHUNK=16<<20
CONTEXT=re.compile(r"context-(hybrid|gpt2|gated)-(short|variable|long)-s100[123]")
ALIGN=re.compile(r"align-(hybrid|gpt2)-(paired|shifted|reconstruct)-s100[123]")


def relative(tag,step):
    if not ((CONTEXT.fullmatch(tag) and step in (2560,5120)) or (ALIGN.fullmatch(tag) and step==1536)):
        raise ValueError("Unregistered G10 stage")
    return Path(tag)/f"step-{step:05d}"


def roots():
    return tuple(x.parent.parent/"g10/runs" for x in disks())


def pack(tag,step):
    import numpy as np
    rel=relative(tag,step);source=Path("/content/g10-out")/rel;dest=Path("/content/g10-packets")/rel
    if (dest/"manifest.json").is_file():return
    dest.mkdir(exist_ok=True,parents=True);r=json.loads((source/"receipt.json").read_text())
    if (r["tag"],r["step"])!=(tag,step):raise ValueError("Stage identity changed")
    doc={"receipt":r,"logs":(source.parent/"train.jsonl").read_text(),"files":{}}
    for filename,label in FILES.items():
        parts=[];digest=hashlib.sha256();path=source/filename
        with path.open("rb") as src:
            while raw:=src.read(CHUNK):
                n=len(raw)//4*4;plane=np.frombuffer(raw[:n],np.uint8).reshape(-1,4).T.copy().tobytes()+raw[n:]
                wire=zlib.compress(plane,1);digest.update(raw);name=f"{filename}.{len(parts):04d}"
                (dest/name).write_bytes(wire)
                parts.append(dict(file=name,compressed=len(wire),bytes=len(raw),sha=hashlib.sha256(wire).hexdigest(),raw_sha=hashlib.sha256(raw).hexdigest()))
        if digest.hexdigest()!=r[label]:raise ValueError("Runtime source checksum mismatch")
        doc["files"][filename]=dict(bytes=path.stat().st_size,parts=parts)
    write(dest/"manifest.json",doc)


def mirror(tag,step):
    rel=relative(tag,step);a,b=roots();stage=a/rel;other=b/rel
    if (stage/"verified.json").exists() and (other/"verified.json").exists():return
    info=json.loads((stage/"receipt.json").read_text())
    if (info["tag"],info["step"])!=(tag,step):raise ValueError("Checkpoint identity changed")
    other.mkdir(exist_ok=True,parents=True)
    with (stage/".mirror.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (stage/"verified.json").exists() and (other/"verified.json").exists():return
        for filename,label in FILES.items():
            if sha(stage/filename)!=info[label]:raise ValueError("Primary digest mismatch")
            target=other/filename
            if not target.exists() or sha(target)!=info[label]:
                tmp=target.with_suffix(".partial");shutil.copyfile(stage/filename,tmp)
                if sha(tmp)!=info[label]:raise ValueError("Secondary digest mismatch")
                tmp.replace(target)
        verified={k:info[k] for k in ("tag","step","inference_sha256","optimizer_sha256")}
        for root in (stage,other):
            write(root/"receipt.json",info);write(root/"verified.json",{**verified,"copies":2})
        if step==info["goal_steps"]:write(stage.parent/"done.json",info)


def component(mode,tag,step,filename,expected,size=0,offset=0):
    if filename not in FILES or not re.fullmatch("[0-9a-f]{64}",expected):raise ValueError("Unknown component")
    stage=roots()[0]/relative(tag,step);stage.mkdir(exist_ok=True,parents=True)
    target=stage/filename;tmp=stage/(filename+".partial")
    if mode=="offset":
        print(-1 if target.exists() and sha(target)==expected else tmp.stat().st_size if tmp.exists() else 0);return
    if not 0<size<=2**30:raise ValueError("Invalid component size")
    with (stage/(filename+".lock")).open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if offset!=(tmp.stat().st_size if tmp.exists() else 0):raise ValueError("Incoming offset changed")
        with tmp.open("ab") as fd:
            shutil.copyfileobj(sys.stdin.buffer,fd,1<<20);fd.flush();os.fsync(fd.fileno())
        if tmp.stat().st_size!=size:raise ValueError("Interrupted file")
        if sha(tmp)!=expected:
            tmp.replace(stage/(filename+".invalid"));raise ValueError("Incoming digest mismatch")
        tmp.replace(target)


def upload_code(name):
    from g9_cloud_archive import notebook
    paths=["scripts/train/g10_store.py","scripts/train/g9_storage.py"]
    source={p:base64.b64encode(Path(p).read_bytes()).decode() for p in paths}
    command="import pathlib,base64\nfor n,data in %r.items(): (pathlib.Path('/content/g10-code')/n).write_bytes(base64.b64decode(data))\nprint('G10_STORAGE_READY')"%source
    if "G10_STORAGE_READY" not in notebook(name,command,70):raise RuntimeError("Runtime storage installation failed")


def inventory(name):
    from g9_cloud_archive import notebook
    code="""import json,pathlib
r=pathlib.Path('/content/g10-out'); rows=[]
for file in sorted(r.glob('*/step-*/receipt.json')):
 d=json.loads(file.read_text())
 if (d['tag'].startswith('context-') and d['step'] in (2560,5120)) or (d['tag'].startswith('align-') and d['step']==1536):
  rows.append([d['tag'],d['step'],d['inference_sha256']])
print('G10_INFO '+json.dumps(dict(stages=rows,progress={p.name:json.loads(p.read_text()) for p in r.glob('*-progress.json')})))"""
    output=notebook(name,code,60)
    return json.loads(next(s[9:] for s in output.splitlines() if s.startswith("G10_INFO ")))


def transfer(name,row):
    from g7_inputs import FED
    from g8_file_transfer import file_data,unpack_chunk
    from g9_cloud_archive import notebook
    tag,step,expected=row;rel=relative(tag,step)
    code="import subprocess,sys\nsubprocess.run([sys.executable,'/content/g10-code/scripts/train/g10_store.py','pack',%r,%r],check=True)\nprint('G10_PACKET_OK')"%(tag,str(step))
    if "G10_PACKET_OK" not in notebook(name,code,200).splitlines():raise RuntimeError("Runtime failed to package "+tag)
    remote="/content/g10-packets/"+str(rel);doc=json.loads(file_data(name,remote+"/manifest.json",2**21))
    r=doc["receipt"]
    if (r["tag"],r["step"],r["inference_sha256"])!=(tag,step,expected):raise ValueError("Incoming generation changed")
    cmd="cd /home/db/dev/slwm-g10 && python3 scripts/train/g10_store.py "
    for filename,label in FILES.items():
        spec=doc["files"][filename];args=shlex.join([tag,str(step),filename,r[label]])
        offset=int(subprocess.check_output([*FED,cmd+"offset "+args],text=True,timeout=110))
        if offset<0:continue
        receiver=subprocess.Popen([*FED,cmd+"receive "+args+" "+str(spec["bytes"])+" "+str(offset)],stdin=subprocess.PIPE)
        try:
            parts=spec["parts"][offset//CHUNK:];skip=offset%CHUNK
            with ThreadPoolExecutor(3) as pool:
                queue=[pool.submit(unpack_chunk,name,remote,b) for b in parts[:3]]
                for i in range(len(parts)):
                    raw=queue.pop(0).result();receiver.stdin.write(memoryview(raw)[skip:]);skip=0
                    if i+3<len(parts):queue.append(pool.submit(unpack_chunk,name,remote,parts[i+3]))
            receiver.stdin.close()
            if receiver.wait(timeout=160):raise RuntimeError("Fedora rejected the generation")
        finally:
            if receiver.poll() is None:receiver.terminate();receiver.wait(timeout=30)
    subprocess.run([*FED,cmd+"finish "+shlex.join([tag,str(step)])],input=json.dumps(dict(receipt=r,logs=doc["logs"])).encode(),check=True,timeout=400)
    return r


def drain(name,ctl):
    receipts=ctl/"receipts";receipts.mkdir(exist_ok=True)
    installed=False;until=json.loads((ctl/"bounds.json").read_text())["deadline"]
    while time.time()<until:
        budget=json.loads((ctl/"budget.json").read_text())
        if budget["phase"] in ("complete","releasing"):return
        try:
            if not installed:upload_code(name);installed=True
            data=inventory(name);write(ctl/(name+"-runtime.json"),data)
            todo=[r for r in data["stages"] if not (receipts/f"{r[0]}-{r[1]}.json").exists()]
            if todo:
                row=min(todo,key=lambda r:(r[1]==2560,r[0],r[1]));write(ctl/(name+"-archive.json"),dict(stage=row[:2],phase="copying"))
                doc=transfer(name,row);write(receipts/f"{row[0]}-{row[1]}.json",doc);continue
            if (ctl/(name+"-release")).exists():
                subprocess.run(["bash","scripts/train/colab_session.sh","stop",name],check=True,timeout=120)
                write(ctl/(name+"-archive.json"),dict(phase="released",count=len(data["stages"])));return
            time.sleep(16)
        except (OSError,ValueError,StopIteration,RuntimeError,subprocess.SubprocessError) as exc:
            write(ctl/(name+"-archive.json"),dict(phase="retry",error=str(exc)[:240]));time.sleep(12)


def local():
    a,_=roots()
    while not (a.parent/"finalized").exists():
        for p in sorted(a.glob("*/step-*/receipt.json")):
            tag=p.parent.parent.name;step=int(p.parent.name.split("-")[1])
            try:relative(tag,step)
            except ValueError:continue
            if not (p.parent/"verified.json").exists():mirror(tag,step)
        time.sleep(12)


if __name__=="__main__":
    op=sys.argv[1];args=sys.argv[2:]
    if op=="local":local()
    elif op=="drain":drain(args[0],Path(args[1]))
    else:
        tag,step=args[0],int(args[1]);rel=relative(tag,step)
        if op=="pack":pack(tag,step)
        elif op in ("offset","receive"):component(op,tag,step,*args[2:4],*map(int,args[4:]))
        elif op=="finish":
            msg=json.load(sys.stdin);root=roots()[0]/rel;write(root/"receipt.json",msg["receipt"])
            write(root.parent/"recipe.json",msg["receipt"]);(root.parent/"train.jsonl").write_text(msg["logs"]);mirror(tag,step)
        else:raise ValueError(op)
