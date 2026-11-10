"""Drain a bounded in-memory file-API window into the same dual-HDD receiver as SSH."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import fcntl
import hashlib
import json
import os
import select
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zlib
from g7_inputs import FED
from g7_ops import save

CLI=str(Path.home()/".local/bin/colab")
ENV=dict(os.environ,CLOUDSDK_CONFIG=str(Path.home()/".config/slwm-colab/gcloud"))
CHUNK=16*2**20


def file_data(session,remote,limit):
    reader,writer=os.pipe()
    try:
        proc=subprocess.Popen([CLI,"--auth","adc","download","-s",session,remote,"/dev/fd/"+str(writer)],env=ENV,
                              pass_fds=(writer,),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    except BaseException:
        os.close(reader);raise
    finally:os.close(writer)
    data=bytearray();deadline=time.monotonic()+160
    try:
        while True:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError("Native file request exceeded its deadline")
            if not select.select([reader],[],[],min(remaining,10))[0]:continue
            part=os.read(reader,min(1<<20,limit+1-len(data)))
            if not part:break
            data.extend(part)
            if len(data)>limit:raise ValueError("Oversized native file packet")
        rc=proc.wait(timeout=20)
        if rc:raise OSError("Native file download refused or interrupted")
        return data
    finally:
        os.close(reader)
        if proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=6)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()


def unpack_chunk(session,path,block):
    wire=file_data(session,path+"/"+block["file"],block["compressed"])
    if len(wire)!=block["compressed"] or hashlib.sha256(wire).hexdigest()!=block["sha"]:raise ValueError("Native packet checksum mismatch")
    z=zlib.decompressobj();shuffled=z.decompress(wire,block["bytes"]+1)
    if len(shuffled)!=block["bytes"] or not z.eof or z.unused_data or z.unconsumed_tail:raise ValueError("Invalid compressed optimizer block")
    # The Mac process uses standard Python and a bounded memory window; raw bytes never land on its disk.
    n=len(shuffled)//4;out=bytearray(len(shuffled))
    for i in range(4):out[i:4*n:4]=shuffled[i*n:(i+1)*n]
    out[4*n:]=shuffled[4*n:]
    if hashlib.sha256(out).hexdigest()!=block["raw_sha"]:raise ValueError("Optimizer block did not reconstruct bit for bit")
    return out


def bundle(repo,session,tag,step,expected):
    with tempfile.NamedTemporaryFile("w",suffix=".py") as f:
        code="import sys\nsys.path.insert(0,'/content/g8-code')\nfrom scripts.train.g8_file_blocks import build\nbuild(%r,%r,%r)\n"%(tag,step,expected)
        f.write(code);f.flush()
        subprocess.run(["bash","scripts/train/colab_session.sh","exec",session,"-f",f.name,"--timeout","200"],cwd=repo,env=ENV,check=True,
                       timeout=235,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    remote=f"/content/g8-packets/{tag}/step-{step:06d}"
    manifest=json.loads(file_data(session,remote+"/manifest.json",16*2**20));header=manifest["header"]
    if (header["run"],header["step"],header["sha"])!=(tag,step,expected) or not 8*2**20<=header["bytes"]<=4*2**30:raise ValueError("Wrong native transfer manifest")
    pos=0
    for i,c in enumerate(manifest["chunks"]):
        if c["file"]!=f"{i:05d}.bin" or c["offset"]!=pos or not 0<c["bytes"]<=CHUNK or not 0<c["compressed"]<=CHUNK+2**16:raise ValueError("Malformed block coordinates")
        pos+=c["bytes"]
    if pos!=header["bytes"]:raise ValueError("Incomplete optimizer block table")
    return remote,manifest


def transfer(a,name,stage):
    tag,step,expected=stage["run"],int(stage["step"]),stage["sha"]
    import re
    if not re.fullmatch(r"g8-v5e-anchor-(?:gpt2|hybrid)-(?:511|512)-(?:sim|other|replay|fresh)",tag) or step not in (10240,12288,16384) or not re.fullmatch("[a-f0-9]{64}",expected):raise ValueError("Unexpected study run")
    file=a.control/"receipts"/f"{tag}-{step}.json"
    with file.with_suffix(".lock").open("a") as fd:
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
        if file.exists():return False
        t=time.monotonic();base="cd /home/db/dev/slwm-g8 && python3 -m scripts.train.g6_artifacts "
        options=f" --run {tag} --step {step} --sha {expected}"
        info=json.loads(subprocess.check_output([*FED,base+"offset"+options],text=True,timeout=40))
        wire=0
        if not info["verified"]:
            offset=int(info["offset"]);remote,m=bundle(a.repo,name,tag,step,expected);header=dict(m["header"],offset=offset)
            blocks=[b for b in m["chunks"] if b["offset"]+b["bytes"]>offset]
            proc=subprocess.Popen([*FED,base+"receive"+options],stdin=subprocess.PIPE)
            try:
                proc.stdin.write(json.dumps(header,separators=(",",":")).encode()+b"\n")
                with ThreadPoolExecutor(max_workers=3) as pool:
                    queue=[pool.submit(unpack_chunk,name,remote,b) for b in blocks[:3]]
                    for i,block in enumerate(blocks):
                        raw=queue.pop(0).result();skip=max(0,offset-block["offset"])
                        proc.stdin.write(memoryview(raw)[skip:]);wire+=block["compressed"]
                        if i+3<len(blocks):queue.append(pool.submit(unpack_chunk,name,remote,blocks[i+3]))
                proc.stdin.close()
                if proc.wait(timeout=180):raise RuntimeError("Dual-disk receiver failed")
            finally:
                if proc.stdin and not proc.stdin.closed:proc.stdin.close()
                if proc.poll() is None:proc.terminate();proc.wait(timeout=20)
        receipt=json.loads(subprocess.check_output([*FED,"cat",f"/mnt/slwm-checkpoints-a/SLWM/g8/{tag}/step-{step:06d}/verified.json"],text=True))
        if receipt["sha256"]["resume.npz"]!=expected or len(set(receipt["disk_uuids"]))!=2:raise ValueError("Missing physical verification")
        save(file,receipt)
        print("File API dual-disk restore",tag,step,"wire",wire,"seconds",round(time.monotonic()-t,1),flush=True)
        return True


def main(a):
    plan=json.loads((a.control/"assignments.json").read_bytes())
    for name in plan:
        subprocess.run([CLI,"--auth","adc","upload","-s",name,"scripts/train/g8_file_blocks.py","/content/g8-code/scripts/train/g8_file_blocks.py"],env=ENV,check=True,timeout=90)
    failures=0
    while True:
        b=json.loads((a.control/"budget.json").read_bytes())
        if b["phase"] in ("complete","releasing"):return
        jobs=[]
        for name in b["active"]:
            if name not in plan:continue
            try:
                result=subprocess.check_output(["bash","scripts/train/colab_session.sh","exec",name,"-f","scripts/train/g8_inventory.py","--timeout","42"],
                                               text=True,stderr=subprocess.DEVNULL,timeout=65)
                data=json.loads(next(l for l in result.splitlines() if l.startswith("SLWM_G8_STATUS ")).split(" ",1)[1])
            except (OSError,subprocess.SubprocessError,ValueError,StopIteration):continue
            for s in data["stages"]:
                if s["evaluated"] and not (a.control/"receipts"/f"{s['run']}-{s['step']}.json").exists():jobs.append((name,s))
        moved=False
        try:
            for name,stage in sorted(jobs,key=lambda v:(v[1]["step"],v[1]["run"]),reverse=True):
                if transfer(a,name,stage):moved=True;break
            failures=0
        except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
            failures+=1;print("Bounded file API backoff",type(exc).__name__,str(exc)[:130],flush=True)
            time.sleep(min(300,30*2**min(failures,3)))
        if not moved:time.sleep(20)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--repo",required=True,type=Path);p.add_argument("--control",required=True,type=Path)
    a=p.parse_args();os.chdir(a.repo);main(a)
