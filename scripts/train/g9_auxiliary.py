"""Move compact decision heads and new-cohort scores into verified Fedora storage."""
import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from g9_storage import disks,sha,write

TAGS=[f"confirm-{c}-z{w}-gelu-s{s}-fresh" for s in (911,912) for w in (128,512) for c in ("hybrid","gpt2")]
SAFE={*(f"horizon/{tag}.json" for tag in TAGS),
      *(f"heads/{tag}/{file}" for tag in TAGS if "-z512-" in tag for file in ("posttraining.json","random-head.npz","parent-head.npz"))}


def collect(ctl):
    from g7_inputs import FED
    from g8_file_transfer import file_data
    from g9_cloud_archive import notebook
    name="slwm-g9-boreal"
    code="""from pathlib import Path
import hashlib,tarfile
wanted=%r
paths=[('/content/g9-'+entry.split('/',1)[0]+'/'+entry.split('/',1)[1],entry) for entry in wanted]
if all(Path(path).is_file() for path,_ in paths):
 with tarfile.open('/content/g9-auxiliary.tgz','w:gz') as dst:
  for path,entry in paths:dst.add(path,arcname=entry,recursive=False)
 p=Path('/content/g9-auxiliary.tgz')
 print('G9_AUX',p.stat().st_size,hashlib.file_digest(p.open('rb'),'sha256').hexdigest())
else:
 print('G9_WAIT',sum(Path(p).exists() for p,_ in paths))
"""%sorted(SAFE)
    while time.time()<json.loads((ctl/"bounds.json").read_text())["deadline"]:
        try:
            out=notebook(name,code,55)
            ready=next((s for s in out.splitlines() if s.startswith("G9_AUX ")),None)
            if ready:
                _,size,expected=ready.split()
                if int(size)>12*2**20:raise ValueError("Unexpected auxiliary archive size")
                data=file_data(name,"/content/g9-auxiliary.tgz",int(size))
                if hashlib.sha256(data).hexdigest()!=expected:raise ValueError("Auxiliary transfer checksum mismatch")
                subprocess.run([*FED,"cd /home/db/dev/slwm-g9 && python3 scripts/train/g9_auxiliary.py receive"],input=data,check=True,timeout=140)
                write(ctl/"auxiliary-verified.json",dict(sha256=expected,files=len(SAFE)));return
            state=next((s for s in out.splitlines() if s.startswith("G9_WAIT ")),None)
            write(ctl/"auxiliary-progress.json",dict(status=state,utc_unix=time.time()))
        except (ValueError,OSError,RuntimeError,subprocess.SubprocessError) as exc:
            write(ctl/"auxiliary-progress.json",dict(status="retry",error=str(exc)[:120],utc_unix=time.time()))
        time.sleep(30)


def receive():
    data=sys.stdin.buffer.read(12*2**20+1)
    if len(data)>12*2**20:raise ValueError("Oversized auxiliary archive")
    a,b=disks();receipt={}
    with tarfile.open(fileobj=io.BytesIO(data),mode="r:gz") as tape:
        members=tape.getmembers()
        if len(members)!=len(SAFE) or {m.name for m in members}!=SAFE:raise ValueError("Unexpected auxiliary files")
        for m in members:
            if not m.isfile() or m.size>2**20:raise ValueError("Invalid auxiliary object")
            raw=tape.extractfile(m).read();digest=hashlib.sha256(raw).hexdigest();receipt[m.name]=digest
            for root in (a.parent,b.parent):
                path=root/"auxiliary"/m.name;path.parent.mkdir(exist_ok=True,parents=True)
                if path.exists() and sha(path)!=digest:raise ValueError("Conflicting auxiliary generation")
                if not path.exists():
                    tmp=path.with_suffix(".new");tmp.write_bytes(raw);tmp.replace(path)
                if sha(path)!=digest:raise ValueError("Auxiliary readback failed")
        for tag in TAGS:
            if "-z512-" not in tag:continue
            doc=json.loads((a.parent/"auxiliary/heads"/tag/"posttraining.json").read_text())
            for name in ("random","parent"):
                if receipt[f"heads/{tag}/{name}-head.npz"]!=doc["arms"][name]["sha256"]:raise ValueError("Decision head receipt mismatch")
    for root in (a.parent,b.parent):write(root/"auxiliary/verified.json",receipt)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("op",choices=["collect","receive"]);p.add_argument("control",type=Path,nargs="?")
    a=p.parse_args();collect(a.control) if a.op=="collect" else receive()
