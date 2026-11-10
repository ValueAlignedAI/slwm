"""Transport dataset archives as resumable, content-addressed 16 MiB objects with SHA-256 receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from colab_artifacts import put_verified

BLOCK=16*1024**2
REMOTE="slwm-drive:SLWM/g5/data-blocks"


def ship(root):
    data=root/"cloud-blocks"
    data.mkdir(exist_ok=True)
    objects=data/"staging"; objects.mkdir(exist_ok=True)
    def send(part):
        with part.open("rb") as f:
            digest=hashlib.file_digest(f,"sha256").hexdigest()
        name=digest+".json"
        receipt=objects/name
        if not receipt.is_file():
            put_verified(part,REMOTE+"/objects/"+digest,digest)
            tmp=part.with_suffix(".receipt")
            tmp.write_text(json.dumps({"sha256":digest})+"\n")
            tmp.replace(receipt)
        part.unlink(missing_ok=True)
        return digest
    for role in ("dev","train","test"):
        summary={}
        for family in ("paired","impact","pusht"):
            name=f"{family}-{role}.tar.zst"
            path=root/"paired"/(role+".tar.zst") if family=="paired" else data/name
            if family=="paired" and not path.is_file():
                path=root/"paired"/role
                path=path.resolve().parent/(role+".tar.zst")
            if not path.is_file():
                path=data/name
                tmp=path.with_suffix(".part")
                subprocess.run(["tar","-chf",tmp,"--use-compress-program","zstd -q -1 -T2","-C",root/family,role],check=True)
                tmp.replace(path)
            futures=[]; whole=hashlib.sha256()
            with ThreadPoolExecutor(max_workers=5) as executor,path.open("rb") as f:
                while buf:=f.read(BLOCK):
                    whole.update(buf)
                    with tempfile.NamedTemporaryFile(dir=objects,delete=False,suffix=".part") as out:
                        out.write(buf); part=Path(out.name)
                    futures.append(executor.submit(send,part))
                parts=[future.result() for future in futures]
            summary[name]={"sha256":whole.hexdigest(),"bytes":path.stat().st_size,"parts":parts}
            print("verified",name,"bytes",path.stat().st_size,"objects",len(parts),flush=True)
        target=data/(role+"-blocks.json")
        tmp=target.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
        tmp.replace(target)
        with target.open("rb") as f:
            put_verified(target,REMOTE+"/"+target.name,hashlib.file_digest(f,"sha256").hexdigest())
        pins=data/(role+".sha256")
        tmp=pins.with_suffix(".tmp")
        tmp.write_text("".join(v["sha256"]+"  "+k+"\n" for k,v in summary.items()))
        tmp.replace(pins)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("root",type=Path); a=p.parse_args()
    if subprocess.check_output(["findmnt","-nro","TARGET","-T",a.root],text=True).strip()!="/mnt/slwm-data":
        raise SystemExit("Require the mounted Fedora data SSD")
    ship(a.root)
