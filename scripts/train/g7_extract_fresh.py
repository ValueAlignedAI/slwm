"""Stream and verify a specific fresh role on a Colab runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
import zstandard

def load(archive,expected,target,role):
    target.mkdir(exist_ok=True,parents=True)
    with archive.open("rb") as fd:
        if hashlib.file_digest(fd,"sha256").hexdigest()!=expected:raise ValueError("Fresh archive changed")
    allowed=set(("crema","libri","text","events","sim"))
    with tempfile.TemporaryDirectory(dir=target,prefix=".loading-") as tmp:
        tmp=Path(tmp)
        with archive.open("rb") as fd,zstandard.ZstdDecompressor().stream_reader(fd) as stream,tarfile.open(fileobj=stream,mode="r|") as tar:
            for m in tar:
                parts=Path(m.name).parts
                if len(parts)<2 or parts[0] not in allowed or parts[1]!=role or ".." in parts or not (m.isfile() or m.isdir()):
                    raise ValueError("Invalid fresh member")
                tar.extract(m,tmp,filter="data")
        for fam in sorted(allowed):
            src=tmp/fam/role; meta=json.loads((src/"card.json").read_bytes())
            for key,pin in meta["sha256"].items():
                if not key.isidentifier():raise ValueError("Invalid field")
                with (src/(key+".npy")).open("rb") as f:
                    if hashlib.file_digest(f,"sha256").hexdigest()!=pin:raise ValueError("Fresh array mismatch")
            d=target/fam/role;d.parent.mkdir(exist_ok=True)
            if d.is_dir():
                if (d/"card.json").read_bytes()!=(src/"card.json").read_bytes():raise ValueError("Conflicting installed data")
            else:src.rename(d)
    (target/(".verified" if role=="train" else "."+role+"-verified")).write_text(expected+"\n")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("archive",type=Path);p.add_argument("sha");p.add_argument("root",type=Path);p.add_argument("role");a=p.parse_args();load(a.archive,a.sha,a.root,a.role)
