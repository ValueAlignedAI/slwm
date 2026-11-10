"""Download only the registered G5 role, verify its frozen archive hashes, and extract regular files."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tarfile
import tempfile

import zstandard


def fetch(root,role,pins):
    root.mkdir(parents=True,exist_ok=True)
    manifest=(pins/(role+".sha256")).read_text().splitlines()
    entries={line.split()[1]:line.split()[0] for line in manifest}
    for family in ("paired","impact","pusht"):
        name=f"{family}-{role}.tar.zst"
        expected=entries[name]
        target=root/family/role
        if (target/"card.json").is_file():
            continue
        target.parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as d:
            d=Path(d)
            local=d/name
            subprocess.run(["rclone","copyto","--multi-thread-streams","8",f"slwm-drive:SLWM/g5/data/{name}",local],check=True)
            with local.open("rb") as f:
                if hashlib.file_digest(f,"sha256").hexdigest()!=expected:
                    raise ValueError(f"Cloud data changed: {name}")
            with local.open("rb") as f,zstandard.ZstdDecompressor().stream_reader(f) as reader:
                with tarfile.open(fileobj=reader,mode="r|") as tar:
                    for member in tar:
                        parts=Path(member.name).parts
                        if parts[:1]!=(role,) or ".." in parts or not (member.isfile() or member.isdir()):
                            raise ValueError("Unexpected archive entry")
                        tar.extract(member,d,filter="data")
            (d/role).rename(target)
        print(f"verified {name}",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("role",choices=["train","dev","test"]); p.add_argument("--root",type=Path,required=True)
    p.add_argument("--pins",type=Path,required=True)
    a=p.parse_args(); fetch(a.root,a.role,a.pins)
