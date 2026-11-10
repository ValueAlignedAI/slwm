"""Download only the registered G5 role, verify its frozen archive hashes, and extract regular files."""
import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
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
            chunks=pins/(role+"-blocks.json")
            landing=Path("/content/g5-landing")/name
            if landing.is_file():
                os.link(landing,local)
            elif chunks.exists():
                entry=json.loads(chunks.read_bytes())[name]
                if entry["sha256"]!=expected:
                    raise ValueError("G5 manifest disagreement")
                def download(item):
                    ix,sha=item
                    if len(sha)!=64 or any(c not in "0123456789abcdef" for c in sha):
                        raise ValueError("Invalid object hash")
                    part=d/f"part-{ix}"
                    subprocess.run(["rclone","copyto",f"slwm-drive:SLWM/g5/data-blocks/objects/{sha}",part],check=True)
                    with part.open("rb") as reader:
                        if hashlib.file_digest(reader,"sha256").hexdigest()!=sha:
                            raise ValueError("Corrupt G5 transfer object")
                    return part
                import shutil
                with ThreadPoolExecutor(max_workers=5) as io,local.open("wb") as dest:
                    for part in io.map(download,enumerate(entry["parts"])):
                        with part.open("rb") as reader:
                            shutil.copyfileobj(reader,dest,1024**2)
                        part.unlink()
                if local.stat().st_size!=entry["bytes"]:
                    raise ValueError("Truncated G5 archive")
            else:
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
