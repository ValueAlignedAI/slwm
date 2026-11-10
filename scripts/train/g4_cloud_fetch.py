"""Fetch and verify a registered data role directly on the accelerator machine."""
import argparse
import fcntl
import hashlib
from pathlib import Path
import subprocess
import tarfile
import tempfile

import zstandard


PINS={"dev":"943dba8f905de033f97ce19b7bfedd2b1e7fbfca1abe631f8c7ea65b5cfa8a24",
      "train":"591dae7d7bc880001580d91cd9e12c9fbb912ae3097bfb4d1302ec7fd5a9870b",
      "test":"6897c7a549e5e56a7daef632ea4e902ed48410138cfec70d09390726288ec2b0"}


def fetch(root, role):
    root.mkdir(parents=True,exist_ok=True)
    with (root/(role+".lock")).open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        dest=root/role
        if (dest/"card.json").is_file(): return
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            folder=Path(tmp)
            file=folder/"pack.zst"
            subprocess.run(["rclone","copyto","--multi-thread-streams","6",f"slwm-drive:SLWM/g4/paired-v1/{role}.tar.zst",file],check=True)
            with file.open("rb") as stream:
                if hashlib.file_digest(stream,"sha256").hexdigest()!=PINS[role]:
                    raise ValueError("TPU data archive hash mismatch")
            with file.open("rb") as stream, zstandard.ZstdDecompressor().stream_reader(stream) as raw:
                with tarfile.open(fileobj=raw,mode="r|") as archive:
                    for item in archive:
                        if Path(item.name).parts[:1]!=(role,) or not (item.isfile() or item.isdir()):
                            raise ValueError(f"Unexpected archive member: {item.name}")
                        archive.extract(item,folder,filter="data")
            (folder/role).rename(dest)
        print(f"verified data role {role}",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("role",choices=PINS); p.add_argument("--root",type=Path,default=Path("/content/g4data"))
    a=p.parse_args(); fetch(a.root,a.role)
