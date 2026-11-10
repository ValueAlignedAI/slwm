"""Place byte-identical train/dev arrays in Colab shared memory, then atomically switch future opens."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def digest_file(src,dst,expected):
    hasher=hashlib.sha256()
    with src.open("rb") as reader,dst.open("wb") as writer:
        while chunk:=reader.read(8*1024**2):
            hasher.update(chunk); writer.write(chunk)
    if hasher.hexdigest()!=expected:
        raise ValueError(f"Data checksum mismatch: {src.name}")


def stage(root):
    if root.resolve()!=root:
        if str(root.resolve()).startswith("/dev/shm/slwm-g6-inputs-") and (root/".verified").is_file():
            print("Verified input cache already active",flush=True); return
        raise ValueError("Unexpected dataset link")
    shm=Path("/dev/shm")
    sources={}
    for family in ("paired","impact","pusht"):
        for role in ("train","dev"):
            card=root/family/role/"card.json"; raw=card.read_bytes()
            sources[family,role]=(raw,json.loads(raw)["sha256"])
    footprint=sum((root/f/r/(k+".npy")).stat().st_size for (f,r),(_,table) in sources.items() for k in table)
    available=next(int(v.split()[1])*1024 for v in Path("/proc/meminfo").read_text().splitlines() if v.startswith("MemAvailable:"))
    if available<2*footprint+6*1024**3 or shutil.disk_usage(shm).free<footprint+1024**3:
        print("Host memory reserve reached; existing data path retained",flush=True); return
    fingerprint=hashlib.sha256(b"".join(raw for raw,_ in sources.values())).hexdigest()
    final=shm/("slwm-g6-inputs-"+fingerprint[:16])
    if not (final/".verified").is_file():
        with tempfile.TemporaryDirectory(dir=shm,prefix="slwm-g6-build-") as t:
            new=Path(t)/"data"; new.mkdir()
            for (family,role),(raw,table) in sources.items():
                dest=new/family/role; dest.mkdir(parents=True)
                for k,expected in table.items():
                    if not k.isidentifier():
                        raise ValueError("Invalid immutable array field")
                    digest_file(root/family/role/(k+".npy"),dest/(k+".npy"),expected)
                (dest/"card.json").write_bytes(raw)
            prior=json.loads(Path("configs/t6/tpu_scale.json").read_bytes())["prior_sha256"]
            digest_file(root/"prior.npz",new/"prior.npz",prior)
            (new/".verified").write_text(fingerprint+"\n")
            new.rename(final)
    if (final/".verified").read_text().strip()!=fingerprint:
        raise ValueError("Cache identity changed")
    # Both names live on the same filesystem; open file descriptors stay valid across the exchange.
    link=root.with_name(root.name+"-ssd")
    if link.exists() or link.is_symlink():
        raise FileExistsError(link)
    link.symlink_to(final,target_is_directory=True)
    libc=ctypes.CDLL(None,use_errno=True)
    result=libc.renameat2(-100,os.fsencode(link),-100,os.fsencode(root),2)
    if result!=0:
        error=ctypes.get_errno(); link.unlink()
        raise OSError(error,os.strerror(error))
    print(json.dumps({"verified_ram_bytes":footprint,"identity":fingerprint,"ssd_fallback":str(link)}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("data",type=Path); a=p.parse_args(); stage(a.data)
