"""Pack the original layout of G3's prepared signal features on a dedicated data disk."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

if __package__:
    from .colab_artifacts import capture, encode, mounted_replica
else:
    from colab_artifacts import capture, encode, mounted_replica


CACHE = Path("artifacts/general_world/diverse_v1/prepared")
G2 = Path("artifacts/predictive_world/g2")
RUN = "g3-prepared-signals-v1"


def inventory(repo):
    files = set()
    allowed = (CACHE, G2, Path("artifacts/t2_audio_visual/broad_stream_prepared_v1"))
    resolved = tuple((repo/p).resolve() for p in allowed)
    def add(path):
        if path.is_absolute() or ".." in path.parts or not any(path.is_relative_to(p) for p in allowed):
            raise ValueError(f"Unexpected cache path: {path}")
        if str(path) in files:
            return
        local = repo/path
        if not local.is_file() or not any(local.resolve().is_relative_to(p) for p in resolved):
            raise ValueError(f"Expected a regular prepared file: {path}")
        files.add(str(path))
    def rows(path):
        add(path)
        return json.loads((repo/path).read_bytes())
    for name in ["normalizers.npz", "dataset_contract.json"]:
        add(G2/name)
    for family in ["coco","speech","ami","clotho","pamap2"]:
        for row in rows(CACHE/family/"rows.json"):
            if family == "coco":
                add(CACHE/family/row["array"])
            elif family in ("speech","ami"):
                add(Path(row["mel_file"]))
                if family == "ami":
                    add(Path(row["pixel_file"]))
            else:
                add(CACHE/family/row["file"])
    for row in rows(G2/"robot_rows.json"):
        for name in ("trajectory.npz","rgb.npy"):
            add(CACHE/"droid_g2"/row["id"]/name)
    return {name:(repo/name).stat().st_size for name in sorted(files)}


def pack(repo, spool, device, describe=False):
    mounted_replica(spool, device)
    files = inventory(repo)
    total = sum(files.values())
    print(f"{len(files)} prepared files, {total} bytes before compression", flush=True)
    if describe:
        return
    if any((spool/"receipts"/RUN).glob("000000000000-*.json")) or (spool/"pending"/RUN/"000000000000").exists():
        print("Generation already pending or committed", flush=True)
        return
    if shutil.disk_usage(spool).free < total*1.05 + 16*1024**3:
        raise OSError("Insufficient data-disk space for the bounded packing job")
    with tempfile.TemporaryDirectory(dir=spool,prefix=".pack-") as tmp:
        temp = Path(tmp)
        listing, archive, manifest = temp/"paths", temp/"g3-prepared.tar.zst", temp/"inventory.json"
        listing.write_bytes(b"".join(name.encode()+b"\0" for name in files))
        subprocess.run(["tar","--create","--dereference","--file",str(archive),"--use-compress-program","zstd -T2 -1 -q",
                        "--format=posix","--pax-option=delete=atime,delete=ctime","--mtime=@0",
                        "--owner=0","--group=0","--numeric-owner","--no-recursion","--null",
                        "--files-from",str(listing)], cwd=repo, check=True)
        manifest.write_bytes(encode({"schema":1,"layout":"paths relative to the SLWM checkout",
                                     "files":files,"source_bytes":total,"archive_bytes":archive.stat().st_size,
                                     "dataset_contract_sha256":hashlib.sha256((repo/G2/"dataset_contract.json").read_bytes()).hexdigest()}))
        capture(spool,RUN,0,{"g3-prepared.tar.zst":archive,"inventory.json":manifest})
        print(f"staged {archive.stat().st_size} compressed bytes",flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--spool", type=Path, required=True)
    p.add_argument("--uuid", required=True)
    p.add_argument("--describe", action="store_true")
    a = p.parse_args()
    pack(a.repo.resolve(),a.spool.resolve(),a.uuid,a.describe)
