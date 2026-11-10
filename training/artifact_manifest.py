"""Streaming content manifests and byte-for-byte verification of research mirrors."""
import argparse
from datetime import datetime,timezone
import fnmatch
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    path=Path(path); before=path.stat(); hasher=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024**2),b""): hasher.update(block)
    after=path.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
        raise ValueError("File changed during hashing: "+str(path))
    return {"bytes":after.st_size,"sha256":hasher.hexdigest()}


def create_manifest(base,roots,output,pattern="*"):
    base=Path(base).resolve(); output=Path(output).resolve(); files={}
    for root in roots:
        folder=base/root
        if not folder.is_dir(): raise ValueError("Missing artifact root: "+str(folder))
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.resolve()==output or not fnmatch.fnmatch(path.name,pattern): continue
            if path.is_symlink(): raise ValueError("Artifact symlink needs explicit resolution: "+str(path))
            files[path.relative_to(base).as_posix()]=digest(path)
    if not files: raise ValueError("Refusing an empty artifact manifest")
    manifest={"format":1,"created_utc":datetime.now(timezone.utc).isoformat(),"scope_roots":list(roots),"filename_pattern":pattern,
              "paths_relative_to":"repository root on each machine","files":files,"file_count":len(files),"bytes":sum(row["bytes"] for row in files.values())}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    return manifest


def verify_manifest(base,manifest):
    base=Path(base); total=0
    for name,expected in manifest["files"].items():
        path=Path(name)
        if path.is_absolute() or ".." in path.parts: raise ValueError("Manifest path escapes the repository")
        actual=digest(base/path)
        if actual!=expected: raise ValueError("Artifact bytes differ: "+name)
        total+=actual["bytes"]
    return {"all_recorded_files_match":True,"verified_files":len(manifest["files"]),"verified_bytes":total}


def main():
    p=argparse.ArgumentParser(); p.add_argument("stage",choices=("create","verify")); p.add_argument("--root",action="append",default=[]); p.add_argument("--manifest",required=True); p.add_argument("--include",default="*"); p.add_argument("--result"); a=p.parse_args(); began=time.perf_counter()
    if a.stage=="create":
        value=create_manifest(Path.cwd(),a.root,a.manifest,a.include); result={"created":True,"files":value["file_count"],"bytes":value["bytes"]}
    else:
        value=json.loads(Path(a.manifest).read_text()); result=verify_manifest(Path.cwd(),value); result["source_manifest_sha256"]=digest(a.manifest)["sha256"]
    result["seconds"]=time.perf_counter()-began
    if a.result: Path(a.result).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__": main()
