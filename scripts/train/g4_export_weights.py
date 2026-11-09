"""Export full-precision inference weights and provenance without materializing optimizer moments."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


def sha(path):
    with path.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()


def main():
    q=argparse.ArgumentParser(description=__doc__); q.add_argument("--root",type=Path,required=True)
    q.add_argument("--prefix",required=True); q.add_argument("--remote",default="slwm-drive:SLWM/g4/weights")
    args=q.parse_args()
    runs=sorted(p.parent for p in args.root.glob(args.prefix+"-*/complete.json"))
    if not runs: raise ValueError("No complete endpoints to export")
    with tempfile.TemporaryDirectory(dir=args.root,prefix=".inference-") as tmp:
        staging=Path(tmp); export=staging/"weights"; export.mkdir()
        for run in runs:
            dest=export/run.name; dest.mkdir()
            cfg=json.loads((run/"complete.json").read_text())
            if sha(run/"resume.npz")!=cfg["weight_file_sha256"]: raise ValueError("Source checkpoint changed")
            with np.load(run/"resume.npz",allow_pickle=False) as record:
                np.savez(dest/"model.npz",**{k:record[k] for k in record.files if k.startswith("p/") or k in ("step","identity","stats")})
            cfg["inference_sha256"]=sha(dest/"model.npz")
            (dest/"complete.json").write_text(json.dumps(cfg,sort_keys=True)+"\n")
            for file in ("config.json","dev.json","timing.json"):
                if (run/file).exists(): shutil.copy2(run/file,dest/file)
        capsule=staging/(args.prefix+".tar")
        subprocess.run(["tar","-cf",capsule,"-C",export,"."],check=True)
        digest=sha(capsule)
        dest=f"{args.remote}/{args.prefix}-{digest}.tar"
        subprocess.run(["rclone","copyto","--checksum","--immutable",capsule,dest],check=True)
        receipt=subprocess.check_output(["rclone","hashsum","SHA-256",dest],text=True).split()[0]
        if receipt!=digest: raise ValueError("Remote inference archive checksum mismatch")
        card={"group":args.prefix,"runs":[r.name for r in runs],"sha256":digest,"bytes":capsule.stat().st_size,"object":dest}
        marker=args.root/(args.prefix+"-inference.json")
        marker.write_text(json.dumps(card,sort_keys=True)+"\n")
        subprocess.run(["rclone","copyto","--checksum","--immutable",marker,f"{args.remote}/{marker.name}"],check=True)
        print(json.dumps(card),flush=True)


if __name__=="__main__": main()
