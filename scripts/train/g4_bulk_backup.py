"""Upload immutable final endpoints in parallel, commit one checked receipt per run, and keep working during training."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time


def sha(path):
    with path.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()


def send(run,remote,receipts):
    endpoint=json.loads((run/"complete.json").read_text())
    binary=run/"resume.npz"
    expected=endpoint["weight_file_sha256"]
    if sha(binary)!=expected: raise ValueError(f"Changed endpoint: {run.name}")
    # Keep a second independent per-run path. The regular vault handles intermediate generations.
    prefix=f"{remote}/{run.name}"
    def put(path,dest,digest):
        subprocess.run(["rclone","copyto","--immutable","--checksum","--drive-chunk-size","64M",path,dest],check=True)
        actual=subprocess.check_output(["rclone","hashsum","SHA-256",dest],text=True).split()[0]
        if actual!=digest: raise ValueError(f"Remote digest failed: {run.name}")
    put(binary,prefix+"/resume.npz",expected)
    card={"run":run.name,"step":endpoint["step"],"sha256":expected,"bytes":binary.stat().st_size,
          "config":json.loads((run/"config.json").read_text()),"complete":endpoint}
    with tempfile.TemporaryDirectory(dir=receipts) as tmp:
        marker=Path(tmp)/"commit.json"; marker.write_text(json.dumps(card,sort_keys=True)+"\n")
        put(marker,prefix+"/commit.json",sha(marker))
        marker.replace(receipts/(run.name+".json"))
    print("verified full optimizer",run.name,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True); p.add_argument("--until",type=float,required=True)
    p.add_argument("--remote",default="slwm-drive:SLWM/g4/optimizer")
    a=p.parse_args(); receipts=a.root/"full-optimizer-receipts"; receipts.mkdir(exist_ok=True)
    while time.time()<a.until:
        pending=[p.parent for p in a.root.glob("g4-*/complete.json") if not (receipts/(p.parent.name+".json")).is_file()]
        pending.sort(key=lambda r:("anchor" not in r.name,"medium" not in r.name,r.name))
        if pending:
            with ThreadPoolExecutor(max_workers=4) as pool:
                jobs={pool.submit(send,r,a.remote,receipts):r for r in pending}
                for task,r in jobs.items():
                    try: task.result()
                    except (ValueError,OSError,subprocess.CalledProcessError) as e: print("pending",r.name,str(e)[:150],flush=True)
        time.sleep(12)


if __name__=="__main__": main()
