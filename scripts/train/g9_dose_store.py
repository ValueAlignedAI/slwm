"""Preserve the separate long-replay terminal generations and their new-cohort results."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from g7_inputs import FED
from g8_file_transfer import file_data
from g9_cloud_archive import archive
from g9_storage import write

def main(ctl):
    name="slwm-g9-boreal";saved=ctl/"dose-receipts";saved.mkdir(exist_ok=True)
    expected={f"exposure-{c}-z512-gelu-s{s}-{kind}" for c in ("hybrid","gpt2") for s in (911,912) for kind in ("fresh","replay")}
    while time.time()<json.loads((ctl/"bounds.json").read_text())["deadline"]:
        try:
            doc=json.loads(file_data(name,"/content/g9-dose/paired.json",2**21))
            if not set(doc)<=expected:raise ValueError("Unknown independent-dose run")
            for tag,item in doc.items():
                dest=saved/(tag+".json")
                if tag.endswith("-replay") and not dest.exists():
                    info=archive(name,tag,4096,item["sha256"],source_root="/content/g9-dose")
                    if info["optimizer_sha256"]!=item["optimizer_sha256"]:raise ValueError("Independent-dose optimizer mismatch")
                    write(dest,info)
            if set(doc)==expected and len(list(saved.glob("*.json")))==4:
                terminal=json.loads(file_data(name,"/content/g9-dose/finished.json",2**21))
                if terminal!=doc:raise ValueError("Independent-dose score changed at publication")
                payload=json.dumps(doc).encode()
                command="cd /home/db/dev/slwm-g9 && python3 -c \"import json,sys;sys.path.insert(0,'scripts/train');from g9_storage import disks,write;d=json.load(sys.stdin);[write(root.parent/'dose'/'scores.json',d) for root in disks()]\""
                subprocess.run([*FED,command],input=payload,check=True,timeout=90)
                write(ctl/"dose-verified.json",dict(models=4,scored_parents=8));return
        except (OSError,ValueError,StopIteration,RuntimeError,subprocess.SubprocessError) as exc:
            write(ctl/"dose-transfer.json",dict(state="retry",error=str(exc)[:180]))
        time.sleep(12)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("control",type=Path);main(p.parse_args().control)
