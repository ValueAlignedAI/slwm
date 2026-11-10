"""Read small G7 training, evaluation and archive metadata through the native notebook channel."""
import json
from pathlib import Path
import shutil

root=Path("/content/g7-out")
def load(p):
    try: return json.loads(p.read_bytes())
    except (FileNotFoundError,json.JSONDecodeError): return {}
def tail(p):
    if not p.exists(): return ""
    with p.open("rb") as f:
        f.seek(max(0,p.stat().st_size-1000)); return f.read().decode(errors="replace")[-600:]
lane=load(root/"lane.json")
rungs=[]
for p in sorted(root.glob("g7-*/step-*/complete.json")):
    d=load(p); m=load(p.parent/"dev-metrics.json")
    if not d: continue
    rungs.append({"run":d["run"],"step":d["step"],"sha":d["sha256"]["resume.npz"],
                  "model":d["sha256"]["inference.npz"],"evaluated":bool(m) and (p.parent/"wall.json").is_file(),
                  "action":m.get("push",{}).get("joint",{}).get("endpoint_px"),
                  "contact":m.get("impact",{}).get("average_precision",{}).get("correct")})
progress={p.parent.name:load(p) for p in root.glob("g7-*/progress.json")}
benches={p.stem:load(p).get("warm_examples_per_second") for p in root.glob("bench*.json")}
report={"lane":lane,"progress":progress,"rungs":rungs,"benchmarks":benches,"ram_inputs":Path("/content/g7-data/.verified").is_file(),
        "free_gib":round(shutil.disk_usage("/content").free/2**30,1)}
if lane.get("phase")=="failed":
    logs=sorted(root.glob("*.log"),key=lambda v:v.stat().st_mtime)
    report["diagnostic"]=tail(logs[-1]) if logs else tail(Path("/content/g7-lane.log"))
print("SLWM_G7_STATUS "+json.dumps(report,sort_keys=True))
