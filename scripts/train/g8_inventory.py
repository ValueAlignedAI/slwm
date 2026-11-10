"""Small native-kernel inventory, independent of the active SSH checkpoint transport."""
import json
import shutil
from pathlib import Path
root=Path("/content/g8-out")
def read(path):
    try:return json.loads(path.read_bytes())
    except (OSError,ValueError):return {}
lane=read(root/"lane.json");stages=[]
for file in sorted(root.glob("g8-*/step-*/complete.json")):
    x=read(file)
    if x:
        stages.append({"run":x["run"],"step":x["step"],"sha":x["sha256"]["resume.npz"],"weight":x["sha256"]["inference.npz"],
            "evaluated":(file.parent/"wall.json").is_file() and (file.parent/"dev-metrics.json").is_file()})
status={"lane":lane,"stages":stages,"bridges":[read(f) for f in sorted(root.glob("bridge-*.json"))],
    "registered_seeds":[int(p.name.split("-")[-1]) for p in sorted(root.glob("seed-*"))],"free_gib":round(shutil.disk_usage("/content").free/2**30,1),
    "progress":{p.parent.name:read(p) for p in root.glob("g8-*/progress.json")}}
if lane.get("phase")=="failed":
    logs=sorted(root.glob("*.log"),key=lambda p:p.stat().st_mtime)
    if logs:
        with logs[-1].open("rb") as f:
            f.seek(max(0,logs[-1].stat().st_size-1900));status["failure_log"]=f.read().decode(errors="replace")[-1400:]
print("SLWM_G8_STATUS "+json.dumps(status,sort_keys=True))
