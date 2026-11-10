"""Fetch a bounded status through the provider notebook channel while transfers use the file API."""
import json
from pathlib import Path
root=Path("/content/g9-out")
def read(p):
    return json.loads(p.read_bytes()) if p.exists() else {}
rows=[]
for p in sorted([*root.glob("confirm-*/step-*/receipt.json"),*root.glob("exposure-*/step-*/receipt.json")]):
    d=read(p)
    if d["step"] in (1024,4096):rows.append([d["tag"],d["step"],d["inference_sha256"]])
log=root/"lane.log"
print("SLWM_G9 "+json.dumps(dict(progress=read(root/"progress.json"),completed=rows,
                               tail=log.read_text(errors="replace")[-420:] if log.exists() else "")))
