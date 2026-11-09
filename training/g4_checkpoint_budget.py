"""Reserve transfer time for outstanding full optimizer checkpoints on a session with a hard expiry."""
import json
import os
from pathlib import Path
import time


def backlog(root,spool):
    pending=0
    for endpoint in root.glob("g4-*/complete.json"):
        run=endpoint.parent
        record=json.loads(endpoint.read_bytes())
        digest=record["weight_file_sha256"]
        verified=root/"full-optimizer-receipts"/(run.name+".json")
        if verified.is_file() and json.loads(verified.read_bytes()).get("sha256")==digest:
            continue
        markers=(spool/"receipts"/run.name).glob(f"{record['step']:012d}-*.json")
        if any(json.loads(m.read_bytes())["files"].get("resume.npz",{}).get("sha256")==digest for m in markers):
            continue
        checkpoint=run/"resume.npz"
        if checkpoint.is_file(): pending+=checkpoint.stat().st_size
    return pending


def admit(root,spool,deadline,estimated_seconds,params,status):
    """A completed inference export alone cannot release the optimizer-backup reserve."""
    speed=float(os.environ.get("SLWM_G4_UPLOAD_MIB_S","6"))*1024**2
    cap=float(os.environ.get("SLWM_G4_PENDING_GIB","2"))*1024**3
    if speed<=0 or cap<=0: raise ValueError("Positive storage budget required")
    next_bytes=12*params+1024**2
    while True:
        outstanding=backlog(root,spool)
        reserve=estimated_seconds+45+(outstanding+next_bytes)/speed
        if time.time()+reserve>deadline: return False
        if not outstanding or outstanding+next_bytes<=cap: return True
        tmp=status.with_suffix(".tmp")
        tmp.write_text(json.dumps({"stage":"verifying_full_optimizers","pending_bytes":outstanding,
                                   "estimated_next_bytes":next_bytes,"checked_unix":time.time()})+"\n")
        tmp.replace(status)
        time.sleep(10)


def parameters(width,depth,expansion=4,groups=1):
    # Exact count for the G4 per-tick encoders, clocks, output matrices, norms and token bigram.
    edges=width*(384+259+4+8+768+128+2)+259*259
    return edges+depth*(4*width*width+2*expansion*width*width//groups+4*width)
