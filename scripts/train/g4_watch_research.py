"""Read-only research-stage monitor. Record ETAs and changed milestones without requesting compute."""
import argparse
import json
from pathlib import Path
import time


def read(path):
    try:
        return json.loads(path.read_bytes())
    except FileNotFoundError:
        return {}


def snapshot(g3,g4):
    now=time.time(); status=read(g3/"status.json"); args=status.get("args",[])
    rows={p.parent.name:read(p) for p in g3.glob("*/progress.json")}
    active=None
    if status.get("phase")=="joint_train" and "--profile" in args and "--seed" in args:
        key=f'{args[args.index("--profile")+1]}-{args[args.index("--seed")+1]}'
        row=rows.get(key,{})
        if row.get("step",0)>0:
            age=now-(g3/key/"progress.json").stat().st_mtime
            active={"run":key,"step":row["step"],"target":row["target"],"last_progress_age_seconds":round(age,1),
                    "estimated_active_minutes_left":round(max(0,row["target"]-row["step"])*row["seconds"]/row["step"]/60,2),
                    "check_gpu_queue_or_worker":age>1800}
    remaining=sum(max(0,8000-r["step"])*r["seconds"]/r["step"] for r in rows.values() if r.get("step",0)>0)
    related={p.parent.name:read(p).get("phase") for p in (g3/"depth6/status.json",g3/"structured/status.json") if p.exists()}
    return {"checked_unix":now,"main_stage":status.get("phase","waiting_for_mount_or_status"),"active":active,
            "main_8000_endpoints":len(list(g3.glob("*/step-8000.json"))),"profiles_reporting":len(rows),
            "main_estimated_training_hours_left":round(remaining/3600,2) if rows else None,
            "main_heldout_summary_ready":(g3/"summary-8000-test.json").exists(),"other_stages":related,
            "g4_heldout_generation":read(g4/"confirmation-generation.json").get("stage"),
            "g4_heldout_transfer":read(g4/"confirmation-transfer.json").get("stage"),
            "fresh_speaker_audit_ready":(g4/"new-speaker-summary.json").exists(),
            "speech_content_audit_ready":(g4/"speech-content.json").exists()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("g3","g4","state"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--interval",type=int,default=30); p.add_argument("--once",action="store_true")
    a=p.parse_args(); a.state.parent.mkdir(parents=True,exist_ok=True); previous=None
    while True:
        result=snapshot(a.g3,a.g4)
        tmp=a.state.with_suffix(".tmp"); tmp.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n"); tmp.replace(a.state)
        active=result["active"] or {}
        key=(result["main_stage"],active.get("run"),active.get("step",0)//250,active.get("check_gpu_queue_or_worker"),
             result["main_8000_endpoints"],result["main_heldout_summary_ready"],str(result["other_stages"]),
             result["g4_heldout_generation"],result["g4_heldout_transfer"],result["fresh_speaker_audit_ready"])
        if key!=previous:
            print(json.dumps(result),flush=True); previous=key
        if a.once:return
        time.sleep(max(5,a.interval))


if __name__=="__main__": main()
