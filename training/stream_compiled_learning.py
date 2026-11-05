"""Short paired eager/compiler learning controls, independent of primary B1."""
import copy
import fcntl
import json
from pathlib import Path
import shutil

import numpy as np
import torch

import training.broad_stream_train as training
from training.broad_batch import StreamCorpus
from training.research_sources import file_sha
from training.t2_real_av import write_json

CONFIG="configs/t2/compiled_stream_pilot.json"


def main():
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
    pilot=json.loads(Path(CONFIG).read_text()); cfg=copy.deepcopy(json.loads(Path(pilot["parent_config"]).read_text()))
    parent=Path(pilot["parent_artifacts"]); root=Path(pilot["artifacts"]); root.mkdir(parents=True,exist_ok=True)
    sources=json.loads((parent/"source_hashes.json").read_text())
    names=list(sources)+[CONFIG,pilot["protocol"],"training/stream_compiled_learning.py"]
    for name in names:
        if name in sources: assert file_sha(name)==sources[name],name
        target=root/"source_snapshot"/name; target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists(): assert file_sha(name)==file_sha(target),name
        else: shutil.copyfile(name,target)
        sources[name]=file_sha(name)
    write_json(root/"source_hashes.json",sources)
    cfg.update(sprint=pilot["sprint"],protocol=pilot["protocol"],artifacts=str(root),runs=pilot["runs"])
    cfg["training"]["steps"]=pilot["steps"]
    constructor=training.construct
    def runtime_construct(config,run):
        model=constructor(config,run)
        if run["compiled"]: model.process=torch.compile(model.process,fullgraph=True,dynamic=False,mode=pilot["compile_mode"])
        return model
    training.construct=runtime_construct
    data=StreamCorpus(cfg,"train"); val=StreamCorpus(cfg,"validation")
    plans=json.loads((parent/"selection_plans.json").read_text()); write_json(root/"selection_plans.json",plans)
    for run in cfg["runs"]:
        out=root/run["id"]; out.mkdir(exist_ok=True)
        if not(out/"registry.json").exists():
            write_json(out/"registry.json",{"experiment_id":run["id"],"status":"registered","config":pilot,
                "source_hashes":sources,"scope":"400-update auxiliary runtime learning check; original train/selection people only",
                "fresh_test_read":False,"same_schedule_and_data_as_internal_eager_control":True})
    while any(not(root/r["id"]/"complete.json").exists() for r in cfg["runs"]):
        for run in cfg["runs"]:
            if (root/run["id"]/"complete.json").exists(): continue
            with Path("/tmp/slwm_research_gpu.lock").open("a") as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                training.train_slice(cfg,run,data,val,plans,sources)
    choices={r["id"]:json.loads((root/r["id"]/"complete.json").read_text()) for r in cfg["runs"]}
    write_json(root/"selection.json",choices); comparisons={}
    for run in cfg["runs"][:2]:
        name=run["id"]; reference=run["eager_reference"]
        a=json.loads((root/name/"curves.json").read_text())[-1]
        b=json.loads((root/reference/"curves.json").read_text())[-1]
        assert a["step"]==b["step"]==pilot["steps"]
        ck=torch.load(root/name/"resume.pt",map_location="cpu",weights_only=False)
        ref=torch.load(root/reference/"resume.pt",map_location="cpu",weights_only=False)
        assert ck["data_rng"]==ref["data_rng"]
        ratio=a["validation"]["selection_loss"]/b["validation"]["selection_loss"]
        comparisons[run["variant"]]={"compiled_id":name,"eager_id":reference,
            "compiled_endpoint":a,"eager_endpoint":b,"relative_validation_loss_change":ratio-1,
            "clears_exploratory_learning_check":ratio<=1+pilot["relative_validation_loss_tolerance"],
            "cold_inclusive_speedup":b["seconds"]/a["seconds"],"exact_data_rng_matches":True,
            "fresh_test_read":False}; del ck,ref
    write_json(root/"summary.json",comparisons)
    lines=["# Compiled stream learning pilot","","Single-seed,400-update validation-only runtime study. Eager and compiled arms have identical400-update LR horizons, initialization, data and selection. The original2000-step B1 matrix is unchanged. No fresh test speaker is read.","","| Variant | Eager loss | Compiled loss | Relative change | Cold-inclusive speedup | Pilot check |","|---|---:|---:|---:|---:|---|"]
    for variant,row in comparisons.items():
        lines.append(f"| {variant} | {row['eager_endpoint']['validation']['selection_loss']:.6f} | {row['compiled_endpoint']['validation']['selection_loss']:.6f} | {100*row['relative_validation_loss_change']:+.3f}% | {row['cold_inclusive_speedup']:.3f}x | {'PASS' if row['clears_exploratory_learning_check'] else 'FAIL'} |")
    lines += ["","The pilot gate is no more than1% loss degradation at the fixed400 endpoint. It is exploratory evidence about short learning, not fresh-person equivalence, convergence or additional primary capability evidence. Full best/resume/final states and every component curve remain available. First-gradient and steady-step measurements, including the failed original SLWM local loss gate, remain in the separate compilation benchmark. Future adoption needs a registered longer comparison and cold-cost accounting."]
    (root/"report.md").write_text("\n".join(lines)+"\n")
    for run in cfg["runs"]:
        path=root/run["id"]/"registry.json"; entry=json.loads(path.read_text()); entry.update(status="completed",completion=choices[run["id"]],fresh_test_read=False); write_json(path,entry)
    print(json.dumps({"compiled_learning_pilot_complete":True,"comparisons":comparisons}),flush=True)


if __name__=="__main__": main()
