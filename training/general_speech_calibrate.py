"""Reserve-role decoder/timing calibration after all G1 weights are frozen."""
import fcntl
import gc
import json
from pathlib import Path

import numpy as np
import torch

from training.general_batch import CACHE
from training.general_sources import write
from training.general_train import initialize
from training.general_speech_media import SpeechRenderer
from training.general_speech_score import Recognizer,aggregate
from training.research_sources import file_sha


def recipes():
    cfg=json.loads(Path("configs/t2/general_transfer.json").read_text())
    recipe=json.loads(Path("configs/t2/general_speech_eval.json").read_text())
    root=Path(cfg["root"])/"speech";root.mkdir(exist_ok=True)
    assert (Path(cfg["root"])/"training_complete.json").exists()
    assert (Path(cfg["root"])/"readouts/training_complete.json").exists()
    return cfg,recipe,root


def case_rows(family,role,secondary=False):
    manifest=json.loads((CACHE/"evaluation_cases.json").read_text())
    cases=[r for r in manifest["cases"][family] if r["role"]==role]
    rows={r["id"]:r for r in json.loads((CACHE/family/"rows.json").read_text()) if r["role"]==role}
    result=[(rows[c["id"]],c) for c in cases]
    if secondary and family=="speech" and role=="test":
        from training.general_data import order
        recipe=json.loads(Path("configs/t2/general_speech_eval.json").read_text())
        for person in sorted({r["person"] for r in cases}):
            selected=sorted([r for r in cases if r["person"]==person],key=lambda r:order(r["id"]+"secondary-gap"))[:recipe["secondary_speech_clips_per_person"]]
            result.extend((rows[c["id"]],{**c,"gap_seconds":gap}) for c in selected for gap in recipe["secondary_gap_seconds"])
    return result


def case_name(row,case):
    return row["id"]+"_"+str(round(case.get("start_seconds",0)*1000))+"_g"+str(round(case.get("gap_seconds",.16)*1000))


def calibrate():
    torch.set_num_threads(4);cfg,recipe,root=recipes();result={}
    from training.general_evaluation_lock import register
    register(cfg)
    assert not (Path(cfg["root"])/"evaluation_lock.json").exists(),"Calibration rules are already frozen"
    for run in (r for r in cfg["runs"] if r["id"] in recipe["runs"]):
        folder=root/"calibration"/run["id"];folder.mkdir(parents=True,exist_ok=True)
        if (folder/"complete.json").exists():result[run["id"]]=json.loads((folder/"complete.json").read_text());continue
        checkpoint=Path(cfg["root"])/run["id"]/"endpoint.pt"
        with open("/tmp/slwm_research_gpu.lock","a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            saved=torch.load(checkpoint,map_location="cpu",weights_only=False,mmap=True);model=initialize(cfg,run).cuda().eval().requires_grad_(False);model.load_state_dict(saved["model"]);del saved
            renderer=SpeechRenderer(cfg);jobs=[]
            for row,case in case_rows("speech","calibration"):
                prepared=renderer.case(row,case);proposal=renderer.infer(model,prepared)
                candidates={"alpha"+str(alpha):(1-alpha)*proposal["direct"]+alpha*proposal["phonetic"] for alpha in recipe["alphas"]}
                files=renderer.save_many(prepared,candidates,folder/case_name(row,case))
                for name,media in files.items():jobs.append({"id":case_name(row,case),"person":row["actor"],"group":row["group"],"reference":prepared["reference"],"condition":name,**media})
            recognizer=Recognizer(recipe,root/"recognizer_cache.json");scored=recognizer.score(jobs);write(folder/"recognized.json",scored)
            values={alpha:aggregate([r for r in scored if r["condition"]=="alpha"+str(alpha)]) for alpha in recipe["alphas"]}
            alpha=min(recipe["alphas"],key=lambda a:(values[a]["wer"],a))
            del recognizer;torch.cuda.empty_cache()
            timing=[]
            for row,case in case_rows("ami","calibration"):
                if case["stratum"]!="own_nonoverlap":continue
                prepared=renderer.case(row,case)
                for condition in ["aligned",*["shift:"+str(n) for n in recipe["primary_shifts_packets"]]]:
                    proposal=renderer.infer(model,prepared,condition)
                    timing.append({"id":case_name(row,case),"person":row["actor"],"group":row["group"],"condition":condition,"probability":proposal["matching_probability"],"aligned":condition=="aligned"})
            thresholds={}
            for threshold in recipe["timing_thresholds"]:
                positive=[r["probability"]>=threshold for r in timing if r["aligned"]]
                negative=[r["probability"]<threshold for r in timing if not r["aligned"]]
                thresholds[threshold]=.5*(np.mean(positive)+np.mean(negative)) if positive and negative else .5
            threshold=max(recipe["timing_thresholds"],key=lambda t:(thresholds[t],t))
            write(folder/"timing_scores.json",timing)
            item={"run":run["id"],"alpha":alpha,"alpha_selection":{str(k):v for k,v in values.items()},"timing_threshold":threshold,
                  "timing_calibration_balanced_accuracy":float(thresholds[threshold]),"timing_threshold_curve":{str(k):float(v) for k,v in thresholds.items()},
                  "timing_calibration_groups":len({r["group"] for r in timing}),"core_checkpoint_sha256":file_sha(checkpoint),"recipe_sha256":file_sha("configs/t2/general_speech_eval.json"),"final_people_used":0}
            write(folder/"complete.json",item);result[run["id"]]=item
            del model,renderer;gc.collect();torch.cuda.empty_cache()
        print(json.dumps({"speech_calibrated":run["id"],"alpha":alpha,"timing_threshold":threshold}),flush=True)
    write(root/"calibration.json",{"runs":result,"all_rules_frozen":True,"fresh_final_people_used":0,"role":"calibration only"})


if __name__=="__main__":calibrate()
