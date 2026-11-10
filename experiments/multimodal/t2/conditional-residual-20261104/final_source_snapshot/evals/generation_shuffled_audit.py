"""Correct an inference-only flag in the frozen G1 shuffled-training control.

The original training function applies `shuffled` at validation/inference too.
Keep all original artifacts/selection and score the fixed checkpoint with correct
test audio. This addendum does not retrain or reselect any model.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from training.t2_generation_train import load_data,make_model,evaluate,predict
from training.t2_prepare_cremad import sha256
from training.t2_real_av import write_json
from exploration.motion_outputs import save_video


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text()); root=Path(cfg["artifacts"]); run=next(r for r in cfg["runs"] if r["id"]=="EXP-T2-415")
    inference_run={**run,"shuffled":False}; checkpoint=torch.load(root/run["id"]/"best.pt",map_location="cpu",weights_only=False)
    torch.set_num_threads(4); model=make_model(run).cuda().eval(); model.load_state_dict(checkpoint["model"])
    gallery=json.loads((root/"exploration/gallery.json").read_text()); fixes=[]
    for split in ("validation","test","external"):
        data=load_data(Path(cfg["corpus"]),split); result=evaluate(model,inference_run,data,32,True)
        result.update({"run":run,"inference_run":inference_run,"checkpoint_sha256":sha256(root/run["id"]/"best.pt"),"selected_step":checkpoint["step"],
            "scope":"evaluation correction only: shuffled TRAINING, correct/wrong/null inputs at inference; original selection retained",
            "original_metrics_preserved":str(root/run["id"]/(split+"_metrics.json")),"original_metrics_sha256":sha256(root/run["id"]/(split+"_metrics.json"))})
        write_json(root/run["id"]/("corrected_"+split+"_metrics.json"),result)
        for example in (e for e in gallery["visual_examples"] if e["split"]==split):
            index=next(i for i,r in enumerate(data["rows"]) if r["sample_id"]==example["sample_id"])
            with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
                output,_=predict(model,inference_run,data,np.array([index]),np.array([example["start_frame"]]),32)
            path=Path(example["folder"])/"EXP-T2-415_correct_audio.gif"
            save_video(path,output[0].float().cpu().numpy(),"SHUFFLED TRAINING; correct audio + seed")
            fixes.append({"sample_id":example["sample_id"],"corrected_media":str(path),"original_media_input":"wrong audio (training flag retained at inference)"})
        del data; torch.cuda.empty_cache()
    write_json(root/"evaluation_correction.json",{"source_sha256":sha256(Path(__file__)),"corrected_run":"EXP-T2-415","no_retraining_or_reselection":True,
        "original_artifacts_preserved":True,"selection_note":"original checkpoint selected using shuffled validation inputs; no corrected-validation reselection",
        "reason":"Training-only shuffled flag was applied at evaluation. Correct-input conditions must not automatically shuffle.","media":fixes})


if __name__=="__main__": main()
