"""Input-corruption diagnostics for synchronization, with no retraining.

These qualify a timing claim: the real audio must beat absent, wrong, or
time-reversed audio when every candidate video window is equally valid.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.slwm_motion import MotionSLWM
from training.t2_motion_train import load_data,crop,shuffled_indices
from training.t2_real_av import write_json,cluster_interval


def evaluate_conditions(model: MotionSLWM,data: dict,cfg: dict) -> dict:
    model.eval(); window=cfg["training"]["window"]; starts=(data["lengths"]-window)//2
    valid=np.where(data["lengths"]>=window+18)[0]; wrong=shuffled_indices(data["rows"])
    records={name:[] for name in ("original","zeroed_audio","same_person_wrong_audio","reversed_audio","envelope_only_audio")}
    with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        for begin in range(0,len(valid),32):
            ids=valid[begin:begin+32]; a,v=crop(data,ids,starts[ids],window)
            wa,_=crop(data,wrong[ids],starts[wrong[ids]],window)
            videos=[crop(data,ids,starts[ids]+offset,window)[1] for offset in cfg["evaluation"]["offsets"]]
            reps=[model.independent_video(vv) for vv in videos]
            envelope=torch.cat([a[...,:64].mean(-1,keepdim=True).expand(-1,-1,64),a[...,64:].mean(-1,keepdim=True).expand(-1,-1,64)],-1)
            visible=torch.ones(a.shape[:2],device=a.device,dtype=torch.bool)
            absent=torch.zeros_like(visible)
            for name,aa in (("original",a),("zeroed_audio",torch.zeros_like(a)),("same_person_wrong_audio",wa),
                            ("reversed_audio",a.flip(1)),("envelope_only_audio",envelope)):
                aw,_=model.encode(aa,torch.zeros_like(v),visible,absent,positions=False)
                ar=F.normalize(model.audio_alignment(aw).float(),dim=-1)
                scores=torch.stack([(ar*vr).sum(-1).mean(-1) for vr in reps],1)
                records[name].append((scores.argmax(1)==3).float().cpu().numpy())
    arrays={k:np.concatenate(v) for k,v in records.items()}; actors=[data["rows"][i]["actor"] for i in valid]
    original=arrays["original"]
    return {"eligible_clips":len(valid),"scope":"frozen input-corruption diagnostic; no model selection or threshold tuning",
            "conditions":{k:{"accuracy":float(v.mean()),"difference_from_original_95_interval":cluster_interval(original-v,actors)} for k,v in arrays.items()},
            "wrong_audio_definition":"A different clip from the same person; identical valid window length, no identity change",
            "causal_audio_contribution":all(cluster_interval(original-arrays[k],actors)[0]>0 for k in ("zeroed_audio","same_person_wrong_audio","reversed_audio"))}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text()); root=Path(cfg["artifacts"])
    torch.set_num_threads(4); data=load_data(Path(cfg["corpus"]),"test",torch.device("cuda"))
    results={}
    for run in cfg["runs"]:
        if run["arm"] not in {"signal_small","signal_large","vanilla_large","shuffled_pairs"}: continue
        folder=root/run["id"]; checkpoint=torch.load(folder/"best.pt",map_location="cuda",weights_only=False)
        model=MotionSLWM(run["model"]).cuda().eval(); model.load_state_dict(checkpoint["model"])
        results[run["id"]]=evaluate_conditions(model,data,cfg)
        print(json.dumps({"id":run["id"],"conditions":results[run["id"]]["conditions"]}),flush=True)
        del model,checkpoint; torch.cuda.empty_cache()
    write_json(root/"causal_checks.json",results)


if __name__=="__main__": main()
