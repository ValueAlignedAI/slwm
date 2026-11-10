"""Frozen old-head/codec-rate controls and a new-domain alignment audit.

The external alignment audit is an additional exploratory diagnostic. It never
selects a checkpoint or changes the registered generation thresholds.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from models.slwm_motion import MotionSLWM
from models.slwm_generation import unpack_mel,audio_errors
from training.t2_generation_train import load_data,crop
from training.t2_motion_train import evaluate as alignment_evaluate,masked_inputs
from training.t2_prepare_cremad import sha256
from training.t2_real_av import write_json


def interpolate25(features: torch.Tensor,old: dict,new: dict) -> torch.Tensor:
    """25 Hz normalized [B,T,64] -> physical-grid 100 Hz [B,4T,64]."""
    x=features.float()*torch.as_tensor(old["audio_scale"],device=features.device,dtype=torch.float32)+torch.as_tensor(old["audio_mean"],device=features.device,dtype=torch.float32)
    time=torch.arange(4*x.shape[1],device=x.device)/4; left=time.long(); right=(left+1).clamp_max(x.shape[1]-1)
    fraction=(time-left)[None,:,None]; y=x[:,left]*(1-fraction)+x[:,right]*fraction
    return (y-y.new_tensor(new["mean"]))/y.new_tensor(new["scale"])


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text()); root=Path(cfg["artifacts"]); torch.set_num_threads(4)
    checkpoint=torch.load(cfg["checkpoint"],map_location="cpu",weights_only=False)
    model=MotionSLWM(checkpoint["config"]["run"]["model"]).cuda().eval(); model.load_state_dict(checkpoint["model"])
    with np.load(Path(cfg["source_corpus"])/"codec.npz",allow_pickle=False) as d: old={k:d[k] for k in d.files}
    with np.load(Path(cfg["corpus"])/"codec100.npz",allow_pickle=False) as d: new={k:d[k] for k in d.files}
    for split in ("validation","test","external"):
        data=load_data(Path(cfg["corpus"]),split); n=len(data["rows"]); starts=(data["lengths"]-32)//2; metrics={}
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            for begin in range(0,n,32):
                ids=np.arange(begin,min(begin+32,n)); a=crop(data,"audio_coarse",ids,starts[ids],32); v=crop(data,"video_coarse",ids,starts[ids],32)
                target=unpack_mel(crop(data,"mel100",ids,starts[ids],32))
                candidates={"codec25_interpolated":interpolate25(a[:,:,:64],old,new)}
                for mode in ("audio","video","null"):
                    am,vm=masked_inputs(a,v,mode); output=model(a,v,am,vm)["audio"][:,:,:64]
                    candidates["old_"+mode+"_interpolated"]=interpolate25(output,old,new)
                for name,pred in candidates.items():
                    for metric,value in audio_errors(pred,target).items(): metrics.setdefault(name+"_"+metric,[]).append(value.float().cpu().numpy())
        flat={k:np.concatenate(v) for k,v in metrics.items()}
        write_json(root/(split+"_audio_controls.json"),{"metrics":{k:float(v.mean()) for k,v in flat.items()},
            "scope":"old/codec outputs compared on the SAME 100 Hz normalized physical grid; linear interpolation with held last three subframes",
            "per_clip":[{"sample_id":r["sample_id"],"actor":r["actor"],**{k:float(v[i]) for k,v in flat.items()}} for i,r in enumerate(data["rows"])]})
        if split=="external":
            alignment_data={"audio":data["audio_coarse"].float(),"video":data["video_coarse"].float(),"rows":data["rows"],"lengths":data["lengths"]}
            result=alignment_evaluate(model,alignment_data,checkpoint["config"]["matrix"],detailed=True)
            result.update({"scope":"Additional exploratory frozen alignment on a new dataset; no external tuning or selection", "checkpoint_sha256":cfg["checkpoint_sha256"],
                "data_manifest_sha256":sha256(Path(cfg["corpus"])/"external_manifest.json"),"source_sha256":sha256(Path(__file__))})
            write_json(root/"external_frozen_alignment.json",result)
        del data; torch.cuda.empty_cache()


if __name__=="__main__": main()
