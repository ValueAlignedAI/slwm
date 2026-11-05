"""Frozen linear readout of ONLY audio-world fields, no observation bypass."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from models.slwm_restoration import RestorationSLWM
from training.restoration_train import load_data,load_codec,crop
from training.t2_real_av import write_json,cluster_interval
from training.t2_prepare_cremad import sha256


def representations(model,codec,data,ids):
    features=[]; targets=[]
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for begin in range(0,len(ids),32):
            index=np.asarray(ids[begin:begin+32]); starts=(data["lengths"][index]-32)//2; y,v=crop(data,index,starts)
            one=torch.ones(y.shape[:2],device=y.device,dtype=torch.bool); obs=codec.observe(y,one)
            z=model.world.encode(obs["audio"],torch.zeros_like(v),one,torch.zeros_like(one))[0]
            features.append(z.float().cpu().numpy()); targets.append(y.reshape(len(y),32,4,64)[:,:,0].cpu().numpy())
    return np.concatenate(features),np.concatenate(targets)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text()); root=Path(cfg["artifacts"]); torch.set_num_threads(4); codec=load_codec(cfg)
    training={s:load_data(cfg,s) for s in ("crema_train","libri_train")}; results={}
    for run in cfg["runs"]:
        if run["arm"] in {"raw_audio","raw_av"}: continue
        checkpoint=torch.load(root/run["id"]/"best.pt",map_location="cpu",weights_only=False); model=RestorationSLWM(cfg,run).cuda().eval(); model.load_state_dict(checkpoint["model"])
        xs=[]; ys=[]; selected={}
        for split,d in training.items():
            order=sorted(range(len(d["rows"])),key=lambda i:hashlib.sha256(("R1-probe/"+d["rows"][i]["sample_id"]).encode()).hexdigest())[:512]
            x,y=representations(model,codec,d,order); xs.append(x.reshape(-1,256)); ys.append(y.reshape(-1,64)); selected[split]=[d["rows"][i]["sample_id"] for i in order]
        x=np.concatenate(xs).astype(np.float64); y=np.concatenate(ys).astype(np.float64); mean=x.mean(0); target_mean=y.mean(0); x-=mean; y-=target_mean
        gram=x.T@x; ridge=.01*np.trace(gram)/256; weight=np.linalg.solve(gram+ridge*np.eye(256),x.T@y).astype(np.float32)
        np.savez_compressed(root/run["id"]/"content_probe.npz",weight=weight,feature_mean=mean.astype(np.float32),target_mean=target_mean.astype(np.float32),ridge=np.array(ridge))
        measured={}
        for split in ("crema_test","libri_test","ravdess_test"):
            d=load_data(cfg,split); a,b=representations(model,codec,d,list(range(len(d["rows"]))))
            pred=(a-mean.astype(np.float32))@weight+target_mean.astype(np.float32); errors=np.mean((pred-b)**2,axis=(1,2))
            measured[split]={"mse":float(errors.mean()),"per_clip":[{"sample_id":r["sample_id"],"actor":r["actor"],"mse":float(e)} for r,e in zip(d["rows"],errors)]}; del d
        result={"run":run,"scope":"Exploratory frozen linear decodability of observed coarse sound; ONLY normalized world features, no raw bypass", "coefficient_bias_values":16448,
            "fitted_input_means":256,"checkpoint_sha256":sha256(root/run["id"]/"best.pt"),"ridge":float(ridge),"training_clips":selected,"metrics":measured}
        write_json(root/run["id"]/"content_probe.json",result); results[run["id"]]=result; del model; torch.cuda.empty_cache()
    comparisons={}
    for key,result in results.items():
        if key=="EXP-T2-509": continue
        comparisons[key]={}
        for split in result["metrics"]:
            ours=result["metrics"][split]["per_clip"]; base=results["EXP-T2-509"]["metrics"][split]["per_clip"]
            if [r["sample_id"] for r in ours]!=[r["sample_id"] for r in base]: raise ValueError("Unpaired content probes")
            gain=np.array([(b["mse"]-a["mse"])/max(b["mse"],1e-9) for a,b in zip(ours,base)])
            comparisons[key][split]={"mean_relative_gain":float(gain.mean()),"person_cluster_95_interval":cluster_interval(gain,[r["actor"] for r in ours])}
    write_json(root/"content_probe_summary.json",{"scope":"Exploratory, not a primary training/selection gate", "source_sha256":sha256(Path(__file__)),"runs":results,"paired_gain_vs_old_frozen_core":comparisons})


if __name__=="__main__": main()
