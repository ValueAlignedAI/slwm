"""Same training-only ridge assay, including the untouched R2 test-other set."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from models.slwm_content import ContentRestorationSLWM
from models.slwm_restoration import RestorationSLWM
from training.restoration_other import expanded_config
from training.restoration_content import load_other
from training.restoration_train import load_codec,load_data,evaluate
from evals.restoration_content_probe import representations
from training.t2_real_av import write_json,cluster_interval
from training.t2_prepare_cremad import sha256


def load_model(path: Path):
    checkpoint=torch.load(path,map_location="cpu",weights_only=False); cfg=checkpoint["config"]; run=checkpoint["run"]
    model=(ContentRestorationSLWM if cfg["sprint"]=="T2-R2" else RestorationSLWM)(cfg,run).cuda().eval(); model.load_state_dict(checkpoint["model"])
    return model


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); args=parser.parse_args(); cfg=expanded_config(args.config); root=Path(cfg["artifacts"])
    parent=json.loads(Path(cfg["parent_config"]).read_text()); previous=Path(parent["artifacts"]); torch.set_num_threads(4); codec=load_codec(cfg)
    train={s:load_data(cfg,s) for s in ("crema_train","libri_train")}
    selection=json.loads((previous/"EXP-T2-501/content_probe.json").read_text())["training_clips"]; results={}
    for run in cfg["runs"]:
        model=load_model(root/run["id"]/"best.pt"); xs=[]; ys=[]
        for split,data in train.items():
            lookup={r["sample_id"]:i for i,r in enumerate(data["rows"])}; ids=[lookup[k] for k in selection[split]]
            x,y=representations(model,codec,data,ids); xs.append(x.reshape(-1,256)); ys.append(y.reshape(-1,64))
        x=np.concatenate(xs).astype(np.float64); y=np.concatenate(ys).astype(np.float64); mean=x.mean(0); bias=y.mean(0); x-=mean; y-=bias
        gram=x.T@x; ridge=.01*np.trace(gram)/256; weight=np.linalg.solve(gram+ridge*np.eye(256),x.T@y).astype(np.float32)
        np.savez_compressed(root/run["id"]/"content_probe.npz",weight=weight,feature_mean=mean.astype(np.float32),target_mean=bias.astype(np.float32),ridge=np.array(ridge))
        measured={}
        for split in ("crema_test","libri_test","ravdess_test","other_test"):
            data=load_other(cfg) if split=="other_test" else load_data(cfg,split); a,b=representations(model,codec,data,list(range(len(data["rows"]))))
            error=np.mean(((a-mean.astype(np.float32))@weight+bias.astype(np.float32)-b)**2,axis=(1,2))
            measured[split]={"mse":float(error.mean()),"per_clip":[{"sample_id":r["sample_id"],"actor":r["actor"],"mse":float(e)} for r,e in zip(data["rows"],error)]}; del data
        result={"run":run,"checkpoint_sha256":sha256(root/run["id"]/"best.pt"),"coefficient_bias_values":16448,"input_means":256,"ridge":float(ridge),"training_clips":selection,"metrics":measured}
        write_json(root/run["id"]/"content_probe.json",result); results[run["id"]]=result; del model; torch.cuda.empty_cache()
    other=load_other(cfg); references={}
    for key in ("EXP-T2-501","EXP-T2-502","EXP-T2-506","EXP-T2-509","EXP-T2-507"):
        model=load_model(previous/key/"best.pt"); metrics=evaluate(model,codec,other,True); write_json(root/"reference_metrics"/(key+"_other_test.json"),metrics)
        if key!="EXP-T2-507":
            old=json.loads((previous/key/"content_probe.json").read_text())
            with np.load(previous/key/"content_probe.npz",allow_pickle=False) as w:
                a,b=representations(model,codec,other,list(range(len(other["rows"]))))
                error=np.mean(((a-w["feature_mean"])@w["weight"]+w["target_mean"]-b)**2,axis=(1,2))
            old["metrics"]["other_test"]={"mse":float(error.mean()),"per_clip":[{"sample_id":r["sample_id"],"actor":r["actor"],"mse":float(e)} for r,e in zip(other["rows"],error)]}
            references[key]=old; write_json(root/"reference_probes"/(key+".json"),old)
        del model; torch.cuda.empty_cache()
    comparisons={}
    for key,result in results.items():
        comparisons[key]={}
        full="EXP-T2-502" if key=="EXP-T2-522" else "EXP-T2-501"
        for split in result["metrics"]:
            comparisons[key][split]={}
            for baseline in (full,"EXP-T2-509"):
                ours=result["metrics"][split]["per_clip"]; base=references[baseline]["metrics"][split]["per_clip"]
                if [r["sample_id"] for r in ours]!=[r["sample_id"] for r in base]: raise ValueError("Unpaired R2 readouts")
                gain=np.array([(b["mse"]-a["mse"])/max(b["mse"],1e-9) for a,b in zip(ours,base)])
                comparisons[key][split][baseline]={"mean_relative_gain":float(gain.mean()),"person_cluster_95_interval":cluster_interval(gain,[r["actor"] for r in ours])}
    write_json(root/"content_probe_summary.json",{"scope":"Frozen observed-sound linear readout, no raw bypass; not a general semantic score", "source_sha256":sha256(Path(__file__)),"runs":results,"references":references,"paired_comparisons":comparisons})


if __name__=="__main__": main()
