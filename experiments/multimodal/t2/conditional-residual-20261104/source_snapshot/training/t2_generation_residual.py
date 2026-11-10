"""Separate G2 remedial experiment; G1 artifacts and source stay immutable."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F

from models.slwm_generation import unpack_mel,audio_errors
from models.slwm_generation_residual import SignalResidualAudioHead
from training.t2_generation_train import load_data,crop
from training.t2_prepare_cremad import sha256
from training.t2_real_av import write_json,stable_hash,cluster_interval


def codecs(cfg: dict) -> tuple[dict,dict]:
    g1=json.loads(Path(cfg["g1_config"]).read_text())
    with np.load(Path(g1["source_corpus"])/"codec.npz",allow_pickle=False) as d: old={k:d[k] for k in d.files}
    with np.load(Path(cfg["corpus"])/"codec100.npz",allow_pickle=False) as d: new={k:d[k] for k in d.files}
    return old,new


def evaluate(model: SignalResidualAudioHead,run: dict,data: dict,detailed: bool=False) -> dict:
    model.eval(); flat={}; count=len(data["rows"]); starts=(data["lengths"]-32)//2
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for begin in range(0,count,32):
            ids=np.arange(begin,min(begin+32,count)); a=crop(data,"audio_coarse",ids,starts[ids],32)
            x=crop(data,run["input"],ids,starts[ids],32); y=unpack_mel(crop(data,"mel100",ids,starts[ids],32))
            candidates={"learned":model(x,a),"interpolation":model.baseline(a)}
            if detailed:
                wrong=data["wrong"][ids]; wx=crop(data,run["input"],wrong,starts[wrong],32)
                # Keep the correct observed skip; only the correction loses alignment.
                candidates["wrong_correction"]=model(wx,a)
            for name,pred in candidates.items():
                for key,value in audio_errors(pred,y).items(): flat.setdefault(name+"_"+key,[]).append(value.float().cpu().numpy())
    flat={k:np.concatenate(v) for k,v in flat.items()}; metrics={k:float(v.mean()) for k,v in flat.items()}
    result={"samples":count,"metrics":metrics,"selection_score":metrics["learned_mel_mse"]}
    if detailed:
        gain=(flat["interpolation_mel_mse"]-flat["learned_mel_mse"])/flat["interpolation_mel_mse"].clip(1e-9)
        result.update({"mean_per_clip_gain_vs_interpolation":float(gain.mean()),"person_cluster_gain_95_interval":cluster_interval(gain,[r["actor"] for r in data["rows"]]),
            "per_clip":[{"sample_id":r["sample_id"],"actor":r["actor"],**{k:float(v[i]) for k,v in flat.items()}} for i,r in enumerate(data["rows"])]})
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--evaluate",action="store_true"); args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text()); root=Path(cfg["artifacts"]); root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True; old,new=codecs(cfg)
    if args.evaluate:
        for split in ("validation","test","external"):
            data=load_data(Path(cfg["corpus"]),split)
            for run in cfg["runs"]:
                checkpoint=torch.load(root/run["id"]/"best.pt",map_location="cpu",weights_only=False)
                model=SignalResidualAudioHead(old,new).cuda().eval(); model.load_state_dict(checkpoint["model"])
                result=evaluate(model,run,data,True); result.update({"run":run,"step":checkpoint["step"],"checkpoint_sha256":sha256(root/run["id"]/"best.pt")})
                write_json(root/run["id"]/(split+"_metrics.json"),result); del model
            del data; torch.cuda.empty_cache()
        return
    files=(args.config,cfg["g1_config"],cfg["protocol"],"models/slwm_generation_residual.py","training/t2_generation_residual.py","models/slwm_generation.py",
        "training/t2_generation_train.py","training/t2_real_av.py","training/t2_prepare_cremad.py","training/t2_motion_train.py")
    hashes={}
    for name in files:
        path=root/"source_snapshot"/name; path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and sha256(path)!=sha256(Path(name)): raise ValueError("Frozen G2 source changed")
        if not path.exists(): shutil.copyfile(name,path)
        hashes[name]=sha256(Path(name))
    write_json(root/"source_hashes.json",hashes)
    train=load_data(Path(cfg["corpus"]),"train"); val=load_data(Path(cfg["corpus"]),"validation"); tc=cfg["training"]
    for run in cfg["runs"]:
        out=root/run["id"]; out.mkdir(exist_ok=True)
        if (out/"training_complete.json").exists(): continue
        torch.manual_seed(run["seed"]); torch.cuda.manual_seed_all(run["seed"]); rng=np.random.default_rng(run["seed"])
        model=SignalResidualAudioHead(old,new).cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=tc["lr"],weight_decay=.01)
        curves=[]; best=float("inf"); start=time.perf_counter()
        provenance={"config_hash":stable_hash(cfg),"source_hashes":hashes,"decoder_trainable":sum(p.numel() for p in model.parameters()),
            "frozen_core":4741089 if run["input"]=="field_audio" else 0,"codec_fitted_values":249376,
            "train_manifest_sha256":sha256(Path(cfg["corpus"])/"train_manifest.json"),"validation_manifest_sha256":sha256(Path(cfg["corpus"])/"validation_manifest.json"),
            "scope":"Exploratory repair informed by G1 external failure; no external fitting"}
        write_json(out/"config.json",{"config":cfg,"run":run,"provenance":provenance})
        for step in range(tc["steps"]+1):
            if step%tc["eval_every"]==0:
                result=evaluate(model,run,val); curves.append({"step":step,"validation":result,"elapsed_seconds":time.perf_counter()-start}); write_json(out/"curves.json",curves)
                if result["selection_score"]<best:
                    best=result["selection_score"]; torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":step,"config":cfg,"run":run,
                        "provenance":provenance,"torch_rng":torch.get_rng_state(),"numpy_generator":rng.bit_generator.state},out/"best.pt")
                print(json.dumps({"run":run["id"],"step":step,"validation_mse":result["selection_score"],"seconds":time.perf_counter()-start}),flush=True)
            if step==tc["steps"]: break
            ids=rng.integers(0,len(train["rows"]),tc["batch_size"]); starts=np.array([rng.integers(0,int(train["lengths"][i])-32+1) for i in ids]); rng.integers(0,len(train["rows"]),tc["batch_size"])
            a=crop(train,"audio_coarse",ids,starts,32); x=crop(train,run["input"],ids,starts,32); y=unpack_mel(crop(train,"mel100",ids,starts,32))
            model.train(); optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups: group["lr"]=tc["lr"]*min(1.,(step+1)/tc["warmup_steps"])*(.1+.9*.5*(1+math.cos(math.pi*step/tc["steps"])))
            with torch.autocast("cuda",dtype=torch.bfloat16):
                pred=model(x,a); loss=F.l1_loss(pred,y)+.5*F.l1_loss(pred[:,1:]-pred[:,:-1],y[:,1:]-y[:,:-1])
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite residual loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
        checkpoint=torch.load(out/"best.pt",map_location="cpu",weights_only=False)
        write_json(out/"training_complete.json",{"run":run,"selected_step":checkpoint["step"],"wall_seconds":time.perf_counter()-start,
            "checkpoint_sha256":sha256(out/"best.pt"),"provenance":provenance,"windows_seen":tc["steps"]*tc["batch_size"]})
        del model,optimizer; torch.cuda.empty_cache()


if __name__=="__main__": main()
