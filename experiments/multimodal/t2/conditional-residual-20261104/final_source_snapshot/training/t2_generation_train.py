"""Matched frozen-core edge-decoder experiment, config-driven and auditable."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from models.slwm_generation import MultirateAudioHead,SeedMotionHead,unpack_mel,visual_loss,visual_errors,audio_errors
from training.t2_motion_train import shuffled_indices
from training.t2_prepare_cremad import sha256
from training.t2_real_av import write_json,stable_hash,cluster_interval


KEYS=("field_audio","field_video","audio_coarse","video_coarse","mel100","mouth")


def load_data(root: Path,split: str,device: str="cuda") -> dict:
    """Verified immutable signal arrays -> tensors, no fitted statistics here."""
    manifest=json.loads((root/(split+"_manifest.json")).read_text())
    for name,digest in manifest["files"].items():
        if sha256(root/name)!=digest: raise ValueError("Generation corpus hash changed: "+name)
    data={k:torch.from_numpy(np.load(root/f"{split}_{k}.npy",allow_pickle=False)).to(device) for k in KEYS}
    data["lengths"]=np.load(root/f"{split}_lengths.npy",allow_pickle=False)
    data["rows"]=json.loads((root/(split+".json")).read_text()); data["wrong"]=shuffled_indices(data["rows"])
    return data


def crop(data: dict,key: str,indices: np.ndarray,starts: np.ndarray,window: int) -> torch.Tensor:
    """Signal [N,128,...] -> gathered unpadded [B,W,...] windows."""
    device=data[key].device; i=torch.as_tensor(indices,device=device)[:,None]
    t=torch.as_tensor(starts,device=device)[:,None]+torch.arange(window,device=device)[None]
    value=data[key][i,t].float()
    return value/255. if key=="mouth" else value


def condition(data: dict,run: dict,ids: np.ndarray,starts: np.ndarray,window: int,mode: str="correct") -> torch.Tensor:
    """Only declared observed input -> [B,W,D]; target pixels never accepted."""
    if run["input"]=="time_only": return torch.zeros((len(ids),window,256),device=data["mouth"].device)
    if mode=="wrong" or run.get("shuffled",False):
        ids=data["wrong"][ids]; starts=(data["lengths"][ids]-window)//2
    x=crop(data,run["input"],ids,starts,window)
    return torch.zeros_like(x) if mode=="null" else x


def make_model(run: dict) -> torch.nn.Module:
    return MultirateAudioHead() if run["task"]=="audio" else SeedMotionHead(run.get("warp",True))


def predict(model: torch.nn.Module,run: dict,data: dict,ids: np.ndarray,starts: np.ndarray,window: int,mode: str="correct") -> tuple[torch.Tensor,torch.Tensor]:
    """Declared conditioning and one seed, when applicable -> output and target."""
    x=condition(data,run,ids,starts,window,mode)
    if run["task"]=="audio": return model(x),unpack_mel(crop(data,"mel100",ids,starts,window))
    target=crop(data,"mouth",ids,starts,window)
    return model(x,target[:,0]),target


def evaluate(model: torch.nn.Module,run: dict,data: dict,window: int=32,detailed: bool=False) -> dict:
    """Fixed middle windows; wrong audio stays within the same person."""
    model.eval(); metrics={}; n=len(data["rows"]); starts=(data["lengths"]-window)//2
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for begin in range(0,n,16):
            ids=np.arange(begin,min(begin+16,n))
            for mode in (("correct","wrong","null") if detailed else ("correct",)):
                output,target=predict(model,run,data,ids,starts[ids],window,mode)
                errors=(audio_errors if run["task"]=="audio" else visual_errors)(output,target)
                for key,value in errors.items(): metrics.setdefault(mode+"_"+key,[]).append(value.float().cpu().numpy())
            if detailed:
                if run["task"]=="visual":
                    errors=visual_errors(target[:,:1].expand_as(target),target)
                else: errors=audio_errors(torch.zeros_like(target),target)
                for key,value in errors.items(): metrics.setdefault("persistence_"+key,[]).append(value.float().cpu().numpy())
    flat={k:np.concatenate(v) for k,v in metrics.items()}
    out={"samples":n,"metrics":{k:float(v.mean()) for k,v in flat.items()}}
    key="correct_mel_mse" if run["task"]=="audio" else "correct_roi_mse"; out["selection_score"]=out["metrics"][key]
    if run["task"]=="visual":
        out["metrics"]["motion_energy_ratio"]=out["metrics"]["correct_motion_energy"]/max(out["metrics"]["correct_true_motion_energy"],1e-10)
    if detailed:
        name="mel_mse" if run["task"]=="audio" else "roi_mse"
        gain=(flat["wrong_"+name]-flat["correct_"+name])/np.maximum(flat["wrong_"+name],1e-9)
        out["correct_vs_same_person_wrong_relative_gain"]=float(gain.mean())
        out["gain_person_cluster_95_interval"]=cluster_interval(gain,[r["actor"] for r in data["rows"]])
        out["per_clip"]=[{"sample_id":r["sample_id"],"actor":r["actor"],**{k:float(v[i]) for k,v in flat.items()}} for i,r in enumerate(data["rows"])]
    return out


def train_run(cfg: dict,run: dict,train: dict,val: dict,source: dict):
    out=Path(cfg["artifacts"])/run["id"]; out.mkdir(parents=True,exist_ok=True)
    if (out/"training_complete.json").exists(): return
    seed=run["seed"]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model=make_model(run).cuda(); tc=cfg["training"]; steps=tc["steps"]; window=tc["window"]
    batch=tc["audio_batch"] if run["task"]=="audio" else tc["visual_batch"]
    optimizer=torch.optim.AdamW(model.parameters(),lr=tc["lr"],weight_decay=.01)
    rng=np.random.default_rng(seed); best=float("inf"); curves=[]; start=time.perf_counter()
    frozen=4741089 if run["input"].startswith("field_") else 0
    provenance={"config_hash":stable_hash(cfg),"run_hash":stable_hash(run),"source_hashes":source,
        "training_manifest_sha256":sha256(Path(cfg["corpus"])/"train_manifest.json"),
        "validation_manifest_sha256":sha256(Path(cfg["corpus"])/"validation_manifest.json"),
        "frozen_core_sha256":cfg["checkpoint_sha256"] if frozen else None,
        "parameter_accounting":{"decoder_trainable":sum(p.numel() for p in model.parameters()),"frozen_core":frozen,
            "codec_fitted_values":249248+128,"external_generation_parameters":0},
        "observations":run["input"]+(" + one observed starting frame" if run["task"]=="visual" else ""),
        "task":"simultaneous conditional generation; diagnostic proposal, no policy commit", "device":torch.cuda.get_device_name()}
    write_json(out/"config.json",{"matrix":cfg,"run":run,"provenance":provenance})
    for step in range(steps+1):
        if step%tc["eval_every"]==0:
            result=evaluate(model,run,val,window); entry={"step":step,"validation":result,"elapsed_seconds":time.perf_counter()-start}
            curves.append(entry); write_json(out/"curves.json",curves)
            if result["selection_score"]<best:
                best=result["selection_score"]
                torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":step,"config":{"matrix":cfg,"run":run},
                    "provenance":provenance,"torch_rng":torch.get_rng_state(),"numpy_generator":rng.bit_generator.state},out/"best.pt")
            print(json.dumps({"run":run["id"],"step":step,"validation":result["selection_score"],"elapsed":entry["elapsed_seconds"]}),flush=True)
        if step==steps: break
        model.train(); ids=rng.integers(0,len(train["rows"]),batch)
        starts=np.array([rng.integers(0,int(train["lengths"][i])-window+1) for i in ids])
        # Always consume these independent draws, even for controls.
        rng.integers(0,len(train["rows"]),batch)
        warm=min(1.,(step+1)/tc["warmup_steps"]); decay=.1+.9*.5*(1+math.cos(math.pi*step/steps))
        for group in optimizer.param_groups: group["lr"]=tc["lr"]*warm*decay
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            output,target=predict(model,run,train,ids,starts,window)
            if run["task"]=="visual": loss=visual_loss(output,target)
            else:
                p,y=output.float(),target.float()
                loss=F.l1_loss(p,y)+.5*F.l1_loss(p[:,1:]-p[:,:-1],y[:,1:]-y[:,:-1])
        if not torch.isfinite(loss): raise RuntimeError("Nonfinite decoder loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
    checkpoint=torch.load(out/"best.pt",map_location="cpu",weights_only=False)
    write_json(out/"training_complete.json",{"id":run["id"],"selected_step":checkpoint["step"],"validation_score":best,
        "steps":steps,"windows_seen":steps*batch,"physical_seconds_sampled":steps*batch*window/25,
        "wall_seconds":time.perf_counter()-start,"checkpoint_sha256":sha256(out/"best.pt"),"provenance":provenance})
    del model,optimizer; torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--evaluate",action="store_true")
    parser.add_argument("--runs",nargs="*"); args=parser.parse_args(); cfg=json.loads(Path(args.config).read_text())
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
    runs=[r for r in cfg["runs"] if not args.runs or r["id"] in args.runs]; root=Path(cfg["corpus"]); out=Path(cfg["artifacts"])
    if args.evaluate:
        for split in ("validation","test","external"):
            data=load_data(root,split)
            for run in runs:
                checkpoint=torch.load(out/run["id"]/"best.pt",map_location="cpu",weights_only=False)
                model=make_model(run).cuda().eval(); model.load_state_dict(checkpoint["model"])
                result=evaluate(model,run,data,cfg["training"]["window"],True)
                result.update({"run":run,"checkpoint_sha256":sha256(out/run["id"]/"best.pt"),"selected_step":checkpoint["step"],
                    "split":split,"data_manifest_sha256":sha256(root/(split+"_manifest.json"))})
                write_json(out/run["id"]/(split+"_metrics.json"),result); del model
            del data; torch.cuda.empty_cache()
        return
    snapshot=out/"source_snapshot"; source={}
    names=(args.config,"models/slwm_generation.py","models/slwm_motion.py","training/t2_generation_train.py",
        "training/t2_prepare_generation.py","training/t2_train_latents.py","training/t2_motion_train.py","training/t2_real_av.py",
        "training/t2_prepare_cremad.py",cfg["protocol"])
    for name in names:
        p=Path(name); q=snapshot/p; q.parent.mkdir(parents=True,exist_ok=True)
        if q.exists() and sha256(q)!=sha256(p): raise ValueError("Training source changed after snapshot: "+name)
        if not q.exists(): shutil.copyfile(p,q)
        source[name]=sha256(p)
    write_json(out/"source_hashes.json",source)
    data=load_data(root,"train"); val=load_data(root,"validation")
    for run in runs: train_run(cfg,run,data,val,source)


if __name__=="__main__": main()
