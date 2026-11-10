"""Registered T2-M1 architecture/scaling experiment on real timed audio/video."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from models.slwm_motion import MotionSLWM
from training.t2_prepare_cremad import sha256
from training.t2_real_av import write_json,stable_hash,cluster_interval


def load_data(root: Path,split: str,device: torch.device) -> dict:
    """Verified [N,128,A/V] arrays and true lengths -> device tensors."""
    card=json.loads((root/"dataset_card.json").read_text())
    for suffix in (".npz",".json"):
        if sha256(root/(split+suffix))!=card["files"][split+suffix]: raise ValueError("Motion corpus hash changed")
    with np.load(root/f"{split}.npz",allow_pickle=False) as d:
        data={k:torch.from_numpy(d[k].astype(np.float32)).to(device) for k in ("audio","video")}
        data["lengths"]=d["lengths"].astype(int)
    data["rows"]=json.loads((root/f"{split}.json").read_text())
    if len(data["rows"])!=len(data["lengths"]): raise ValueError("Metadata length mismatch")
    return data


def crop(data: dict,indices: np.ndarray,starts: np.ndarray,window: int) -> tuple[torch.Tensor,torch.Tensor]:
    """Gather equal physical-time [B,W,A/V] windows without padding."""
    device=data["audio"].device
    i=torch.as_tensor(indices,device=device)[:,None]
    t=torch.as_tensor(starts,device=device)[:,None]+torch.arange(window,device=device)[None]
    return data["audio"][i,t],data["video"][i,t]


def shuffled_indices(rows: list[dict]) -> np.ndarray:
    """Fixed same-person derangement; no identities or labels enter the model."""
    order=np.arange(len(rows))
    for actor in sorted({r["actor"] for r in rows}):
        ii=np.array([i for i,r in enumerate(rows) if r["actor"]==actor]); order[ii]=np.roll(ii,1)
    return order


def masked_inputs(a: torch.Tensor,v: torch.Tensor,mode: str) -> tuple[torch.Tensor,torch.Tensor]:
    """[B,W,A/V] -> two [B,W] visibility masks; never use target values."""
    one=torch.ones(a.shape[:2],device=a.device,dtype=torch.bool); zero=torch.zeros_like(one)
    return (one if mode in {"paired","audio"} else zero,one if mode in {"paired","video"} else zero)


def alignment_loss(model: MotionSLWM,a: torch.Tensor,v: torch.Tensor,negative_video: torch.Tensor,sentences: list[str],local: bool=True) -> tuple[torch.Tensor,dict]:
    """Independent streams -> global and same-recording temporal alignment."""
    ar,vr=model.independent(a,v)
    ag=F.normalize(ar.mean(1),dim=-1); vg=F.normalize(vr.mean(1),dim=-1)
    scale=model.logit_scale.exp().clamp(max=50.)
    logits=(ag@vg.T)*scale
    labels=torch.arange(len(a),device=a.device)
    same=torch.as_tensor(np.equal.outer(sentences,sentences),device=a.device)
    same.fill_diagonal_(False)
    # Repeated prompts are weak labels used only to avoid false negatives.
    # They are not encoder inputs or prediction targets in T2-M1.
    logits=logits.masked_fill(same,-1e4)
    global_loss=.5*(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))
    nv=model.independent_video(negative_video)
    positive=(ar*vr).sum(-1).mean(-1)
    negative=(ar*nv).sum(-1).mean(-1)
    sync=F.softplus((negative-positive)*scale).mean()
    loss=.15*global_loss+(.7*sync if local else 0.)
    return loss,{"global_alignment":global_loss,"sync_alignment":sync,"sync_choice":(positive>negative).float().mean()}


def basic_errors(out: dict,a: torch.Tensor,v: torch.Tensor) -> dict[str,torch.Tensor]:
    """Decoder [B,W,F] -> per-sample [B] modality errors."""
    return {"audio_mel":((out["audio"][...,:64].float()-a[...,:64])**2).mean((1,2)),
            "audio_delta":((out["audio"][...,64:].float()-a[...,64:])**2).mean((1,2)),
            "video_appearance":((out["video"][...,:96].float()-v[...,:96])**2).mean((1,2)),
            "video_motion":((out["video"][...,96:].float()-v[...,96:])**2).mean((1,2))}


def offset_baseline(a: torch.Tensor,candidates: list[torch.Tensor]) -> np.ndarray:
    """Untrained centered audio-energy / visual-motion correlation [B,K]."""
    envelope=a[...,:64].mean(-1).float()
    envelope=envelope-envelope.mean(-1,keepdim=True)
    envelope=F.normalize(envelope,dim=-1)
    scores=[]
    for v in candidates:
        motion=v[...,96:].float().square().mean(-1).sqrt()
        motion=motion-motion.mean(-1,keepdim=True)
        scores.append((envelope*F.normalize(motion,dim=-1)).sum(-1).cpu().numpy())
    return np.stack(scores,axis=1)


def evaluate(model: MotionSLWM,data: dict,cfg: dict,*,detailed: bool=False) -> dict:
    model.eval(); window=cfg["training"]["window"]; batch=cfg["training"]["batch_size"]
    device=data["audio"].device; amp=device.type=="cuda"
    count=len(data["rows"]); starts=(data["lengths"]-window)//2
    wrong=shuffled_indices(data["rows"])
    errors={}; areps=[]; vreps=[]; offsets=[]; baseline_offsets=[]; eligible_rows=[]
    with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):
        for begin in range(0,count,batch):
            indices=np.arange(begin,min(begin+batch,count)); a,v=crop(data,indices,starts[indices],window)
            for mode in ("audio","video"):
                am,vm=masked_inputs(a,v,mode); result=basic_errors(model(a,v,am,vm),a,v)
                for name,value in result.items(): errors.setdefault(mode+"_input_"+name,[]).append(value.float().cpu().numpy())
            ar,vr=model.independent(a,v); areps.append(ar.mean(1).float().cpu().numpy()); vreps.append(vr.mean(1).float().cpu().numpy())
            if detailed:
                wa,wv=crop(data,wrong[indices],starts[wrong[indices]],window)
                for mode,aa,vv in (("audio",wa,v),("video",a,wv),("null",a,v)):
                    am,vm=masked_inputs(aa,vv,mode); out=model(aa,vv,am,vm)
                    result=basic_errors(out,a,v)
                    for name,value in result.items(): errors.setdefault(mode+"_wrong_"+name,[]).append(value.float().cpu().numpy())
                for name,value in (("null_mean_audio",a[...,:64].square().mean((1,2))),
                                   ("null_mean_motion",v[...,96:].square().mean((1,2)))):
                    errors.setdefault(name,[]).append(value.cpu().numpy())
            valid=indices[(data["lengths"][indices]>=window+18)]
            if len(valid):
                aa,_=crop(data,valid,starts[valid],window)
                clips=[crop(data,valid,starts[valid]+shift,window)[1] for shift in cfg["evaluation"]["offsets"]]
                av,_=model.independent(aa,clips[3])
                scores=[]
                for vv in clips:
                    vvrep=model.independent_video(vv)
                    scores.append((av*vvrep).sum(-1).mean(-1).float().cpu().numpy())
                offsets.append(np.stack(scores,1)); baseline_offsets.append(offset_baseline(aa,clips))
                eligible_rows.extend([data["rows"][i] for i in valid])
    flat={key:np.concatenate(value) for key,value in errors.items()}
    means={key:float(value.mean()) for key,value in flat.items()}
    A,V=np.concatenate(areps),np.concatenate(vreps)
    A=A/np.linalg.norm(A,axis=1,keepdims=True).clip(1e-8); V=V/np.linalg.norm(V,axis=1,keepdims=True).clip(1e-8)
    similarities=A@V.T; n=len(A); gold=np.arange(n)
    result={"samples":n,"errors":means,"retrieval":{"candidate_count":n,"chance_r1":1/n,
            "audio_to_video_r1":float(np.mean(similarities.argmax(1)==gold)),"video_to_audio_r1":float(np.mean(similarities.argmax(0)==gold)),
            "independent":True}}
    offset_scores=np.concatenate(offsets); baseline_scores=np.concatenate(baseline_offsets)
    # All seven candidates are independently encoded equal-length valid windows.
    success=(offset_scores.argmax(1)==3).astype(float); base=(baseline_scores.argmax(1)==3).astype(float)
    result["synchronization"]={"eligible_clips":len(success),"candidate_count":7,"chance":1/7,
        "accuracy":float(success.mean()),"motion_energy_baseline_accuracy":float(base.mean())}
    result["selection_score"]=(1-float(success.mean()))+.05*(means["video_input_audio_mel"]+means["audio_input_video_motion"])
    if detailed:
        actors=[r["actor"] for r in data["rows"]]
        sa=[r["actor"] for r in eligible_rows]
        result["synchronization"].update({"person_cluster_accuracy_interval":cluster_interval(success,sa),
            "improvement_over_baseline_interval":cluster_interval(success-base,sa),
            "per_clip":[{"sample_id":r["sample_id"],"actor":r["actor"],"correct":float(s),"baseline_correct":float(b)} for r,s,b in zip(eligible_rows,success,base)]})
        result["cross_modal_contribution"]={}
        for name,paired,shuffled in (("video_to_audio","video_input_audio_mel","video_wrong_audio_mel"),
                                     ("audio_to_motion","audio_input_video_motion","audio_wrong_video_motion")):
            p,s=flat[paired],flat[shuffled]; gain=(s-p)/max(float(s.mean()),1e-8)
            result["cross_modal_contribution"][name]={"relative_gain":float(gain.mean()),"person_cluster_interval":cluster_interval(gain,actors)}
        result["per_clip"]=[{"sample_id":r["sample_id"],"actor":r["actor"],**{key:float(value[i]) for key,value in flat.items()}} for i,r in enumerate(data["rows"])]
        sentences=np.array([r["sentence"] for r in data["rows"]])
        result["retrieval"].update({"audio_to_video_sentence_match":float(np.mean(sentences[similarities.argmax(1)]==sentences)),
                                   "video_to_audio_sentence_match":float(np.mean(sentences[similarities.argmax(0)]==sentences)),
                                   "same_person_choice_accuracy":float(np.mean(similarities[gold,gold]>similarities[gold,wrong]))})
    return result


def train_one(cfg: dict,run: dict,data: dict,card: dict,root: Path) -> dict:
    seed=run["seed"]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    rng=np.random.default_rng(seed); device=data["train"]["audio"].device
    model=MotionSLWM(run["model"]).to(device); t=cfg["training"]
    optimizer=torch.optim.AdamW(model.parameters(),lr=t["lr"],weight_decay=.05)
    folder=root/run["id"]; folder.mkdir(parents=True,exist_ok=True)
    config={"matrix":cfg,"run":run}; write_json(folder/"config.json",config)
    sources=["models/slwm_motion.py","training/t2_motion_train.py","training/t2_prepare_motion.py","training/t2_train_latents.py",cfg["protocol"]]
    provenance={"source_commit":cfg["source_commit"],"config_sha256":stable_hash(config),
                "source_hashes":{p:sha256(Path(p)) for p in sources},"dataset_card_sha256":sha256(Path(cfg["corpus"])/"dataset_card.json")}
    registry={"experiment_id":run["id"],"sprint":"T2-M1","status":"running","parameter_counts":model.parameter_counts(),
        "fitted_codec_values":card["feature_spec"]["fitted_codec_values"],"split_counts":card["split_counts"],
        "hardware":torch.cuda.get_device_name(0),"torch":torch.__version__,"numpy":np.__version__,"seed":seed,
        "precision":"BF16 autocast, FP32 parameters and optimizer","training_steps":t["steps"],"batch_size":t["batch_size"],**provenance}
    write_json(folder/"registry.json",registry)
    initial=evaluate(model,data["validation"],cfg); best=initial["selection_score"]; best_step=0
    curves=[{"step":0,"validation":initial}]; start=time.perf_counter()
    def save(step):
        torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"config":config,"step":step,
                    "validation_score":best,"provenance":provenance,"codec_path":str(Path(cfg["corpus"])/"codec.npz"),
                    "rng":{"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all(),"numpy":rng.bit_generator.state}},folder/"best.pt")
    save(0); torch.cuda.reset_peak_memory_stats()
    for step in range(1,t["steps"]+1):
        model.train(); indices=rng.choice(len(data["train"]["rows"]),t["batch_size"],replace=False)
        slack=data["train"]["lengths"][indices]-t["window"]
        starts=np.array([rng.choice([0,s]) if s<6 else rng.integers(0,s+1) for s in slack]); negative_starts=[]
        for s,start_at in zip(slack,starts):
            options=[i for i in range(max(0,start_at-9),min(s,start_at+9)+1) if abs(i-start_at)>=3]
            negative_starts.append(rng.choice(options))
        a,v=crop(data["train"],indices,starts,t["window"])
        _,negative=crop(data["train"],indices,np.array(negative_starts),t["window"])
        sentences=[data["train"]["rows"][i]["sentence"] for i in indices]
        if run.get("shuffled"):
            v=v.roll(1,0); negative=negative.roll(1,0)
        modes=torch.as_tensor(rng.integers(0,3,len(indices)),device=device)
        am=(modes!=2)[:,None].expand(-1,t["window"]); vm=(modes!=1)[:,None].expand_as(am)
        warmup=t["warmup_steps"]; factor=min(step/warmup,1.)*.5*(1+math.cos(math.pi*max(0,step-warmup)/max(1,t["steps"]-warmup)))
        for group in optimizer.param_groups: group["lr"]=t["lr"]*factor
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            out=model(a,v,am,vm); e=basic_errors(out,a,v)
            aw=torch.where(modes==2,1.,.25); vw=torch.where(modes==1,1.,.25)
            prediction=((e["audio_mel"]+.25*e["audio_delta"])*aw+(e["video_motion"]+.1*e["video_appearance"])*vw).mean()
            align,details=alignment_loss(model,a,v,negative,sentences,local=not run.get("no_local_alignment",False))
            loss=prediction+align
        if not torch.isfinite(loss): raise RuntimeError(f"Nonfinite loss: {run['id']} step {step}")
        loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(grad): raise RuntimeError("Nonfinite gradient; stopped")
        optimizer.step()
        if step%50==0:
            curves.append({"step":step,"loss":float(loss.detach()),"prediction":float(prediction.detach()),
                **{k:float(value.detach()) for k,value in details.items()},"gradient_norm":float(grad),"elapsed_seconds":time.perf_counter()-start})
        if step%t["eval_every"]==0 or step==t["steps"]:
            validation=evaluate(model,data["validation"],cfg)
            if validation["selection_score"]<best:
                best=validation["selection_score"]; best_step=step; save(step)
            curves.append({"step":step,"validation":validation}); write_json(folder/"curves.json",{"curves":curves})
            print(json.dumps({"id":run["id"],"step":step,"sync":validation["synchronization"]["accuracy"],
                "v2a_mse":validation["errors"]["video_input_audio_mel"],"a2motion_mse":validation["errors"]["audio_input_video_motion"],
                "best_step":best_step,"seconds":time.perf_counter()-start}),flush=True)
    seconds=time.perf_counter()-start
    model.load_state_dict(torch.load(folder/"best.pt",map_location=device,weights_only=False)["model"])
    validation=evaluate(model,data["validation"],cfg,detailed=True)
    test=evaluate(model,data["test"],cfg,detailed=True)
    sync=test["synchronization"]
    gates={"sync_pass":sync["accuracy"]>=cfg["gates"]["sync_accuracy"] and sync["accuracy"]>sync["motion_energy_baseline_accuracy"] and sync["improvement_over_baseline_interval"][0]>0}
    for direction,key,null in (("video_to_audio","video_input_audio_mel","null_mean_audio"),("audio_to_motion","audio_input_video_motion","null_mean_motion")):
        contribution=test["cross_modal_contribution"][direction]
        gates[direction+"_pass"]=(1-test["errors"][key]/test["errors"][null]>=.05 and contribution["relative_gain"]>=.02 and contribution["person_cluster_interval"][0]>0)
    result={"experiment_id":run["id"],"run":run,"initial_validation":initial,"validation":validation,"test":test,
        "best_step":best_step,"completed_steps":t["steps"],"samples_seen":t["steps"]*t["batch_size"],"training_seconds":seconds,
        "peak_gpu_memory_mb":torch.cuda.max_memory_allocated()/2**20,"parameter_counts":model.parameter_counts(),
        "checkpoint":str(folder/"best.pt"),"checkpoint_sha256":sha256(folder/"best.pt"),"gates":gates,**provenance}
    write_json(folder/"metrics.json",result); write_json(folder/"curves.json",{"curves":curves})
    registry.update(status="completed",training_seconds=seconds,best_step=best_step,checkpoint=result["checkpoint"],checkpoint_sha256=result["checkpoint_sha256"])
    write_json(folder/"registry.json",registry)
    print(json.dumps({"completed":run["id"],"gates":gates,"sync":sync["accuracy"],"test_errors":test["errors"]}),flush=True)
    del model,optimizer; torch.cuda.empty_cache(); return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text()); root=Path(cfg["artifacts"]); root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True,warn_only=True); torch.backends.cudnn.benchmark=False
    card=json.loads((Path(cfg["corpus"])/"dataset_card.json").read_text())
    data={s:load_data(Path(cfg["corpus"]),s,torch.device("cuda")) for s in ("train","validation","test")}
    snapshot=root/"source_snapshot"
    if not snapshot.exists():
        for source in ["models/slwm_motion.py","training/t2_motion_train.py","training/t2_prepare_motion.py","training/t2_train_latents.py",cfg["protocol"],args.config]:
            target=snapshot/source; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
    results=[]
    for run in cfg["runs"]:
        path=root/run["id"]/"metrics.json"
        if path.exists():
            result=json.loads(path.read_text())
            if result["config_sha256"]!=stable_hash({"matrix":cfg,"run":run}): raise ValueError("Completed run config changed")
        else: result=train_one(cfg,run,data,card,root)
        results.append(result); write_json(root/"summary.json",{"config":cfg,"runs":results})


if __name__=="__main__": main()
