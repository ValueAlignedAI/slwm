"""Two-stage controlled audio/visual repair on the RTX 3090."""
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
from models.slwm_video_repair import VideoRepairSystem,gap_masks
from training.video_repair_common import load_data,crop,get_codec,make_masks,hole_errors
from training.t2_prepare_cremad import sha256
from training.t2_real_av import write_json,stable_hash,cluster_interval


def load_model(path,device="cuda"):
    ck=torch.load(path,map_location="cpu",weights_only=False)
    model=VideoRepairSystem(ck["expert_config"],ck["expert_run"],ck["run"].get("arm","audio"),ck["run"]["seed"],ck["anchor_only"])
    model.load_state_dict(ck["model"],strict=True); return model.to(device).eval(),ck


def initialize(cfg,run,anchor_only):
    torch.manual_seed(run["seed"]); torch.cuda.manual_seed_all(run["seed"]); source=cfg["experts"][str(run["seed"])]
    if sha256(Path(source["path"]))!=source["sha256"]: raise ValueError("Content expert changed")
    ck=torch.load(source["path"],map_location="cpu",weights_only=False)
    model=VideoRepairSystem(ck["config"],ck["run"],run.get("arm","audio"),run["seed"],anchor_only)
    model.expert.load_state_dict(ck["model"],strict=True)
    if not anchor_only:
        anchor_id=next(r["id"] for r in cfg["anchors"] if r["seed"]==run["seed"])
        anchor=torch.load(Path(cfg["artifacts"])/anchor_id/"best.pt",map_location="cpu",weights_only=False)
        missing=model.load_state_dict(anchor["model"],strict=False)
        if missing.unexpected_keys or any(not key.startswith("branch.") for key in missing.missing_keys): raise ValueError("Wrong anchor state")
    return model.cuda(),ck["config"],ck["run"]


def video_control(data,ids,starts,batch,mode,gap_start,gap_length):
    if mode=="wrong":
        wrong=data["wrong"][ids]; alternate=crop(data,wrong,(data["lengths"][wrong]-32)//2); return alternate["mouth"],alternate["video"]
    if mode in {"shift_plus","shift_minus"}:
        offset=6 if mode=="shift_plus" else -6; moved=np.clip(starts+offset,0,data["lengths"][ids]-32); alternate=crop(data,ids,moved)
        return alternate["mouth"],alternate["video"]
    mouth,video=batch["mouth"].clone(),batch["video"].clone()
    if mode=="static":
        index=gap_start-1; b=torch.arange(len(ids),device=mouth.device); mouth=mouth[b,index][:,None].expand_as(mouth).clone(); video=video[b,index][:,None].expand_as(video).clone(); video[...,96:]=0
    elif mode=="occluded":
        mean=mouth.float().mean((1,2,3)).round().to(torch.uint8); mouth[:,:,12:40,16:80]=mean[:,None,None,None]; video.zero_()
    elif mode=="gap_shuffle":
        for j in range(len(ids)):
            start=int(gap_start[j]); length=int(gap_length[j]); order=torch.arange(start,start+length,device=mouth.device).flip(0)
            mouth[j,start:start+length]=mouth[j,order]; video[j,start:start+length]=video[j,order]
    elif mode=="blank": mouth.zero_(); video.zero_()
    return mouth,video


def evaluate(model,codec,data,gap=8,detailed=False,eligible_only=True):
    model.eval(); ids_all=np.where(data["lengths"]>=(50 if eligible_only else 32))[0]; arrays={}; records=[]
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for begin in range(0,len(ids_all),16):
            ids=ids_all[begin:begin+16]; starts=(data["lengths"][ids]-32)//2; batch=crop(data,ids,starts); y=batch["mel100"]
            masks,s,l=make_masks(len(ids),gap,y.device); observation=codec.observe(y,masks["visible"]); present=torch.ones(len(ids),device=y.device,dtype=torch.bool)
            modes=("correct","wrong","shift_plus","shift_minus","static","occluded","gap_shuffle","no_video") if detailed and not model.anchor_only else ("correct",)
            for mode in modes:
                mouth,video=video_control(data,ids,starts,batch,mode,s,l)
                out=model(observation,mouth,video,masks["unknown"],torch.zeros_like(present) if mode=="no_video" else present)
                for key,value in (("hole_mse",hole_errors(out["prediction"],observation["target"],masks["target"])),("unknown_mse",hole_errors(out["prediction"],observation["target"],masks["unknown"])),("mean_gate",out["gate"].mean(1))):
                    arrays.setdefault(mode+"_"+key,[]).append(value.float().cpu().numpy())
                if mode=="correct" and not model.anchor_only: arrays.setdefault("anchor_hole_mse",[]).append(hole_errors(out["anchor"],observation["target"],masks["target"]).cpu().numpy())
            arrays.setdefault("interpolation_hole_mse",[]).append(hole_errors(observation["base"],observation["target"],masks["target"]).cpu().numpy()); records.extend(data["rows"][i] for i in ids)
    flat={k:np.concatenate(v) for k,v in arrays.items()}; result={"samples":len(records),"excluded_short":len(data["rows"])-len(records),"gap_packets":gap,"metrics":{k:float(v.mean()) for k,v in flat.items()},"selection_score":float(flat["correct_hole_mse"].mean())}
    if detailed:
        result["per_clip"]=[{"sample_id":r["sample_id"],"actor":r["actor"],**{k:float(v[i]) for k,v in flat.items()}} for i,r in enumerate(records)]
        result["control_gains"]={}
        for control in ("wrong","shift_plus","shift_minus","static"):
            if control+"_hole_mse" not in flat: continue
            gain=(flat[control+"_hole_mse"]-flat["correct_hole_mse"])/flat[control+"_hole_mse"].clip(1e-9)
            result["control_gains"][control]={"mean_per_clip_gain":float(gain.mean()),"person_cluster_95_interval":cluster_interval(gain,[r["actor"] for r in records])}
    return result


def repair_loss(pred,target,masks):
    return hole_errors(pred,target,masks["target"]).mean()+.25*hole_errors(pred,target,masks["unknown"]).mean()+.1*F.l1_loss(pred[:,1:]-pred[:,:-1],target[:,1:]-target[:,:-1])


def train(cfg,run,anchor_only,data,codec,sources):
    root=Path(cfg["artifacts"]); out=root/run["id"]; out.mkdir(parents=True,exist_ok=True)
    if (out/"training_complete.json").exists(): return
    model,expert_cfg,expert_run=initialize(cfg,run,anchor_only); parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=cfg["training"]["lr"],weight_decay=.01); rng=np.random.default_rng(run["seed"]); tc=cfg["training"]
    steps=tc["anchor_steps"] if anchor_only else tc["steps"]*run.get("steps_multiplier",1); best=float("inf"); curves=[]; started=time.perf_counter()
    counts={"trainable":sum(p.numel() for p in parameters),"frozen":sum(p.numel() for p in model.parameters() if not p.requires_grad),"total":sum(p.numel() for p in model.parameters()),"codec_fitted_values":249376}
    provenance={"config_hash":stable_hash(cfg),"source_hashes":sources,"counts":counts,"manifests":{s:d["manifest_sha256"] for s,d in data.items()},"historical_expert":cfg["experts"][str(run["seed"])],"gpu":torch.cuda.get_device_name(),"true_gap_support_guard":True}
    write_json(out/"config.json",{"matrix":cfg,"run":run,"anchor_only":anchor_only,"provenance":provenance}); c,r=data["crema_train"],data["rav_train"]; batch_size=tc["batch_size"]; nc=tc["crema_batch"]; losses=[]
    for step in range(steps+1):
        if step%tc["eval_every"]==0 or step==steps:
            val={s:evaluate(model,codec,data[s]) for s in ("crema_validation","rav_validation")}; score=float(np.mean([v["selection_score"] for v in val.values()]))
            curves.append({"step":step,"validation":val,"selection_score":score,"seconds":time.perf_counter()-started,"training_loss":float(np.mean(losses)) if losses else None}); losses=[]; write_json(out/"curves.json",curves)
            if score<best:
                best=score; torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":step,"config":cfg,"run":run,"expert_config":expert_cfg,"expert_run":expert_run,"anchor_only":anchor_only,"provenance":provenance,"torch_rng":torch.get_rng_state(),"numpy_rng":rng.bit_generator.state},out/"best.pt")
            print(json.dumps({"run":run["id"],"step":step,"selection":score,"seconds":time.perf_counter()-started}),flush=True)
        if step==steps: break
        ci=rng.integers(0,len(c["rows"]),nc); ri=rng.integers(0,len(r["rows"]),batch_size-nc)
        cs=np.array([rng.integers(0,int(c["lengths"][i])-31) for i in ci]); rs=np.array([rng.integers(0,int(r["lengths"][i])-31) for i in ri])
        lengths=rng.choice([4,8,12],batch_size,p=[.25,.5,.25]); starts=np.array([rng.integers(2,32-int(l)-1) for l in lengths]); gains=rng.uniform(-12,12,batch_size).astype(np.float32); eq=rng.uniform(-6,6,(batch_size,4)).astype(np.float32); kinds=rng.integers(0,4,batch_size)
        cb,rb=crop(c,ci,cs),crop(r,ri,rs); whole={k:torch.cat([cb[k],rb[k]]) for k in cb}; y=codec.augment(whole["mel100"],torch.from_numpy(gains).cuda(),torch.from_numpy(eq).cuda())
        begin=torch.from_numpy(starts).cuda(); size=torch.from_numpy(lengths).cuda(); masks=gap_masks(begin,size); obs=codec.observe(y,masks["visible"]); present=torch.ones(batch_size,device=y.device,dtype=torch.bool)
        if run.get("arm")=="shuffled_video":
            ca=crop(c,c["wrong"][ci],(c["lengths"][c["wrong"][ci]]-32)//2); ra=crop(r,r["wrong"][ri],(r["lengths"][r["wrong"][ri]]-32)//2)
            mouth,video=torch.cat([ca["mouth"],ra["mouth"]]),torch.cat([ca["video"],ra["video"]])
        else: mouth,video=whole["mouth"],whole["video"]
        model.train(); optimizer.zero_grad(set_to_none=True); factor=min(1.,(step+1)/tc["warmup_steps"])*(.1+.9*.5*(1+math.cos(math.pi*step/steps)))
        for group in optimizer.param_groups: group["lr"]=tc["lr"]*factor
        with torch.autocast("cuda",dtype=torch.bfloat16):
            result=model(obs,mouth,video,masks["unknown"],present); loss=repair_loss(result["prediction"],obs["target"],masks)
            if not anchor_only and run["arm"]!="audio":
                projected=result["projected_hint"]; target=result["alignment_target"]; align=((1-(projected*target).sum(-1))*masks["visible"]).sum()/masks["visible"].sum().clamp_min(1)
                loss=loss+tc["alignment_weight"]*align
                if run["arm"]!="no_negatives":
                    negatives=[]
                    for d,ids,ss,original in ((c,ci,cs,cb),(r,ri,rs,rb)):
                        n=len(ids); pieces=[]
                        for j,mode in enumerate(("wrong","shift_plus","static","blank")):
                            mm,vv=video_control(d,ids,ss,original,mode,begin[:n] if d is c else begin[nc:],size[:n] if d is c else size[nc:]); pieces.append((mm,vv))
                        key=torch.as_tensor(kinds[:n] if d is c else kinds[nc:],device=y.device); index=torch.arange(n,device=y.device)
                        negatives.append((torch.stack([p[0] for p in pieces])[key,index],torch.stack([p[1] for p in pieces])[key,index]))
                    nm,nv=torch.cat([n[0] for n in negatives]),torch.cat([n[1] for n in negatives])
                    negative=model.branch(result["field"],result["anchor"],nm,nv,masks["visible"],masks["unknown"],target,present)
                    loss=loss+tc["negative_weight"]*negative["residual"].square().mean()
                    if run["arm"]!="ungated_video": loss=loss+tc["gate_weight"]*(F.binary_cross_entropy_with_logits(result["gate_logits"],torch.ones_like(result["gate_logits"]))+F.binary_cross_entropy_with_logits(negative["gate_logits"],torch.zeros_like(negative["gate_logits"])))
        if not torch.isfinite(loss): raise RuntimeError("Nonfinite repair loss")
        loss.backward()
        if step==0:
            counts["graph_connected_trainable"]=sum(p.numel() for p in parameters if p.grad is not None)
            write_json(out/"counts.json",counts)
        torch.nn.utils.clip_grad_norm_(parameters,1.); optimizer.step(); losses.append(float(loss.detach()))
    torch.save({"model":model.state_dict(),"step":steps,"config":cfg,"run":run,"expert_config":expert_cfg,"expert_run":expert_run,"anchor_only":anchor_only,"provenance":provenance},out/"last.pt")
    best_ck=torch.load(out/"best.pt",map_location="cpu",weights_only=False)
    write_json(out/"training_complete.json",{"run":run,"anchor_only":anchor_only,"selected_step":best_ck["step"],"best_validation":best,"wall_seconds":time.perf_counter()-started,"checkpoint_sha256":sha256(out/"best.pt"),"last_checkpoint_sha256":sha256(out/"last.pt"),"steps":steps,"positive_windows":steps*batch_size,"provenance":provenance})
    del model,optimizer; torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--stage",choices=("anchors","branches","evaluate"),required=True); args=parser.parse_args(); cfg=json.loads(Path(args.config).read_text()); root=Path(cfg["artifacts"]); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True; codec=get_codec(cfg)
    if args.stage=="evaluate":
        for split in ("crema_validation","rav_validation","crema_test","rav_test"):
            data=load_data(cfg,split)
            for run in cfg["anchors"]+cfg["runs"]:
                model,ck=load_model(root/run["id"]/"best.pt")
                for gap in ((8,) if split.endswith("validation") else (4,8,12)):
                    metrics=evaluate(model,codec,data,gap,detailed=True); metrics.update({"run":run,"selected_step":ck["step"],"checkpoint_sha256":sha256(root/run["id"]/"best.pt"),"split":split,"manifest_sha256":data["manifest_sha256"]})
                    write_json(root/run["id"]/(split+f"_gap{gap}.json"),metrics)
                del model; print(json.dumps({"evaluated":run["id"],"split":split}),flush=True)
            del data; torch.cuda.empty_cache()
        return
    names=(args.config,cfg["protocol"],"models/slwm_video_repair.py","training/video_repair_data.py","training/video_repair_common.py","training/video_repair_train.py","models/slwm_content.py","models/slwm_restoration.py","models/slwm_motion.py","models/slwm_generation.py","training/t2_train_latents.py","training/restoration_train.py","training/t2_prepare_generation.py","training/t2_prepare_motion.py","training/t2_prepare_cremad.py","training/t2_real_av.py")
    sources={}
    for name in names:
        path=Path(name); copy=root/"source_snapshot"/path; copy.parent.mkdir(parents=True,exist_ok=True)
        if copy.exists() and sha256(copy)!=sha256(path): raise ValueError("Source changed after registration: "+name)
        if not copy.exists(): shutil.copyfile(path,copy)
        sources[name]=sha256(path)
    write_json(root/"source_hashes.json",sources)
    data={s:load_data(cfg,s) for s in ("crema_train","rav_train","crema_validation","rav_validation")}
    for run in cfg["anchors"] if args.stage=="anchors" else cfg["runs"]: train(cfg,run,args.stage=="anchors",data,codec,sources)


if __name__=="__main__": main()
