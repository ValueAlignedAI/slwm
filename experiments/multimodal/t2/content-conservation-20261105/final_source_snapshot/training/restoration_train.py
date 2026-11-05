"""Preregistered mixed-modality restoration, with true same-window controls."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
import torch

from models.slwm_restoration import RestorationSLWM,RestorationCodec,restoration_loss,restoration_errors
from training.t2_motion_train import alignment_loss,shuffled_indices,offset_baseline
from training.t2_prepare_cremad import sha256
from training.t2_real_av import write_json,stable_hash,cluster_interval


def load_codec(cfg: dict,device: str="cuda") -> RestorationCodec:
    with np.load(Path(cfg["old_corpus"])/"codec.npz",allow_pickle=False) as d: old={k:d[k] for k in d.files}
    with np.load(Path(cfg["generation_corpus"])/"codec100.npz",allow_pickle=False) as d: new={k:d[k] for k in d.files}
    return RestorationCodec(old,new).to(device)


def load_data(cfg: dict,split: str,device: str="cuda") -> dict:
    """Verified targets/video [N,128,F] -> tensors; no cached world features."""
    if split.startswith("crema_"):
        root=Path(cfg["generation_corpus"]); name=split.removeprefix("crema_"); video_name="video_coarse"
    else: root=Path(cfg["new_corpus"]); name=split; video_name="video"
    manifest=root/(name+"_manifest.json"); card=json.loads(manifest.read_text())
    for file,digest in card["files"].items():
        if sha256(root/file)!=digest: raise ValueError("Source corpus changed: "+file)
    rows=json.loads((root/(name+".json")).read_text())
    data={key:torch.from_numpy(np.load(root/f"{name}_{source}.npy",allow_pickle=False)).to(device) for key,source in (("mel100","mel100"),("video",video_name))}
    data.update({"rows":rows,"lengths":np.load(root/f"{name}_lengths.npy"),"wrong":shuffled_indices(rows),"manifest_sha256":sha256(manifest),"split":split,
        "video_present":np.array([r.get("video_present",True) for r in rows])})
    return data


def crop(data: dict,ids: np.ndarray,starts: np.ndarray,window: int=32) -> tuple[torch.Tensor,torch.Tensor]:
    """Fixed physical-time [N,128,F] -> target/video [B,32,F], no padding."""
    device=data["mel100"].device; i=torch.as_tensor(ids,device=device)[:,None]; t=torch.as_tensor(starts,device=device)[:,None]+torch.arange(window,device=device)[None]
    return data["mel100"][i,t].float(),data["video"][i,t].float()


def holes(first: np.ndarray,second: np.ndarray,device: torch.device,window: int=32) -> torch.Tensor:
    """Two independent four-packet intervals -> observed [B,W] mask."""
    index=np.arange(window)[None]; missing=((index>=first[:,None])&(index<first[:,None]+4))|((index>=second[:,None])&(index<second[:,None]+4))
    return torch.from_numpy(~missing).to(device)


def fixed_holes(rows: list[dict],device: torch.device) -> torch.Tensor:
    values=[int(hashlib.sha256(("T2-R1/"+r["sample_id"]).encode()).hexdigest()[:16],16) for r in rows]
    return holes(np.array([2+x%10 for x in values]),np.array([18+(x//10)%9 for x in values]),device)


def output(model: RestorationSLWM,observation: dict,video: torch.Tensor,video_visible: torch.Tensor) -> torch.Tensor:
    return model(observation["audio"],video,observation["visible"],video_visible,observation["base"])


def evaluate(model: RestorationSLWM,codec: RestorationCodec,data: dict,detailed: bool=False) -> dict:
    model.eval(); count=len(data["rows"]); starts=(data["lengths"]-32)//2; arrays={}; device=data["mel100"].device
    with torch.inference_mode(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
        for begin in range(0,count,32):
            ids=np.arange(begin,min(begin+32,count)); rows=[data["rows"][i] for i in ids]; y,v=crop(data,ids,starts[ids])
            visible=fixed_holes(rows,device); obs=codec.observe(y,visible); one=torch.ones_like(visible); zero=torch.zeros_like(visible)
            vm=one&torch.as_tensor(data["video_present"][ids],device=device)[:,None]
            conditions={"paired":(obs,v,vm)}
            if detailed:
                wrong=data["wrong"][ids]; _,wv=crop(data,wrong,starts[wrong])
                conditions.update({"clean_audio":(codec.observe(y,one),v,zero),"masked_audio":(obs,v,zero),"wrong_video":(obs,wv,vm),"null_video":(obs,torch.zeros_like(v),zero)})
            for name,(o,vv,vvm) in conditions.items():
                pred=output(model,o,vv,vvm)
                for key,value in restoration_errors(pred,o["target"],o["visible"]).items(): arrays.setdefault(name+"_"+key,[]).append(value.float().cpu().numpy())
            for key,value in restoration_errors(obs["base"],obs["target"],visible).items(): arrays.setdefault("interpolation_"+key,[]).append(value.float().cpu().numpy())
    flat={k:np.concatenate(v) for k,v in arrays.items()}; result={"samples":count,"metrics":{k:float(v.mean()) for k,v in flat.items()}}
    result["selection_score"]=result["metrics"]["paired_hole_mse"]
    if detailed:
        gains={}
        for control in ("masked_audio","wrong_video","interpolation"):
            gain=(flat[control+"_hole_mse"]-flat["paired_hole_mse"])/flat[control+"_hole_mse"].clip(1e-9)
            gains[control]={"mean_per_clip_gain":float(gain.mean()),"person_cluster_95_interval":cluster_interval(gain,[r["actor"] for r in data["rows"]])}
        result["paired_gains"]=gains
        result["per_clip"]=[{"sample_id":r["sample_id"],"actor":r["actor"],**{k:float(v[i]) for k,v in flat.items()}} for i,r in enumerate(data["rows"])]
    return result


def synchronization(model: RestorationSLWM,codec: RestorationCodec,data: dict) -> dict:
    """Independent 32-packet encodes; seven valid equal-time shift candidates."""
    if not model.has_core: return {"status":"not_applicable_raw_control"}
    model.eval(); scores=[]; wrong_scores=[]; zero_scores=[]; baseline=[]; records=[]; device=data["mel100"].device
    eligible=np.where(data["lengths"]>=50)[0]; offsets=(-9,-6,-3,0,3,6,9)
    with torch.inference_mode(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
        for begin in range(0,len(eligible),32):
            ids=eligible[begin:begin+32]; center=(data["lengths"][ids]-32)//2; y,v=crop(data,ids,center)
            one=torch.ones(y.shape[:2],device=device,dtype=torch.bool); a=codec.observe(y,one)["audio"]
            ar,_=model.world.independent(a,v)
            wrong=data["wrong"][ids]; wy,_=crop(data,wrong,(data["lengths"][wrong]-32)//2); wa=codec.observe(wy,one)["audio"]
            wr,_=model.world.independent(wa,v); zr,_=model.world.independent(torch.zeros_like(a),v)
            clips=[crop(data,ids,center+shift)[1] for shift in offsets]; ss=[]; ws=[]; zs=[]
            for vv in clips:
                vr=model.world.independent_video(vv)
                ss.append((ar*vr).sum(-1).mean(-1).float().cpu().numpy()); ws.append((wr*vr).sum(-1).mean(-1).float().cpu().numpy()); zs.append((zr*vr).sum(-1).mean(-1).float().cpu().numpy())
            scores.append(np.stack(ss,1)); wrong_scores.append(np.stack(ws,1)); zero_scores.append(np.stack(zs,1)); baseline.append(offset_baseline(a,clips)); records.extend(data["rows"][i] for i in ids)
    if not records: return {"status":"no_eligible_clips"}
    correct=(np.concatenate(scores).argmax(1)==3).astype(float); base=(np.concatenate(baseline).argmax(1)==3).astype(float)
    return {"eligible_clips":len(correct),"accuracy":float(correct.mean()),"wrong_audio_accuracy":float(np.mean(np.concatenate(wrong_scores).argmax(1)==3)),
        "zero_audio_accuracy":float(np.mean(np.concatenate(zero_scores).argmax(1)==3)),"energy_baseline_accuracy":float(base.mean()),"chance":1/7,
        "person_cluster_accuracy_95_interval":cluster_interval(correct,[r["actor"] for r in records]),
        "improvement_over_energy_95_interval":cluster_interval(correct-base,[r["actor"] for r in records]),
        "per_clip":[{"sample_id":r["sample_id"],"actor":r["actor"],"correct":float(s),"energy_correct":float(b)} for r,s,b in zip(records,correct,base)]}


def initialize(cfg: dict,run: dict) -> RestorationSLWM:
    torch.manual_seed(run["seed"]); torch.cuda.manual_seed_all(run["seed"]); model=RestorationSLWM(cfg,run)
    if model.has_core:
        source=cfg["checkpoints"]["vanilla" if run["arm"]=="vanilla" else "signal"]
        if sha256(Path(source["path"]))!=source["sha256"]: raise ValueError("Warmstart changed")
        old=torch.load(source["path"],map_location="cpu",weights_only=False); model.initialize_world(old["model"])
    return model.cuda()


def train(cfg: dict,run: dict,data: dict,codec: RestorationCodec,source: dict):
    out=Path(cfg["artifacts"])/run["id"]; out.mkdir(parents=True,exist_ok=True)
    if (out/"training_complete.json").exists(): return
    model=initialize(cfg,run); tc=cfg["training"]; rng=np.random.default_rng(run["seed"])
    world=[p for n,p in model.named_parameters() if n.startswith("world.") and p.requires_grad]
    heads=[p for n,p in model.named_parameters() if not n.startswith("world.") and p.requires_grad]
    optimizer=torch.optim.AdamW([{"params":world,"lr":tc["core_lr"],"base_lr":tc["core_lr"]},{"params":heads,"lr":tc["head_lr"],"base_lr":tc["head_lr"]}],weight_decay=.01)
    counts={"trainable":sum(p.numel() for p in model.parameters() if p.requires_grad),"frozen":sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "decoder":sum(p.numel() for p in model.head.parameters()),"total":sum(p.numel() for p in model.parameters()),"codec_fitted_values":249376}
    provenance={"config_hash":stable_hash(cfg),"source_hashes":source,"counts":counts,"data_manifests":{s:d["manifest_sha256"] for s,d in data.items()},
        "pretraining":"corresponding 1600-step M1 core" if model.has_core else "none","context":"exactly the same 32 packets; no cached external context","gpu":torch.cuda.get_device_name()}
    write_json(out/"config.json",{"matrix":cfg,"run":run,"provenance":provenance}); curves=[]; best=float("inf"); start=time.perf_counter(); train_losses=[]
    c,l=data["crema_train"],data["libri_train"]; batch=tc["batch_size"]; nc=tc["crema_batch"]
    for step in range(tc["steps"]+1):
        if step%tc["eval_every"]==0:
            validation={s:evaluate(model,codec,data[s]) for s in ("crema_validation","libri_validation")}; score=float(np.mean([v["selection_score"] for v in validation.values()]))
            entry={"step":step,"validation":validation,"selection_score":score,"training_loss_since_last_eval":float(np.mean(train_losses)) if train_losses else None,"seconds":time.perf_counter()-start}
            curves.append(entry); train_losses=[]; write_json(out/"curves.json",curves)
            if score<best:
                best=score; torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":step,"config":cfg,"run":run,"provenance":provenance,
                    "torch_rng":torch.get_rng_state(),"numpy_rng":rng.bit_generator.state},out/"best.pt")
            print(json.dumps({"run":run["id"],"step":step,"selection":score,"seconds":entry["seconds"]}),flush=True)
        if step==tc["steps"]: break
        ci=rng.integers(0,len(c["rows"]),nc); li=rng.integers(0,len(l["rows"]),batch-nc)
        cs=np.array([rng.integers(0,int(c["lengths"][i])-31) for i in ci]); ls=np.array([rng.integers(0,int(l["lengths"][i])-31) for i in li])
        first=rng.integers(2,12,batch); second=rng.integers(18,27,batch); gain=rng.uniform(-12,12,batch).astype(np.float32); eq=rng.uniform(-6,6,(batch,4)).astype(np.float32)
        keep=rng.random(batch)<.5; shifts=rng.choice([-9,-6,-3,3,6,9],nc); use_wrong=rng.random(nc)<.5
        cy,cv=crop(c,ci,cs); ly,lv=crop(l,li,ls); y=torch.cat([cy,ly]); video=torch.cat([cv,lv])
        g=torch.from_numpy(gain).cuda(); q=torch.from_numpy(eq).cuda()
        if run["arm"]!="no_augmentation": y=codec.augment(y,g,q)
        visible=holes(first,second,y.device); obs=codec.observe(y,visible)
        present=np.concatenate([c["video_present"][ci],l["video_present"][li]])&keep; vm=torch.as_tensor(present,device=y.device)[:,None].expand_as(visible)
        model.train(); optimizer.zero_grad(set_to_none=True)
        factor=min(1.,(step+1)/tc["warmup_steps"])*(.1+.9*.5*(1+math.cos(math.pi*step/tc["steps"])))
        for group in optimizer.param_groups: group["lr"]=group["base_lr"]*factor
        with torch.autocast("cuda",dtype=torch.bfloat16):
            pred=output(model,obs,video,vm); loss=restoration_loss(pred,obs["target"],visible)
            if model.has_core and run["arm"] not in {"no_alignment","frozen_core"} and step%tc["alignment_every"]==0:
                eligible=np.where(c["lengths"][ci]>=50)[0]
                if len(eligible)>=2:
                    ai=ci[eligible]; center=(c["lengths"][ai]-32)//2; ay,av=crop(c,ai,center)
                    if run["arm"]!="no_augmentation": ay=codec.augment(ay,g[eligible],q[eligible])
                    a=codec.observe(ay,torch.ones(ay.shape[:2],device=y.device,dtype=torch.bool))["audio"]
                    _,shifted=crop(c,ai,center+shifts[eligible]); wi=c["wrong"][ai]; _,wrong=crop(c,wi,(c["lengths"][wi]-32)//2)
                    negative=torch.where(torch.as_tensor(use_wrong[eligible],device=y.device)[:,None,None],wrong,shifted)
                    align,_=alignment_loss(model.world,a,av,negative,[c["rows"][i]["sentence"] for i in ai])
                    loss=loss+tc["alignment_weight"]*align
        if not torch.isfinite(loss): raise RuntimeError("Nonfinite restoration objective")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); train_losses.append(float(loss.detach()))
    checkpoint=torch.load(out/"best.pt",map_location="cpu",weights_only=False)
    torch.save({"model":model.state_dict(),"step":tc["steps"],"config":cfg,"run":run,"provenance":provenance},out/"last.pt")
    write_json(out/"training_complete.json",{"run":run,"selected_step":checkpoint["step"],"best_validation_score":best,"wall_seconds":time.perf_counter()-start,
        "checkpoint_sha256":sha256(out/"best.pt"),"last_checkpoint_sha256":sha256(out/"last.pt"),"provenance":provenance,"windows_seen":batch*tc["steps"]})
    del model,optimizer; torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--evaluate",action="store_true"); parser.add_argument("--runs",nargs="*"); args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text()); out=Path(cfg["artifacts"]); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
    runs=[r for r in cfg["runs"] if not args.runs or r["id"] in args.runs]; codec=load_codec(cfg)
    if args.evaluate:
        for split in ("crema_validation","libri_validation","crema_test","libri_test","ravdess_test"):
            data=load_data(cfg,split)
            for run in runs:
                checkpoint=torch.load(out/run["id"]/"best.pt",map_location="cpu",weights_only=False); model=RestorationSLWM(cfg,run).cuda().eval(); model.load_state_dict(checkpoint["model"],strict=True)
                result=evaluate(model,codec,data,True)
                if split in {"crema_test","ravdess_test"}: result["synchronization"]=synchronization(model,codec,data)
                result.update({"run":run,"split":split,"selected_step":checkpoint["step"],"checkpoint_sha256":sha256(out/run["id"]/"best.pt"),"manifest_sha256":data["manifest_sha256"]})
                write_json(out/run["id"]/(split+"_metrics.json"),result); del model
                print(json.dumps({"evaluated":run["id"],"split":split}),flush=True)
            del data; torch.cuda.empty_cache()
        return
    names=(args.config,cfg["protocol"],"models/slwm_restoration.py","models/slwm_motion.py","models/slwm_generation.py","training/restoration_train.py",
        "training/restoration_data.py","training/generation_external.py","training/t2_motion_train.py","training/t2_train_latents.py","training/t2_real_av.py",
        "training/t2_prepare_cremad.py","training/t2_prepare_generation.py","training/t2_prepare_motion.py")
    source={}
    for name in names:
        p=Path(name); q=out/"source_snapshot"/p; q.parent.mkdir(parents=True,exist_ok=True)
        if q.exists() and sha256(q)!=sha256(p): raise ValueError("Training source changed after freezing: "+name)
        if not q.exists(): shutil.copyfile(p,q)
        source[name]=sha256(p)
    write_json(out/"source_hashes.json",source)
    data={s:load_data(cfg,s) for s in ("crema_train","libri_train","crema_validation","libri_validation")}
    for run in runs: train(cfg,run,data,codec,source)


if __name__=="__main__": main()
