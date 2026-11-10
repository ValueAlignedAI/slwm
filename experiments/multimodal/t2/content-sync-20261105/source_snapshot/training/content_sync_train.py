"""Registered content/candidate training, independent calibration and fresh tests."""
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
from models.slwm_content_sync import SpeechContentBase,ContentSyncSystem
from models.slwm_content import content_loss
from models.slwm_video_repair import gap_masks
from training.content_sync_common import load_data,crop,get_codec,make_masks,unit_loss,hole_mse,repair_loss,control_mouth
from training.research_sources import file_sha
from training.t2_real_av import write_json,stable_hash,cluster_interval


def load_model(path,device="cuda"):
    ck=torch.load(path,map_location="cpu",weights_only=False); run=ck["run"]; cfg=ck["config"]
    if ck["kind"]=="base": model=SpeechContentBase(ck["expert_cfg"],ck["expert_run"],ck["words"],run["seed"])
    else: model=ContentSyncSystem(ck["expert_cfg"],ck["expert_run"],ck["words"],run["seed"],run["arm"],cfg["training"]["audio_front_width"])
    model.load_state_dict(ck["model"],strict=True); return model.to(device).eval(),ck


def initialize(cfg,run,kind):
    source=cfg["experts"][str(run["seed"])]; assert file_sha(source["path"])==source["sha256"]
    original=torch.load(source["path"],map_location="cpu",weights_only=False); words=len(json.loads((Path(cfg["corpus"])/"vocabulary.json").read_text())); torch.manual_seed(run["seed"]); torch.cuda.manual_seed_all(run["seed"])
    expert_cfg,expert_run=original["config"],original["run"]
    if kind=="base":
        model=SpeechContentBase(expert_cfg,expert_run,words,run["seed"]); model.expert.load_state_dict(original["model"],strict=True)
    else:
        model=ContentSyncSystem(expert_cfg,expert_run,words,run["seed"],run["arm"],cfg["training"]["audio_front_width"])
        base=torch.load(Path(cfg["artifacts"])/cfg["base_by_seed"][str(run["seed"])]/"best.pt",map_location="cpu",weights_only=False); model.base.load_state_dict(base["model"],strict=True)
        if run["arm"]!="audio":
            audio=torch.load(Path(cfg["artifacts"])/cfg["audio_by_seed"][str(run["seed"])]/"best.pt",map_location="cpu",weights_only=False)
            kept={k.removeprefix("branch."):v for k,v in audio["model"].items() if k.startswith("branch.")}; model.audio_candidate.load_state_dict(kept,strict=True)
            for key,value in model.base.state_dict().items(): assert torch.equal(value,audio["model"]["base."+key])
    return model.cuda(),expert_cfg,expert_run,words


def predict(model,kind,obs,mouth,masks,threshold=None,present=None):
    return model(obs) if kind=="base" else model(obs,mouth,masks["unknown"],present,threshold)


def evaluate(model,kind,codec,data,gap=8,threshold=None,detailed=False,max_examples=None):
    model.eval(); ids_all=np.where(data["lengths"]>=50)[0]
    if max_examples is not None: ids_all=ids_all[:max_examples]
    records=[]; totals={}; actuals=[]
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for begin in range(0,len(ids_all),16):
            ids=ids_all[begin:begin+16]; starts=(data["lengths"][ids]-32)//2; batch=crop(data,ids,starts); masks=make_masks(len(ids),gap,batch["mel100"].device); obs=codec.observe(batch["mel100"],masks["visible"])
            video=kind!="base" and model.arm!="audio"; modes=("correct","wrong","shift6","shift-6","shift4","shift-4","static","blank","missing") if detailed and video and data["mouth"] is not None else ("correct",)
            for mode in modes:
                mouth=control_mouth(data,ids,starts,batch,mode,gap); present=torch.full((len(ids),),mode!="missing" and data["mouth"] is not None,device=mouth.device,dtype=torch.bool)
                result=predict(model,kind,obs,mouth,masks,threshold,present); pred=result["prediction"]; unit=result["unit_logits"].argmax(-1); target=batch["labels"]; valid=(target>0)&masks["target"].reshape(len(ids),32,4)[:,:,0]
                values={"hole_mse":hole_mse(pred,obs["target"],masks["target"]),"unknown_mse":hole_mse(pred,obs["target"],masks["unknown"]),"word_correct":((unit==target)&valid).sum(1).float(),"word_frames":valid.sum(1).float()}
                if video:
                    values.update({"quality":result["quality"],"accepted":result.get("accepted",present).float(),"offset":result["predicted_offset"].float(),"raw_hole_mse":hole_mse(result["raw_prediction"],obs["target"],masks["target"]),"fallback_hole_mse":hole_mse(result["fallback"],obs["target"],masks["target"])})
                for k,v in values.items(): totals.setdefault(mode+"_"+k,[]).append(v.float().cpu().numpy())
            clean=codec.observe(batch["mel100"],torch.ones_like(masks["visible"])); clear_masks={**masks,"unknown":torch.zeros_like(masks["unknown"])}
            cp=predict(model,kind,clean,batch["mouth"],clear_masks,threshold,torch.full((len(ids),),data["mouth"] is not None,device=batch["mouth"].device,dtype=torch.bool))["prediction"]
            totals.setdefault("clean_mse",[]).append((cp-clean["target"]).square().mean((1,2)).float().cpu().numpy()); actuals.extend(data["rows"][i] for i in ids)
    flat={k:np.concatenate(v) for k,v in totals.items()}; metrics={k:float(v.mean()) for k,v in flat.items()}
    for mode in modes: metrics[mode+"_word_accuracy"]=float(flat[mode+"_word_correct"].sum()/flat[mode+"_word_frames"].sum()) if flat[mode+"_word_frames"].sum()>0 else None
    value={"samples":len(actuals),"excluded_short":len(data["rows"])-len(actuals),"gap_packets":gap,"metrics":metrics,"selection_score":metrics["correct_hole_mse"]}
    if detailed: value["per_clip"]=[{"sample_id":r["sample_id"],"actor":r["actor"],**{k:float(v[i]) for k,v in flat.items()}} for i,r in enumerate(actuals)]
    return value


class EpochSampler:
    """Shuffle complete clip pools without replacement before beginning a new pass."""
    def __init__(self,data,rng,candidate=False):
        self.rng=rng; self.pool=np.where(data["lengths"]>=(50 if candidate else 32))[0]; self.order=rng.permutation(self.pool); self.position=0; self.epochs=0
    def take(self,n):
        if self.position+n>len(self.order):
            tail=self.order[self.position:]; needed=n-len(tail); self.order=self.rng.permutation(self.pool); self.position=needed; self.epochs+=1; return np.r_[tail,self.order[:needed]]
        ids=self.order[self.position:self.position+n]; self.position+=n; return ids
    def state(self): return {"order":self.order,"position":self.position,"epochs":self.epochs}


def train(cfg,run,kind,data,codec,sources,max_eval=None):
    root=Path(cfg["artifacts"]); out=root/run["id"]; out.mkdir(parents=True,exist_ok=True)
    if (out/"complete.json").exists(): return
    model,expert_cfg,expert_run,words=initialize(cfg,run,kind); tc=cfg["training"]; steps=tc["base_steps"] if kind=="base" else tc["candidate_steps"]; rng=np.random.default_rng(run["seed"])
    params=[p for p in model.parameters() if p.requires_grad]; groups=[{"params":[p for name,p in model.named_parameters() if p.requires_grad and (name.startswith("expert.world") if kind=="base" else False)],"lr":tc["core_lr"],"base_lr":tc["core_lr"]},{"params":[p for name,p in model.named_parameters() if p.requires_grad and not(kind=="base" and name.startswith("expert.world"))],"lr":tc["lr"],"base_lr":tc["lr"]}]
    optimizer=torch.optim.AdamW([g for g in groups if g["params"]],weight_decay=.01); sets=[data["train"],data["crema_train"]]+([data["libri_train"]] if kind=="base" else []); sizes=[24,4,4] if kind=="base" else [24,8]; samplers=[EpochSampler(d,rng,kind!="base") for d in sets]
    counts={"total":sum(p.numel() for p in model.parameters()),"trainable":sum(p.numel() for p in params),"frozen":sum(p.numel() for p in model.parameters() if not p.requires_grad)}
    provenance={"config_hash":stable_hash(cfg),"sources":sources,"manifests":{s:d["manifest_sha256"] for s,d in data.items()},"counts":counts,"historical_expert":cfg["experts"][str(run["seed"])],"gpu":torch.cuda.get_device_name()}
    write_json(out/"config.json",{"matrix":cfg,"run":run,"kind":kind,"provenance":provenance}); curves=[]; losses=[]; best=float("inf"); started=time.perf_counter(); consecutive=0; stopped=False
    def checkpoint(step,with_optimizer=True):
        ck={"model":model.state_dict(),"step":step,"config":cfg,"run":run,"kind":kind,"words":words,"expert_cfg":expert_cfg,"expert_run":expert_run,"provenance":provenance}
        if with_optimizer: ck.update({"optimizer":optimizer.state_dict(),"numpy_rng":rng.bit_generator.state,"samplers":[s.state() for s in samplers],"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()})
        return ck
    for step in range(steps+1):
        if step%tc["eval_every"]==0 or step==steps:
            val=evaluate(model,kind,codec,data["validation"],max_examples=max_eval); score=val["selection_score"]; old=best
            curves.append({"step":step,"validation":val,"training_loss":float(np.mean(losses)) if losses else None,"seconds":time.perf_counter()-started}); losses=[]
            if score<best: best=score; torch.save(checkpoint(step),out/"best.pt")
            consecutive=consecutive+1 if score>old*(1+tc["overfit_relative_rise"]) else 0
            write_json(out/"curves.json",curves); print(json.dumps({"run":run["id"],"step":step,"validation_mse":score,"seconds":time.perf_counter()-started}),flush=True)
            if consecutive>=tc["overfit_patience"]: stopped=True; break
        if step==steps: break
        batches=[]; selections=[]
        for d,n,sampler in zip(sets,sizes,samplers):
            ids=sampler.take(n); margin=8 if kind!="base" else 0; starts=np.array([rng.integers(margin,int(d["lengths"][i])-31-margin) for i in ids]); batches.append(crop(d,ids,starts)); selections.append((d,ids,starts))
        whole={k:torch.cat([b[k] for b in batches]) for k in batches[0]}; n=len(whole["mel100"]); lengths=rng.choice(cfg["gaps"]["training_lengths"],n,p=cfg["gaps"]["training_probabilities"]); starts=np.array([rng.integers(2,32-int(h)-1) for h in lengths]); gains=rng.uniform(-8,8,n).astype(np.float32); eq=rng.uniform(-4,4,(n,4)).astype(np.float32)
        y=codec.augment(whole["mel100"],torch.as_tensor(gains,device="cuda"),torch.as_tensor(eq,device="cuda")); masks=gap_masks(torch.as_tensor(starts,device="cuda"),torch.as_tensor(lengths,device="cuda")); obs=codec.observe(y,masks["visible"]); labels=whole["labels"]
        mouth=whole["mouth"]
        if run.get("arm")=="shuffled": mouth=torch.cat([control_mouth(d,ids,ss,b,"wrong") for (d,ids,ss),b in zip(selections,batches)])
        offsets=rng.choice(cfg["calibration"]["training_shift_sizes"],n)*rng.choice([-1,1],n); wrong=rng.integers(0,2,n).astype(bool)
        model.train(); optimizer.zero_grad(set_to_none=True); factor=min(1.,(step+1)/tc["warmup"])*(.1+.9*.5*(1+math.cos(math.pi*step/steps)))
        for g in optimizer.param_groups: g["lr"]=g["base_lr"]*factor
        with torch.autocast("cuda",dtype=torch.bfloat16):
            result=predict(model,kind,obs,mouth,masks); loss=repair_loss(result["prediction"],obs["target"],masks)+run["unit_weight"]*unit_loss(result["unit_logits"],labels)
            if kind=="base":
                loss=loss+tc["observed_content_weight"]*content_loss(result["content_readout"],obs["target"],masks["visible"])
                if step%4==0:
                    supplied=model.decode(obs,result["field"],result["unit_logits"],labels); loss=loss+tc["timed_text_weight"]*repair_loss(supplied,obs["target"],masks)
            else:
                loss=loss+run["unit_weight"]*tc["hint_word_weight"]*unit_loss(result["hint_unit_logits"],labels)
                if run["arm"]!="audio":
                    # Same random intervention plan within a seed. Real shifts
                    # have valid8-packet margins; only observed audio matches.
                    negatives=[]; at=0
                    for (d,ids,ss),b in zip(selections,batches):
                        count=len(ids); shifted=crop(d,ids,ss+offsets[at:at+count])["mouth"]; other=control_mouth(d,ids,ss,b,"wrong"); pick=torch.as_tensor(wrong[at:at+count],device="cuda")[:,None,None,None]
                        negatives.append(torch.where(pick,other,shifted)); at+=count
                    nm=torch.cat(negatives); nh=model.branch.hint(result["base_field"],nm); negative=model.branch.synchronize(obs["audio"],nh,obs["visible"])
                    sync=F.cross_entropy(result["offset_logits"],torch.full((n,),8,device="cuda",dtype=torch.long))+F.binary_cross_entropy_with_logits(result["match_logits"],torch.ones(n,device="cuda"))+F.binary_cross_entropy_with_logits(negative["match_logits"],torch.zeros(n,device="cuda"))
                    paired=torch.as_tensor(~wrong,device="cuda")
                    if paired.any(): sync=sync+F.cross_entropy(negative["offset_logits"][paired],torch.as_tensor(8-offsets,device="cuda").long()[paired])
                    loss=loss+tc["sync_weight"]*sync
        if not torch.isfinite(loss): raise RuntimeError("Nonfinite content/sync loss")
        loss.backward()
        if step==0:
            counts["graph_connected_trainable"]=sum(p.numel() for p in params if p.grad is not None); write_json(out/"counts.json",counts)
        torch.nn.utils.clip_grad_norm_(params,1.); optimizer.step(); losses.append(float(loss.detach()))
    torch.save(checkpoint(step,False),out/"last.pt"); selected=torch.load(out/"best.pt",map_location="cpu",weights_only=False)
    write_json(out/"complete.json",{"run":run,"kind":kind,"selected_step":selected["step"],"selected_validation_mse":best,"updates":step,"positive_windows":step*32,"extra_negative_windows":step*32 if kind!="base" and run["arm"]!="audio" else 0,"overfit_stop":stopped,"wall_seconds":time.perf_counter()-started,"counts":counts,"checkpoint_sha256":file_sha(out/"best.pt"),"last_sha256":file_sha(out/"last.pt")})
    del model,optimizer,selected; torch.cuda.empty_cache()


def rejection_threshold(correct,shifted,config):
    """Quality[B] and two shifted qualities[2,B] -> calibrated hard threshold.

    Offset!=0 is encoded as−1. Each polarity must meet the false-accept cap.
    Low aligned coverage remains a declared failure, not successful abstention.
    """
    grid=np.linspace(0,1,config["threshold_grid_points"])
    feasible=[float(t) for t in grid if (shifted>=t).mean(1).max()<=config["max_shift_acceptance"]]
    threshold=min(feasible) if feasible else 1.0001
    return {"threshold":threshold,"aligned_coverage":float((correct>=threshold).mean()),"shift_acceptance":float((shifted>=threshold).mean(1).max()),
        "shift_acceptance_by_direction":(shifted>=threshold).mean(1).tolist(),"coverage_gate_passes":bool((correct>=threshold).mean()>=config["required_correct_coverage"])}


def calibrate(cfg,codec):
    root=Path(cfg["artifacts"]); selection={r["id"]:json.loads((root/r["id"]/"complete.json").read_text()) for r in cfg["base_runs"]+cfg["candidate_runs"]}; write_json(root/"selection.json",selection)
    if (root/"calibration.json").exists():
        previous=json.loads((root/"calibration.json").read_text())
        for name,value in previous.items(): assert value["checkpoint_sha256"]==selection[name]["checkpoint_sha256"]
        return previous
    if any(root.glob("EXP-T2-*/test_gap*.json")): raise ValueError("Cannot tune calibration after final test evaluation")
    data=load_data(cfg,"calibration"); calibration={}
    for run in cfg["candidate_runs"]:
        if run["arm"]=="audio": continue
        model,ck=load_model(root/run["id"]/"best.pt"); metrics=evaluate(model,"candidate",codec,data,detailed=True); rows=metrics["per_clip"]
        correct=np.array([r["correct_quality"] if r["correct_offset"]==0 else -1 for r in rows]); shifted=np.array([[r["shift6_quality"] if r["shift6_offset"]==0 else -1 for r in rows],[r["shift-6_quality"] if r["shift-6_offset"]==0 else -1 for r in rows]])
        calibration[run["id"]]={**rejection_threshold(correct,shifted,cfg["calibration"]),"samples":len(rows),"source":"Independent calibration people22–24, no test targets","checkpoint_sha256":file_sha(root/run["id"]/"best.pt")}; write_json(root/run["id"]/"calibration.json",{**calibration[run["id"]],"per_clip":rows}); del model; torch.cuda.empty_cache()
    write_json(root/"calibration.json",calibration)
    # Freeze actual listening IDs before the first final signal score.
    rows=json.loads((Path(cfg["corpus"])/"test.json").read_text()); fixed=[]
    for actor in sorted({r["actor"] for r in rows}):
        pool=sorted([r for r in rows if r["actor"]==actor and r["frames"]>=50],key=lambda r:r["sample_id"])
        for index in (0,(len(pool)-1)//3,2*(len(pool)-1)//3,len(pool)-1): fixed.append(pool[index]["sample_id"])
    assert len(fixed)==32 and len(set(fixed))==32
    write_json(root/"listening_ids.json",fixed)
    return calibration


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/t2/content_sync.json"); p.add_argument("--stage",choices=("train","calibrate","evaluate","smoke"),required=True); a=p.parse_args(); cfg=json.loads(Path(a.config).read_text()); root=Path(cfg["artifacts"]); root.mkdir(parents=True,exist_ok=True); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True; codec=get_codec(cfg)
    card=json.loads((Path(cfg["corpus"])/"dataset_card.json").read_text())
    assert card["vocabulary"]==json.loads((Path(cfg["corpus"])/"vocabulary.json").read_text())
    if a.stage in {"calibrate","evaluate"}:
        for name,digest in json.loads((root/"source_hashes.json").read_text()).items():
            if file_sha(name)!=digest: raise ValueError("Registered source changed before evaluation: "+name)
    if a.stage=="calibrate": calibrate(cfg,codec); return
    if a.stage=="evaluate":
        calibration=json.loads((root/"calibration.json").read_text()); selection=json.loads((root/"selection.json").read_text())
        for split in ("test","test_seen","train_person_holdout","crema_test","libri_test"):
            data=load_data(cfg,split)
            for run in cfg["base_runs"]+cfg["candidate_runs"]:
                kind="base" if run in cfg["base_runs"] else "candidate"; model,_=load_model(root/run["id"]/"best.pt"); threshold=calibration[run["id"]]["threshold"] if run["id"] in calibration else None
                for gap in ((4,8,12) if split=="test" else (8,)):
                    value=evaluate(model,kind,codec,data,gap,threshold,detailed=True); value.update({"run":run,"checkpoint_sha256":selection[run["id"]]["checkpoint_sha256"],"split":split,"threshold":threshold,"manifest_sha256":data["manifest_sha256"]}); write_json(root/run["id"]/(split+f"_gap{gap}.json"),value)
                del model; torch.cuda.empty_cache(); print(json.dumps({"evaluated":run["id"],"split":split}),flush=True)
            del data; torch.cuda.empty_cache()
        return
    data={s:load_data(cfg,s) for s in ("train","validation","crema_train","libri_train")}
    if a.stage=="smoke":
        cfg["artifacts"]="/tmp/slwm_content_sync_disposable_smoke"; cfg["training"]={**cfg["training"],"base_steps":2,"candidate_steps":2,"eval_every":2,"warmup":1}
        for run,kind in [(cfg["base_runs"][1],"base")]+[(cfg["candidate_runs"][i],"candidate") for i in (0,2,4,5)]: train(cfg,run,kind,data,codec,{},max_eval=16)
        print(json.dumps({"disposable_smoke_complete":True}),flush=True); return
    names=(a.config,cfg["protocol"],cfg["data_config"],"models/slwm_content_sync.py","training/content_sync_train.py","training/content_sync_common.py","training/grid_content_data.py","models/slwm_content.py","models/slwm_restoration.py","models/slwm_motion.py","models/slwm_generation.py","models/slwm_video_repair.py","training/t2_train_latents.py","scripts/train/run_content_sync_fedora.sh","tests/test_content_sync_scale.py")
    sources={}
    for name in names:
        destination=root/"source_snapshot"/name; destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists() and file_sha(destination)!=file_sha(name): raise ValueError("Registered content source changed")
        if not destination.exists(): shutil.copyfile(name,destination)
        sources[name]=file_sha(name)
    write_json(root/"source_hashes.json",sources)
    for run in cfg["base_runs"]: train(cfg,run,"base",data,codec,sources)
    for run in cfg["candidate_runs"]: train(cfg,run,"candidate",data,codec,sources)


if __name__=="__main__": main()
