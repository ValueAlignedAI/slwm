"""Frozen-world active sensing: train-only counterfactual risks, continuous replay."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch import nn
from training.broad_batch import StreamCorpus
from training.broad_stream_train import load_model
from training.research_sources import file_sha
from training.stream_codec import PACKET_SECONDS
from training.t2_real_av import write_json

CONFIG="configs/policy/stream_sensing.json"
BITS=np.asarray([[1,0],[0,1],[1,1],[0,0]],np.float32)
STRATEGIES=("learned","listen","look","both","wait","alternating","periodic")


class RiskActor(nn.Module):
    def __init__(self,features,hidden=128,dropout=.1):
        super().__init__(); self.net=nn.Sequential(nn.Linear(features,hidden),nn.SiLU(),nn.Dropout(dropout),nn.Linear(hidden,8))
    def forward(self,x): return self.net(x).reshape(-1,4,2)


def goal_error(errors,goal):
    return errors[...,0] if goal=="audio" else .2*errors[...,1] if goal=="video" else errors[...,0]+.2*errors[...,1]


def choose(errors,goal,cost):
    return (goal_error(errors,goal)+cost*torch.as_tensor(BITS.sum(1),device=errors.device)).argmin(-1)


def features(out,bits):
    return torch.cat((out["field"].float().mean(1),out["memory"].float().mean(1),out["risk"].float().mean(1,keepdim=True),bits),-1)


def weight_digest(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items(): h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def episodes(data,starts,cfg):
    name=next(iter(data.rows)); result=[]; needed=(cfg["episode_chunks"]+1)*32+10
    for i,row in enumerate(data.rows[name]):
        for seconds in starts:
            at=int(round(seconds/PACKET_SECONDS)); n=min(needed,row["length"]-at)
            if n<needed: raise ValueError("Registered episode exceeds common decoded support: "+row["id"])
            result.append({"name":name,"index":i,"id":row["id"],"actor":row["actor"],"start":at,"packets":needed,"start_seconds":at*PACKET_SECONDS})
    return result


def sequence(data,plans,device):
    audio=[]; video=[]
    for p in plans:
        row=data.rows[p["name"]][p["index"]]; start=p["start"]; end=start+p["packets"]; mel,_=data.arrays[p["name"]]
        audio.append((np.asarray(mel[row["offset"]+start:row["offset"]+end],np.float32)-data.mean)/data.scale); video.append(np.asarray(data.videos[row["id"]][start:end],np.float32))
    return torch.as_tensor(np.stack(audio),device=device),torch.as_tensor(np.stack(video),device=device)


def observe(model,audio,video,at,bits,memory=None):
    a=audio[:,at:at+32]; v=video[:,at:at+32]; valid=torch.ones(a.shape[:2],device=a.device,dtype=torch.bool); zero=torch.zeros_like(valid)
    return model(a,v,torch.zeros_like(valid,dtype=torch.long),valid&bits[:,0:1].bool(),valid&bits[:,1:2].bool(),zero,valid,torch.zeros(len(a),64,device=a.device),memory)


def future_error(out,audio,video,at):
    begin=at+32+2; target_a=audio[:,begin:begin+8]; target_v=video[:,begin:begin+8]; future=out["future"][:,0]
    return torch.stack(((future[:,:,:400]-target_a).abs().mean((1,2)),(future[:,:,400:]-target_v).square().mean((1,2))),-1)


def risk_masks(actors,cfg):
    people=sorted(set(actors),key=lambda p:hashlib.sha256(("P1-RISK-"+p).encode()).hexdigest()); holdout=people[:cfg["risk_holdout_people"]]; validation=np.isin(np.asarray(actors),holdout); assert validation.any() and (~validation).any(); return ~validation,validation,holdout


def collect(model,data,plans,cfg,device="cuda",return_actors=False):
    xs=[]; ys=[]; owners=[]; rng=np.random.default_rng(cfg["seed"]); amp=torch.autocast("cuda",dtype=torch.bfloat16) if device=="cuda" else nullcontext()
    with torch.inference_mode(),amp:
        for begin in range(0,len(plans),16):
            group=plans[begin:begin+16]; audio,video=sequence(data,group,device); b=len(group); bits=torch.ones(b,2,device=device); current=observe(model,audio,video,0,bits); aa=audio.repeat_interleave(4,0); vv=video.repeat_interleave(4,0); actions=torch.as_tensor(BITS,device=device).repeat(b,1)
            for tick in range(cfg["episode_chunks"]):
                xs.append(features(current,bits).cpu().numpy()); owners.extend(p["actor"] for p in group); at=(tick+1)*32; candidate=observe(model,aa,vv,at,actions,current["memory"].repeat_interleave(4,0)); error=future_error(candidate,aa,vv,at).reshape(b,4,2); ys.append(error.cpu().numpy()); selected=torch.as_tensor(rng.integers(4,size=b),device=device); ids=torch.arange(b,device=device)*4+selected; current={k:v[ids] for k,v in candidate.items()}; bits=actions[ids]
    result=(np.concatenate(xs),np.concatenate(ys)); return (*result,owners) if return_actors else result


def replay(model,actor,normalization,thresholds,data,plans,cfg,goal,strategy,device="cuda",paced=False):
    records=[]; amp=torch.autocast("cuda",dtype=torch.bfloat16) if device=="cuda" else nullcontext(); before=weight_digest(model); started=time.perf_counter()
    with torch.inference_mode(),amp:
        # Fixed, identical bootstrap. Subsequent decisions depend only on state.
        for begin in range(0,len(plans),1 if paced else 16):
            group=plans[begin:begin+(1 if paced else 16)]; audio,video=sequence(data,group,device); bits=torch.ones(len(group),2,device=device); current=observe(model,audio,video,0,bits); group_start=time.perf_counter()
            for tick in range(cfg["episode_chunks"]):
                tick_start=time.perf_counter(); x=features(current,bits); pred=actor((x-normalization[0])/normalization[1]).float().clamp(-12,8).exp()
                if strategy=="learned": selected=choose(pred,goal,cfg["sensor_cost"])
                else:
                    action={"listen":0,"look":1,"both":2,"wait":3,"alternating":tick%2,"periodic":2 if tick%2==0 else 3}[strategy]; selected=torch.full((len(group),),action,device=device,dtype=torch.long)
                bits=torch.as_tensor(BITS,device=device)[selected]; at=(tick+1)*32; current=observe(model,audio,video,at,bits,current["memory"]); actual=future_error(current,audio,video,at); predicted=pred[torch.arange(len(group),device=device),selected]; error=goal_error(actual,goal); risk=goal_error(predicted,goal); costs=cfg["sensor_cost"]*bits.sum(-1); commit=risk<=thresholds.get(goal,float("inf")); cpu=torch.stack((error,costs,risk,commit.float(),actual[:,0],actual[:,1],current["memory"].float().norm(dim=(1,2))),-1).cpu().numpy(); actions=selected.cpu().tolist(); elapsed=time.perf_counter()-tick_start
                for i,p in enumerate(group): records.append({"actor":p["actor"],"id":p["id"],"start_seconds":p["start_seconds"],"tick":tick,"source_received_until_seconds":p["start_seconds"]+(at+32)*PACKET_SECONDS,"future_target_start_seconds":p["start_seconds"]+(at+34)*PACKET_SECONDS,"strategy":strategy,"goal":goal,"action":cfg["actions"][actions[i]],"error":float(cpu[i,0]),"sensor_cost":float(cpu[i,1]),"total_cost":float(cpu[i,0]+cpu[i,1]),"predicted_error":float(cpu[i,2]),"committed":bool(cpu[i,3]),"audio_error":float(cpu[i,4]),"video_error":float(cpu[i,5]),"memory_norm":float(cpu[i,6]),"compute_seconds_batch":elapsed,"deadline_missed":elapsed>32*PACKET_SECONDS})
                if paced:
                    delay=(tick+1)*32*PACKET_SECONDS-(time.perf_counter()-group_start)
                    if delay>0: time.sleep(delay)
    assert weight_digest(model)==before,"Inference changed model weights"
    return records,{"weights_unchanged":True,"weights_sha256":before,"wall_seconds":time.perf_counter()-started,"source_phase_used":False,"world_weights_trainable":False,"physical_world_control":False}


def freeze(cfg):
    root=Path(cfg["artifacts"]); root.mkdir(parents=True,exist_ok=True); names=(CONFIG,cfg["protocol"],"training/stream_sensing.py","tests/test_stream_sensing.py")
    for name in names:
        target=root/"source_snapshot"/name; target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and file_sha(name)!=file_sha(target): raise ValueError("Frozen policy source changed: "+name)
        if not target.exists(): shutil.copyfile(name,target)
    write_json(root/"source_hashes.json",{n:file_sha(n) for n in names})


def fit(cfg):
    world_cfg=json.loads(Path("configs/t2/broad_stream.json").read_text()); root=Path(cfg["artifacts"]); train=StreamCorpus(world_cfg,"train"); train.rows={k:v for k,v in train.rows.items() if k.startswith("ami_")}; cal=StreamCorpus(world_cfg,"calibration"); training_plans=episodes(train,cfg["training_starts_seconds"],cfg); calibration_plans=episodes(cal,cfg["validation_starts_seconds"],cfg); write_json(root/"training_plans.json",training_plans); write_json(root/"calibration_plans.json",calibration_plans)
    for run in cfg["world_runs"]:
        out=root/run; out.mkdir(exist_ok=True)
        if (out/"frozen.json").exists(): continue
        path=Path(cfg["world_root"])/run/"best.pt"; model,ck=load_model(path); model.requires_grad_(False); before=weight_digest(model); start=time.perf_counter(); x,y,owners=collect(model,train,training_plans,cfg,return_actors=True); assert weight_digest(model)==before; fit_mask,hold_mask,held_people=risk_masks(owners,cfg); np.savez_compressed(out/"training_risks.npz",features=x,errors=y,actors=np.asarray(owners),fit_mask=fit_mask,hold_mask=hold_mask); mean=x[fit_mask].mean(0); scale=x[fit_mask].std(0).clip(.05); torch.manual_seed(cfg["seed"]); actor=RiskActor(x.shape[1],cfg["risk_hidden"],cfg["risk_dropout"]).cuda(); optimizer=torch.optim.AdamW(actor.parameters(),lr=cfg["risk_lr"]); xx=torch.as_tensor((x-mean)/scale,device="cuda"); yy=torch.as_tensor(np.log(y+1e-6),device="cuda"); fit_ids=np.flatnonzero(fit_mask); hold_ids=torch.as_tensor(np.flatnonzero(hold_mask),device="cuda"); rng=np.random.default_rng(cfg["seed"]); curves=[]; best=float("inf"); rises=0; selected=None
        for step in range(cfg["risk_steps"]):
            actor.train(); ids=torch.as_tensor(rng.choice(fit_ids,size=cfg["risk_batch"]),device="cuda"); loss=(actor(xx[ids])-yy[ids]).square().mean(); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            if (step+1)%cfg["risk_eval_every"]==0:
                actor.eval()
                with torch.inference_mode(): value=float((actor(xx[hold_ids])-yy[hold_ids]).square().mean())
                rises=rises+1 if value>best*1.05 else 0; curves.append({"step":step+1,"training_log_risk_mse":float(loss.detach()),"held_person_log_risk_mse":value})
                if value<best:
                    best=value; selected={"model":{k:v.detach().cpu().clone() for k,v in actor.state_dict().items()},"optimizer":optimizer.state_dict(),"step":step+1,"held_person_log_risk_mse":best,"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),"data_rng":rng.bit_generator.state,"mean":mean,"scale":scale,"config":cfg,"world_run":run}; torch.save(selected,out/"best_optimizer.pt")
                if rises>=2: break
        last_step=step+1; torch.save({"model":actor.state_dict(),"optimizer":optimizer.state_dict(),"step":last_step,"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),"data_rng":rng.bit_generator.state,"mean":mean,"scale":scale,"config":cfg,"world_run":run},out/"last_optimizer.pt"); selected=torch.load(out/"best_optimizer.pt",map_location="cpu",weights_only=False); actor.load_state_dict(selected["model"],strict=True); write_json(out/"fit.json",{"selected_step":selected["step"],"actual_steps":last_step,"early_stop":rises>=2,"fit_examples":int(fit_mask.sum()),"validation_examples":int(hold_mask.sum()),"validation_people":held_people,"normalization_fit_people":len(set(owners))-len(held_people),"best_held_person_log_risk_mse":best,"calibration_people_used_for_weight_selection":False,"test_read":False})
        actor.eval(); norm=(torch.as_tensor(mean,device="cuda"),torch.as_tensor(scale,device="cuda")); calibration=[]; thresholds={}
        for goal in cfg["goals"]:
            rows,audit=replay(model,actor,norm,{},cal,calibration_plans,cfg,goal,"learned"); thresholds[goal]=float(np.quantile([r["predicted_error"] for r in rows],cfg["calibration_coverage"])); calibration.extend(rows)
        torch.save({"model":actor.state_dict(),"features":x.shape[1],"hidden":cfg["risk_hidden"],"mean":mean,"scale":scale,"thresholds":thresholds,"world_checkpoint_sha256":file_sha(path),"world_run":run,"config":cfg,"id":cfg["policy_ids"][run]},out/"actor.pt"); write_json(out/"calibration.json",calibration); write_json(out/"curves.json",curves); write_json(out/"frozen.json",{"id":cfg["policy_ids"][run],"world_checkpoint_sha256":file_sha(path),"world_weights_unchanged":weight_digest(model)==before,"actor_sha256":file_sha(out/"actor.pt"),"best_optimizer_sha256":file_sha(out/"best_optimizer.pt"),"fit_sha256":file_sha(out/"fit.json"),"parameters":sum(p.numel() for p in actor.parameters()),"training_examples":int(fit_mask.sum()),"counterfactuals_per_example":4,"thresholds":thresholds,"seconds":time.perf_counter()-start,"test_read":False}); print(json.dumps({"policy_frozen":run,"seconds":time.perf_counter()-start}),flush=True); del model,actor,optimizer,xx,yy; torch.cuda.empty_cache()
    write_json(root/"selection.json",{r:json.loads((root/r/"frozen.json").read_text()) for r in cfg["world_runs"]})


def load_actor(root,run,device):
    ck=torch.load(root/run/"actor.pt",map_location="cpu",weights_only=False); actor=RiskActor(ck["features"],ck["hidden"],ck["config"]["risk_dropout"]); actor.load_state_dict(ck["model"],strict=True); actor.to(device).eval().requires_grad_(False); return actor,(torch.as_tensor(ck["mean"],device=device),torch.as_tensor(ck["scale"],device=device)),ck


def evaluate(cfg,paced=False):
    root=Path(cfg["artifacts"]); assert (root/"selection.json").exists(); data=StreamCorpus(json.loads(Path("configs/t2/broad_stream.json").read_text()),"test_ami"); plans=episodes(data,[cfg["test_start_seconds"]],cfg); write_json(root/"test_plans.json",plans); device="cpu" if paced else "cuda"
    for run in cfg["world_runs"][:1] if paced else cfg["world_runs"]:
        path=Path(cfg["world_root"])/run/"best.pt"; model,_=load_model(path,device); model.requires_grad_(False); actor,norm,ck=load_actor(root,run,device); assert file_sha(path)==ck["world_checkpoint_sha256"]
        for goal in cfg["goals"][:1] if paced else cfg["goals"]:
            for strategy in ("learned",) if paced else STRATEGIES:
                target=root/run/("paced_cpu.json" if paced else goal+"_"+strategy+".json")
                if target.exists(): continue
                rows,audit=replay(model,actor,norm,ck["thresholds"],data,plans[:1] if paced else plans,cfg,goal,strategy,device,paced); write_json(target,{"rows":rows,"audit":audit}); print(json.dumps({"policy_evaluated":run,"goal":goal,"strategy":strategy,"paced":paced}),flush=True)
        del model,actor; torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(); p.add_argument("--stage",choices=("fit","evaluate","paced"),required=True); a=p.parse_args(); torch.set_num_threads(4); cfg=json.loads(Path(CONFIG).read_text()); assert (Path(cfg["world_root"])/"selection.json").exists(); freeze(cfg)
    if a.stage=="paced": evaluate(cfg,True); return
    with Path("/tmp/slwm_research_gpu.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if a.stage=="fit": fit(cfg)
        else: evaluate(cfg)


if __name__=="__main__": main()
