"""Bounded GPU-lock slices, matched data calls and reusable G2 artifacts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import gc
import json
import math
from pathlib import Path
import shutil
import os
import importlib.metadata
import subprocess
import time
import numpy as np
import torch
from models.predictive_world import PredictiveWorld
from training.general_batch import schedule,tensor_tree
from training.predictive_data import Sources
from training.predictive_losses import loss as batch_loss
from training.general_sources import write
from training.research_sources import file_sha

CONFIG="configs/t3/predictive_world.json"
PROTOCOL="docs/research/predictive_world_protocol_20261106.md"


def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def device_tree(value,device):
    if isinstance(value,torch.Tensor):return value.to(device,non_blocking=True)
    if isinstance(value,dict):return {k:device_tree(v,device) for k,v in value.items()}
    return value


def atomic_save(path,value):
    temporary=path.with_name(path.name+".tmp")
    with temporary.open('wb') as stream:torch.save(value,stream)
    temporary.replace(path)


def atomic_link(source,target):
    temporary=target.with_name(target.name+'.link.tmp')
    if temporary.exists():temporary.unlink()
    os.link(source,temporary);temporary.replace(target)


def source_guard(cfg):
    root=Path(cfg["root"]);root.mkdir(parents=True,exist_ok=True)
    names=(CONFIG,PROTOCOL,"configs/t3/predictive_readouts.json","docs/research/predictive_world_evaluation_20261106.md","models/predictive_world.py","models/slwm_general.py","training/predictive_train.py","training/predictive_losses.py","training/predictive_data.py","training/predictive_robot.py","training/predictive_reclock.py","training/predictive_proto.py","training/predictive_decoder_audit.py","training/predictive_eval.py","training/predictive_readouts.py","training/predictive_readout_data.py","training/predictive_features.py","training/predictive_har.py","training/predictive_dynamics_eval.py","training/predictive_alignment.py","training/predictive_speech_retention.py","training/predictive_live.py","training/predictive_report.py","exploration/predictive_infer.py","training/general_batch.py","training/general_data.py","training/general_dataset.py","training/general_sources.py","training/general_losses.py","training/general_frozen_features.py","training/general_readout_data.py","training/general_consequences.py","training/general_alignment_eval.py","training/general_robot_metadata.py","training/general_droid_sample.py","training/action_data_probe.py","training/research_sources.py","training/temporal_masks.py","tests/test_predictive_world.py","tests/test_predictive_losses.py","tests/test_predictive_robot.py","tests/test_predictive_readouts.py","tests/test_predictive_live.py","tests/test_predictive_source_losses.py","tests/test_predictive_checkpoint.py","tests/test_predictive_data.py","tests/test_predictive_evaluation.py","tests/test_predictive_proto.py","training/predictive_pipeline.py")
    sources={name:file_sha(name) for name in names}
    for name,digest in sources.items():
        target=root/"source_snapshot"/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert file_sha(target)==digest,"G2 source changed after study fitting: "+name
        else:shutil.copyfile(name,target)
    versions={name:importlib.metadata.version(name) for name in ('torch','numpy','scipy','tiktoken','h5py','protobuf')}
    environment=root/'environment.json'
    if environment.exists():assert json.loads(environment.read_text())['packages']==versions,'Training dependencies changed'
    else:write(environment,{'packages':versions,'cuda':torch.version.cuda,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'tracked_source_dirty':bool(subprocess.check_output(['git','diff','--name-only'],text=True).strip())})
    write(root/"source_hashes.json",sources);return sources


def initialize(cfg,run):
    names=json.loads((Path(cfg["data_root"])/"speech/phone_vocabulary.json").read_text())
    return PredictiveWorld(variant=run["variant"],width=run["width"],layers=run["layers"],heads=cfg["heads"],slots=cfg["slots"],memory=cfg["memory"],phone_count=len(names),seed=run["seed"],action_history=run.get("action_history",True),bypass=run.get("bypass",False))


def allowed(run,family):return not run.get("only_families") or family in run["only_families"]


def validate(model,teacher,source,cfg,run):
    records={};model.eval()
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        for index,family in enumerate(cfg["family_schedule_100"]):
            if not allowed(run,family):continue
            values=[]
            for batch_index in range(4):
                batch=source.make(family,10_000_000+index*10000+batch_index,9001,"cuda",16)
                _,value=batch_loss(model,teacher,batch,run,cfg,batch_index,True);values.append(value)
            records[family]={key:sum(row.get(key,0) for row in values)/len(values) for key in set.union(*(set(row) for row in values))}
    return {"selection":sum(v["selection"] for v in records.values())/len(records),"families":records}


def entry(cfg,run):
    folder=Path(cfg["root"])/run["id"];folder.mkdir(parents=True,exist_ok=True)
    path=folder/"resume.pt"
    if path.exists():
        saved=torch.load(path,map_location="cpu",weights_only=False,mmap=True);state=copy.deepcopy(saved["state"]);del saved
    else:state={"position":0,"optimizer_updates":0,"best":float("inf"),"selected_position":0,"rises":0,"finished":False,"training_seconds":0.,"compute_seconds":0.,"data_seconds":0.,"validation_seconds":0.,"family_calls":{},"active_parameter_names":[]}
    return {"run":run,"folder":folder,"model":None,"teacher":None,"optimizer":None,"state":state,"pending":None}


def persist(item,cfg,sources,validation,improved,milestone):
    folder=item["folder"];state=copy.deepcopy(item["state"]);model_state=item["model"].state_dict();teacher_state=item["teacher"].state_dict();optimizer=item["optimizer"]
    inference={"model":model_state,"state":state,"run":item["run"],"config":cfg,"sources":sources,"dataset_contract_sha256":file_sha(Path(cfg["dataset_manifest"])),"normalizers":dict(np.load(cfg['normalizers'])),'phone_count':len(json.loads((Path(cfg['data_root'])/'speech/phone_vocabulary.json').read_text()))}
    def save():
        atomic_save(folder/"resume.pt",{**inference,"teacher":teacher_state,"optimizer":optimizer})
        paths=[]
        if improved:paths.append(folder/'best.pt')
        if milestone:paths.append(folder/('position_'+str(state['position'])+'.pt'))
        if state['finished']:paths.append(folder/'endpoint.pt')
        if paths:
            atomic_save(paths[0],inference)
            for target in paths[1:]:atomic_link(paths[0],target)
        if state["finished"]:
            write(folder/"complete.json",{"run":item["run"],**state,"parameters":sum(v.numel() for v in model_state.values() if v.dtype!=torch.bool),"endpoint_sha256":file_sha(folder/"endpoint.pt"),"best_sha256":file_sha(folder/"best.pt"),"resume_sha256":file_sha(folder/"resume.pt"),"target_reached":state["position"]==cfg["updates"]})
    return save


def train_slice(item,source,selection,cfg,target,prefetch):
    state=item["state"];run=item["run"];begin=state["position"];boundary=((begin//cfg["validation_every"])+1)*cfg["validation_every"]
    end=min(begin+cfg["slice_updates"],target,boundary,cfg["updates"]);positions=[p for p in range(begin,end) if allowed(run,schedule(cfg,p))]
    model=initialize(cfg,run).cuda();teacher=copy.deepcopy(model).eval().requires_grad_(False);opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],betas=(.9,.95),weight_decay=cfg["weight_decay"],fused=True)
    path=item["folder"]/"resume.pt"
    if path.exists():
        saved=torch.load(path,map_location="cpu",weights_only=False,mmap=True);model.load_state_dict(saved["model"]);teacher.load_state_dict(saved["teacher"]);opt.load_state_dict(saved["optimizer"]);del saved
    losses=[];active=set(state["active_parameter_names"]);torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    pending=prefetch.submit(source.make,schedule(cfg,positions[0]),positions[0]*cfg["accumulation"],run["seed"],"cpu") if positions else None
    with (item["folder"]/"exposure.jsonl").open("a") as exposure:
        for index,position in enumerate(positions):
            model.train();opt.zero_grad(set_to_none=True);micro_metrics=[]
            lr=cfg["learning_rate"]*min(1.,(position+1)/cfg["warmup_updates"])
            if position>=cfg["warmup_updates"]:
                fraction=(position-cfg["warmup_updates"])/(cfg["updates"]-cfg["warmup_updates"])
                lr=cfg["minimum_lr"]+.5*(cfg["learning_rate"]-cfg["minimum_lr"])*(1+math.cos(math.pi*fraction))
            for group in opt.param_groups:group["lr"]=lr
            for micro in range(cfg["accumulation"]):
                data_start=time.perf_counter();batch=device_tree(pending.result(),"cuda")
                if micro+1<cfg["accumulation"]:
                    pending=prefetch.submit(source.make,schedule(cfg,position),position*cfg["accumulation"]+micro+1,run["seed"],"cpu")
                elif index+1<len(positions):
                    p=positions[index+1];pending=prefetch.submit(source.make,schedule(cfg,p),p*cfg["accumulation"],run["seed"],"cpu")
                torch.cuda.synchronize();state["data_seconds"]+=time.perf_counter()-data_start;compute=time.perf_counter()
                with torch.autocast("cuda",dtype=torch.bfloat16):objective,metrics=batch_loss(model,teacher,batch,run,cfg,position,False)
                assert torch.isfinite(objective);(objective/cfg["accumulation"]).backward()
                torch.cuda.synchronize();state["compute_seconds"]+=time.perf_counter()-compute;micro_metrics.append(metrics)
                exposure.write(json.dumps({"position":position,"micro":micro,"family":batch["family"],"ids":batch["ids"],"context_packets":batch["obs"]["time"].shape[1],"examples":len(batch["ids"]),"physical_observed_seconds":batch["physical_observed_seconds"]})+"\n")
                del batch,objective
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["gradient_clip"]);assert torch.isfinite(norm)
            active.update(name for name,p in model.named_parameters() if p.grad is not None);opt.step()
            with torch.no_grad():
                torch._foreach_mul_(list(teacher.parameters()),cfg["ema_decay"]);torch._foreach_add_(list(teacher.parameters()),list(model.parameters()),alpha=1-cfg["ema_decay"])
            state["optimizer_updates"]+=1
            family=schedule(cfg,position);state["family_calls"][family]=state["family_calls"].get(family,0)+cfg["accumulation"]
            losses.append(sum(v["objective"] for v in micro_metrics)/len(micro_metrics))
    state["training_seconds"]+=time.perf_counter()-started;state["position"]=end;state["active_parameter_names"]=sorted(active)
    validation=None;improved=False
    if end%cfg["validation_every"]==0 or end in cfg["milestones"]:
        start=time.perf_counter();validation=validate(model,teacher,selection,cfg,run);state["validation_seconds"]+=time.perf_counter()-start
        improved=validation["selection"]<state["best"];state["rises"]=state["rises"]+1 if validation["selection"]>state["best"]*1.1 else 0
        if improved:state["best"]=validation["selection"];state["selected_position"]=end
    state["finished"]=end==cfg["updates"] or state["rises"]>=2
    curves_path=item["folder"]/"curves.json";curves=json.loads(curves_path.read_text()) if curves_path.exists() else []
    record={"position":end,"optimizer_updates":state["optimizer_updates"],"mean_objective":sum(losses)/len(losses) if losses else None,"validation":validation,"training_seconds":state["training_seconds"],"compute_seconds":state["compute_seconds"],"data_seconds":state["data_seconds"],"validation_seconds":state["validation_seconds"],"peak_bytes":torch.cuda.max_memory_allocated(),"family_calls":state["family_calls"].copy()};curves.append(record);write(curves_path,curves)
    opt.zero_grad(set_to_none=True);item["optimizer"]=cpu_tree(opt.state_dict());item["model"]=model.cpu();item["teacher"]=teacher.cpu();del opt,model,teacher
    print(json.dumps({"run":run["id"],**record}),flush=True)
    return validation,improved,end in cfg["checkpoint_milestones"]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("smoke","screen","train"),required=True);args=parser.parse_args();cfg=json.loads(Path(CONFIG).read_text());torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    contract=Path(cfg["dataset_manifest"]);assert contract.exists(),"Freeze the complete data contract before fitting";source=Sources(cfg,"train");selection=Sources(cfg,"selection")
    data=json.loads(contract.read_text());assert data["configuration_sha256"]==file_sha(CONFIG)
    assert data["normalizers_sha256"]==file_sha(cfg["normalizers"])
    if args.stage=="smoke":
        with open("/tmp/slwm_research_gpu.lock","a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);model=initialize(cfg,cfg["runs"][0]).cuda();teacher=copy.deepcopy(model).eval().requires_grad_(False)
            records=[]
            for index,family in enumerate(cfg["family_schedule_100"]):
                model.zero_grad(set_to_none=True);batch=source.make(family,739100+index,101,"cuda")
                with torch.autocast("cuda",dtype=torch.bfloat16):loss,metrics=batch_loss(model,teacher,batch,cfg["runs"][0],cfg,index)
                loss.backward();assert torch.isfinite(loss);assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());records.append({"family":family,"metrics":metrics,"parameters":sum(p.numel() for p in model.parameters()),"no_optimizer_step":True});print(json.dumps(records[-1]),flush=True)
            write(Path(cfg["root"])/"real_data_gradient_preflight.json",records)
        return
    sources=source_guard(cfg);items=[entry(cfg,run) for run in cfg["runs"]]
    # Main representations get an early512-position screen; controls retain the
    # exact same target-family calls and all comparisons freeze before final tests.
    stages=[(512,items[:4])] if args.stage=='screen' else [(3000,items[:4]),(3000,items[4:]),(6000,items),(cfg["updates"],items)]
    with ThreadPoolExecutor(max_workers=1) as prefetch,ThreadPoolExecutor(max_workers=2) as saves:
        for target,current in stages:
            while any(not item["state"]["finished"] and item["state"]["position"]<target for item in current):
                for item in current:
                    if item["state"]["finished"] or item["state"]["position"]>=target:continue
                    if item["pending"] is not None:item["pending"].result()
                    with open("/tmp/slwm_research_gpu.lock","a") as lock:
                        fcntl.flock(lock,fcntl.LOCK_EX);validation,improved,milestone=train_slice(item,source,selection,cfg,target,prefetch);gc.collect();torch.cuda.empty_cache()
                    # Capture the completed CPU tensors/state before releasing
                    # the item. Only file I/O belongs in the background worker.
                    save_job=persist(item,cfg,sources,validation,improved,milestone)
                    item["pending"]=saves.submit(save_job)
                    # Keep only scalar progress between runs. Checkpoint tensors
                    # stay alive in their bounded writer, leaving RAM for signal
                    # caches instead of13 resident optimizer/EMA copies.
                    item["model"]=None;item["teacher"]=None;item["optimizer"]=None
        for item in items:
            if item["pending"] is not None:item["pending"].result()
    if args.stage=='screen':
        write(Path(cfg['root'])/'screen_complete.json',{'runs':{i['run']['id']:i['state'] for i in items[:4]},'source_hashes':sources,'final_tests_opened':False});return
    for original in cfg['runs']:
        if original['id'] not in ('G2-201','G2-202','G2-205','G2-206'):continue
        run={**original,'id':'RND-'+original['id'].split('-')[-1],'untrained':True};folder=Path(cfg['root'])/run['id'];folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists():continue
        model=initialize(cfg,run)
        atomic_save(folder/'endpoint.pt',{'model':model.state_dict(),'config':cfg,'run':run,'sources':sources,'phone_count':model.phone_count,'normalizers':dict(np.load(cfg['normalizers']))})
        write(folder/'complete.json',{'run':run,'optimizer_updates':0,'parameters':sum(p.numel() for p in model.parameters()),'endpoint_sha256':file_sha(folder/'endpoint.pt'),'untrained_control':True});del model;gc.collect()
    write(Path(cfg["root"])/"training_complete.json",{"runs":{i["run"]["id"]:i["state"] for i in items},"source_hashes":sources,"all_weights_frozen":True,"final_tests_opened":False})


if __name__=="__main__":main()
