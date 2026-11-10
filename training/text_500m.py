"""Matched S4-to-500M continuation; parent artifacts and old methods stay immutable."""
from __future__ import annotations
import argparse
import fcntl
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import time
import torch

from training.research_sources import file_sha
from training.scale_text_train import UniqueWindows,initialize,evaluate,generate
from training.text_fused_loss import fused_loss

CONFIG="configs/t1/fineweb_500m.json"


def atomic_json(path,value):
    path=Path(path); temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); temporary.replace(path)


def atomic_save(path,value):
    path=Path(path); temporary=path.with_name(path.name+".tmp")
    torch.save(value,temporary); temporary.replace(path)


def window_index(cfg,step):
    """New optimizer update -> next unused logical512-token source-window index."""
    return cfg["parent_tokens"]//cfg["sequence_length"]+step*cfg["global_batch"]


def learning_rate(step,cfg):
    warm=cfg["warmup_updates"]; total=cfg["additional_updates"]
    if step<warm:
        return cfg["start_lr"]+(cfg["peak_lr"]-cfg["start_lr"])*step/max(1,warm-1)
    phase=(step-warm)/max(1,total-1-warm)
    return cfg["floor_lr"]+(cfg["peak_lr"]-cfg["floor_lr"])*.5*(1+math.cos(math.pi*phase))


def freeze(cfg):
    root=Path(cfg["artifacts"]); root.mkdir(parents=True,exist_ok=True)
    names=(CONFIG,cfg["protocol"],cfg["parent_config"],"training/text_500m.py","training/text_fused_loss.py","training/scale_text_train.py","training/t1_torch_text.py","tests/test_text_500m.py")
    hashes={name:file_sha(name) for name in names}
    for name,digest in hashes.items():
        target=root/"source_snapshot"/name; target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and file_sha(target)!=digest: raise ValueError("Frozen S5 source changed: "+name)
        if not target.exists(): shutil.copyfile(name,target)
    atomic_json(root/"source_hashes.json",hashes); return hashes


def train_slice(cfg,run,base,sources,smoke=False):
    root=Path(cfg["artifacts"]); out=root/run["id"]; out.mkdir(exist_ok=True)
    if (out/"complete.json").exists() and not smoke: return
    parent=Path(cfg["parent_root"])/run["parent"]/"best.pt"
    assert file_sha(parent)==run["parent_sha256"],"Immutable parent differs"
    path=out/"resume.pt"; state=torch.load(path if path.exists() and not smoke else parent,map_location="cpu",weights_only=False)
    if not path.exists() or smoke: assert state["total_token_equivalent_step"]*16384==cfg["parent_tokens"]
    model=initialize(base,run["variant"],run["seed"]).cuda(); model.load_state_dict(state["model"],strict=True)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["peak_lr"],betas=(.9,.95),weight_decay=base["training"]["weight_decay"],fused=True)
    opt.load_state_dict(state["optimizer"]); torch.set_rng_state(state["torch_rng"]); torch.cuda.set_rng_state_all(state["cuda_rng"])
    begin=state.get("s5_updates",0); best=state.get("s5_best_validation",state.get("best_validation",float("inf")))
    selected=state.get("s5_best_updates",0); rises=state.get("s5_rises",0); elapsed=state.get("s5_training_validation_seconds",0.)
    parent_steps=state.get("s5_parent_optimizer_updates",int(next(iter(opt.state.values()))["step"].item()))
    parameters=sum(p.numel() for p in model.parameters()); del state; gc.collect()
    stream=UniqueWindows(Path(base["corpus"])/"train.tokens.bin",512,run["seed"])
    valset=UniqueWindows(Path(base["corpus"])/"validation.tokens.bin",512,901)
    if smoke:
        x,y=stream.batch(window_index(cfg,0),64,"cuda"); model.train(); opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16): loss=fused_loss(model,x,y)
        loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),base["training"]["gradient_clip"])
        assert torch.isfinite(loss) and torch.isfinite(norm)
        print(json.dumps({"s5_smoke":run["id"],"loss":float(loss),"parameters":parameters,"parent_sha256":file_sha(parent),"parent_unchanged":True}),flush=True)
        del model,opt,x,y,loss; gc.collect(); torch.cuda.empty_cache(); return
    curves=json.loads((out/"curves.json").read_text()) if (out/"curves.json").exists() else []
    milestones=[(value-cfg["parent_tokens"])//(64*512) for value in cfg["milestone_tokens"]]
    end=min(begin+cfg["slice_updates"],cfg["additional_updates"],min((x for x in milestones if x>begin),default=cfg["additional_updates"]))
    provenance={"parent_run":run["parent"],"parent_checkpoint_sha256":run["parent_sha256"],"source_hashes":sources,
        "parameters":parameters,"dataset_card_sha256":file_sha(Path(base["corpus"])/"dataset_card.json"),"optimizer_preserved":True,
        "input_window_sequence_preserved":True,"training_context":512,"global_batch":64,"warm_restart":True}
    atomic_json(out/"config.json",{"matrix":base,"continuation":cfg,"run":run,"provenance":provenance})
    started=time.perf_counter(); torch.cuda.reset_peak_memory_stats(); losses=[]
    for step in range(begin,end):
        model.train(); opt.zero_grad(set_to_none=True)
        for group in opt.param_groups: group["lr"]=learning_rate(step,cfg)
        x,y=stream.batch(window_index(cfg,step),64,"cuda")
        with torch.autocast("cuda",dtype=torch.bfloat16): loss=fused_loss(model,x,y)
        if not torch.isfinite(loss): raise RuntimeError("Nonfinite S5 loss")
        loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),base["training"]["gradient_clip"])
        if not torch.isfinite(norm): raise RuntimeError("Nonfinite S5 gradient")
        opt.step(); losses.append(float(loss.detach()))
    val=evaluate(model,valset,base); elapsed+=time.perf_counter()-started
    improved=val["nll"]<best; rises=rises+1 if val["nll"]>best*(1+cfg["overfit_relative_rise"]) else 0
    if improved: best=val["nll"]; selected=end
    tokens=cfg["parent_tokens"]+end*64*512
    record={"new_updates":end,"tokens":tokens,"validation":val,"training_nll":sum(losses)/len(losses),
        "training_validation_seconds":elapsed,"peak_gpu_bytes":torch.cuda.max_memory_allocated(),"last_lr":learning_rate(end-1,cfg)}
    curves.append(record); atomic_json(out/"curves.json",curves)
    saved={"model":model.state_dict(),"optimizer":opt.state_dict(),"config":base,"run":run,"step":parent_steps+end,
        "total_token_equivalent_step":tokens//16384,"provenance":provenance,"s5_updates":end,"s5_best_validation":best,
        "s5_best_updates":selected,"s5_rises":rises,"s5_parent_optimizer_updates":parent_steps,"s5_training_validation_seconds":elapsed,
        "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()}
    atomic_save(out/"resume.pt",saved)
    if improved: atomic_save(out/"best.pt",saved)
    elif not(out/"best.pt").exists():
        original=torch.load(parent,map_location="cpu",weights_only=False); original.update(run=run,provenance=provenance,s5_updates=0)
        atomic_save(out/"best.pt",original); del original
    if end in milestones: atomic_save(out/("tokens_"+str(tokens)+".pt"),{k:v for k,v in saved.items() if k!="optimizer"})
    if end==cfg["additional_updates"] or rises>=cfg["overfit_patience"]:
        atomic_save(out/"last.pt",{k:v for k,v in saved.items() if k!="optimizer"})
        atomic_json(out/"complete.json",{"run":run,"parameters":parameters,"tokens_seen":tokens,"new_updates":end,
            "actual_optimizer_updates":parent_steps+end,"selected_tokens":cfg["parent_tokens"]+selected*64*512,
            "selected_validation_nll":best,"overfit_stop":rises>=cfg["overfit_patience"],"target_reached":tokens==cfg["target_tokens"],
            "training_validation_seconds":elapsed,"checkpoint_sha256":file_sha(out/"best.pt"),"resume_sha256":file_sha(out/"resume.pt"),"last_sha256":file_sha(out/"last.pt")})
    print(json.dumps({"run":run["id"],**record}),flush=True)
    del model,opt,saved,x,y,loss; gc.collect(); torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--stage",required=True,choices=("smoke","train","evaluate")); args=parser.parse_args()
    cfg=json.loads(Path(CONFIG).read_text()); base=json.loads(Path(cfg["parent_config"]).read_text())
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
    assert importlib.metadata.version("liger-kernel")==cfg["fused_loss_version"]
    assert cfg["target_tokens"]==cfg["parent_tokens"]+cfg["additional_updates"]*64*512 and cfg["micro_batch"]==cfg["global_batch"]==64
    root=Path(cfg["artifacts"]); root.mkdir(parents=True,exist_ok=True)
    if args.stage=="smoke":
        with Path("/tmp/slwm_research_gpu.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for run in cfg["runs"][:2]: train_slice(cfg,run,base,{},True)
        return
    sources=freeze(cfg); card=json.loads((Path(base["corpus"])/"dataset_card.json").read_text())
    for name,digest in card["files"].items(): assert file_sha(Path(base["corpus"])/name)==digest
    if args.stage=="train":
        while any(not(root/r["id"]/"complete.json").exists() for r in cfg["runs"]):
            for run in cfg["runs"]:
                if (root/run["id"]/"complete.json").exists(): continue
                with Path("/tmp/slwm_research_gpu.lock").open("a") as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX); train_slice(cfg,run,base,sources)
        return
    assert all((root/r["id"]/"complete.json").exists() for r in cfg["runs"])
    class Fresh:
        def __init__(self): self.source=UniqueWindows(Path(base["corpus"])/"test.tokens.bin",512,902)
        def batch(self,index,batch,device): return self.source.batch(cfg["test_window_start"]+index,batch,device)
    testcfg={**base,"training":{**base["training"],"eval_batches":cfg["test_batches"]}}; results={}
    with Path("/tmp/slwm_research_gpu.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for run in cfg["runs"]:
            folder=root/run["id"]; values={}
            for kind,path in (("selected",folder/"best.pt"),("endpoint",folder/"last.pt"),("parent",Path(cfg["parent_root"])/run["parent"]/"best.pt")):
                ck=torch.load(path,map_location="cpu",weights_only=False); model=initialize(base,run["variant"],run["seed"]); model.load_state_dict(ck["model"],strict=True); model.cuda().eval()
                values[kind]={**evaluate(model,Fresh(),testcfg),"training_tokens":ck["total_token_equivalent_step"]*16384,"sha256":file_sha(path)}
                if kind=="selected": atomic_json(folder/"samples.json",generate(model,base,run["seed"],False)+generate(model,base,run["seed"],True))
                del model,ck; gc.collect(); torch.cuda.empty_cache()
            atomic_json(folder/"test.json",values); results[run["id"]]=values
    choices={r["id"]:json.loads((root/r["id"]/"complete.json").read_text()) for r in cfg["runs"]}
    atomic_json(root/"selection.json",choices); atomic_json(root/"summary.json",{"runs":choices,"test":results,"test_start_window":cfg["test_window_start"],"test_tokens_per_run":524288,"sources":sources})
    print(json.dumps({"s5_evaluation_complete":True}),flush=True)


if __name__=="__main__": main()
