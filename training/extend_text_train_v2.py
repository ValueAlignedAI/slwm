"""Paired optimizer-preserving extension; bounded GPU slices and immutable parents."""
from __future__ import annotations
import argparse
import fcntl
import gc
import json
import importlib.metadata
import math
from pathlib import Path
import shutil
import time
import torch
import torch.nn.functional as F
from training.scale_text_train import UniqueWindows,initialize,evaluate,generate
from training.research_sources import file_sha
from training.text_fused_loss import fused_loss
from training.t2_real_av import write_json


def source_guard(cfg):
    root=Path(cfg["artifacts"]); root.mkdir(parents=True,exist_ok=True)
    names=("configs/t1/fineweb_extend_v2.json",cfg["protocol"],cfg["parent_config"],"training/extend_text_train_v2.py","training/text_fused_loss.py","training/scale_text_train.py","training/t1_torch_text.py")
    hashes={}
    for name in names:
        target=root/"source_snapshot"/name; target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and file_sha(target)!=file_sha(name): raise ValueError("Frozen extension source changed: "+name)
        if not target.exists(): shutil.copyfile(name,target)
        hashes[name]=file_sha(name)
    write_json(root/"source_hashes.json",hashes); return hashes


def learning_rate(step,cfg):
    # Initial 512 token-equivalent updates exactly preserve the registered S3
    # schedule. Thereafter extend its current LR smoothly over the larger budget.
    if step<cfg["warmup"]: return 6e-5+(cfg["lr"]-6e-5)*(step+1)/cfg["warmup"]
    initial_end=cfg["initial_small_updates"]
    if step<initial_end:
        phase=(step-cfg["warmup"])/(4096-cfg["warmup"])
        return cfg["lr_floor"]+(cfg["lr"]-cfg["lr_floor"])*.5*(1+math.cos(math.pi*phase))
    initial_phase=(initial_end-cfg["warmup"])/(4096-cfg["warmup"])
    start_lr=cfg["lr_floor"]+(cfg["lr"]-cfg["lr_floor"])*.5*(1+math.cos(math.pi*initial_phase))
    phase=(step-initial_end)/(cfg["additional_updates"]-initial_end)
    return cfg["lr_floor"]+(start_lr-cfg["lr_floor"])*.5*(1+math.cos(math.pi*phase))


def train_slice(cfg,run,base,sources):
    root=Path(cfg["artifacts"]); out=root/run["id"]; out.mkdir(exist_ok=True); tc=base["training"]
    if (out/"complete.json").exists(): return
    prior=Path(cfg["parent"])/run["parent"]/"best.pt"; parent_selection=json.loads((Path(cfg["parent"])/"selection.json").read_text())[run["parent"]]
    assert file_sha(prior)==parent_selection["checkpoint_sha256"] and parent_selection["selected_step"]==1024
    resume=out/"resume.pt"; old_out=Path(cfg["resume_root"])/run["resume"]; old_resume=old_out/"resume.pt"
    initial=resume if resume.exists() else old_resume if old_resume.exists() else prior
    state=torch.load(initial,map_location="cpu",weights_only=False)
    model=initialize(base,run["variant"],run["seed"]).cuda(); model.load_state_dict(state["model"],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["lr"],betas=(.9,.95),weight_decay=tc["weight_decay"],fused=True); optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"]); torch.cuda.set_rng_state_all(state["cuda_rng"])
    begin=state.get("extension_step",0); best=state.get("best_validation",parent_selection["selected_validation_nll"]); consecutive=state.get("consecutive_rises",0); cumulative=state.get("extension_seconds",0.); best_step=state.get("best_extension_step",0)
    stream=UniqueWindows(Path(base["corpus"])/"train.tokens.bin",512,run["seed"]); validation=UniqueWindows(Path(base["corpus"])/"validation.tokens.bin",512,901)
    curve_path=out/"curves.json" if (out/"curves.json").exists() else old_out/"curves.json"
    curves=json.loads(curve_path.read_text()) if curve_path.exists() else []
    provenance={"parent_checkpoint_sha256":parent_selection["checkpoint_sha256"],"parent_run":run["parent"],"source_hashes":sources,"parameters":sum(p.numel() for p in model.parameters()),"dataset_card_sha256":file_sha(Path(base["corpus"])/"dataset_card.json"),"warm_restart":True,"runtime_revision":2,"resumed_initial_source":str(initial),"runtime_loss":"plain CE through512, Liger fused CE thereafter","token_equivalent_update_size":16384}
    write_json(out/"config.json",{"extension":cfg,"matrix":base,"run":run,"provenance":provenance})
    if not (out/"best.pt").exists():
        chosen=torch.load(old_out/"best.pt",map_location="cpu",weights_only=False) if (old_out/"best.pt").exists() else state
        torch.save({**chosen,"config":base,"run":run,"provenance":provenance},out/"best.pt")
        if chosen is not state: del chosen
    del state; gc.collect(); torch.cuda.reset_peak_memory_stats(); started=time.perf_counter(); losses=[]
    end=min(begin+cfg["slice_updates"],cfg["additional_updates"])
    global_batch=32 if begin<cfg["initial_small_updates"] else cfg["global_batch"]
    micro_batch=8 if begin<cfg["initial_small_updates"] else cfg["micro_batch"]
    increment=global_batch//32
    for step in range(begin,end,increment):
        model.train(); optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups: group["lr"]=learning_rate(step,cfg)
        for micro in range(global_batch//micro_batch):
            index=(1024+step)*32+micro*micro_batch; x,y=stream.batch(index,micro_batch,"cuda")
            with torch.autocast("cuda",dtype=torch.bfloat16):
                loss=F.cross_entropy(model(x).float().reshape(-1,base["model"]["vocab_size"]),y.reshape(-1)) if global_batch==32 else fused_loss(model,x,y)
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite extended text loss")
            (loss/(global_batch//micro_batch)).backward(); losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(model.parameters(),tc["gradient_clip"]); optimizer.step()
    val=evaluate(model,validation,base); improved=val["nll"]<best
    consecutive=consecutive+1 if val["nll"]>best*1.05 else 0
    if improved: best=val["nll"]; best_step=end
    elapsed=cumulative+time.perf_counter()-started; tokens=(1024+end)*32*512
    curves.append({"extension_step":end,"total_token_equivalent_step":1024+end,"global_batch":global_batch,"micro_batch":micro_batch,"tokens":tokens,"validation":val,"training_nll":sum(losses)/len(losses),"seconds":elapsed,"peak_gpu_bytes":torch.cuda.max_memory_allocated()}); write_json(out/"curves.json",curves)
    saved={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"config":base,"run":run,"step":int(next(iter(optimizer.state.values()))["step"].item()),"total_token_equivalent_step":1024+end,"extension_step":end,"best_validation":best,"best_extension_step":best_step,"consecutive_rises":consecutive,"extension_seconds":elapsed,"provenance":provenance,"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()}
    torch.save(saved,resume)
    if improved: torch.save(saved,out/"best.pt")
    if end==cfg["additional_updates"] or consecutive>=2:
        torch.save({k:v for k,v in saved.items() if k!="optimizer"},out/"last.pt")
        write_json(out/"complete.json",{"run":run,"parameters":provenance["parameters"],"tokens_seen":tokens,"extension_token_equivalent_updates":end,"actual_optimizer_updates":saved["step"],"selected_total_token_equivalent_step":1024+best_step,"selected_validation_nll":best,"overfit_stop":consecutive>=2,"wall_seconds":elapsed,"checkpoint_sha256":file_sha(out/"best.pt"),"resume_sha256":file_sha(resume),"last_sha256":file_sha(out/"last.pt"),"parent_sha256":provenance["parent_checkpoint_sha256"]})
    print(json.dumps({"run":run["id"],"extension_step":end,"tokens":tokens,"validation_nll":val["nll"],"seconds":elapsed}),flush=True)
    del model,optimizer,saved; gc.collect(); torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(); p.add_argument("--stage",choices=("train","evaluate"),required=True); a=p.parse_args(); cfg=json.loads(Path("configs/t1/fineweb_extend_v2.json").read_text()); base=json.loads(Path(cfg["parent_config"]).read_text()); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
    assert importlib.metadata.version("liger-kernel")=="0.8.5"
    root=Path(cfg["artifacts"]); sources=source_guard(cfg); card=json.loads((Path(base["corpus"])/"dataset_card.json").read_text())
    for name,digest in card["files"].items(): assert file_sha(Path(base["corpus"])/name)==digest
    if a.stage=="train":
        while any(not(root/r["id"]/"complete.json").exists() for r in cfg["runs"]):
            for run in cfg["runs"]:
                if (root/run["id"]/"complete.json").exists(): continue
                with Path("/tmp/slwm_research_gpu.lock").open("a") as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX); train_slice(cfg,run,base,sources)
        return
    choices={r["id"]:json.loads((root/r["id"]/"complete.json").read_text()) for r in cfg["runs"]}; write_json(root/"selection.json",choices)
    class FreshTest:
        def __init__(self): self.source=UniqueWindows(Path(base["corpus"])/"test.tokens.bin",512,902)
        def batch(self,index,batch,device): return self.source.batch(cfg["test_window_start"]+index,batch,device)
    testcfg={**base,"training":{**base["training"],"eval_batches":cfg["test_batches"]}}; results={}
    with Path("/tmp/slwm_research_gpu.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for run in cfg["runs"]:
            out=root/run["id"]; ck=torch.load(out/"best.pt",map_location="cpu",weights_only=False); model=initialize(base,run["variant"],run["seed"]); model.load_state_dict(ck["model"],strict=True); model.cuda().eval()
            results[run["id"]]=evaluate(model,FreshTest(),testcfg); write_json(out/"test.json",results[run["id"]]); write_json(out/"samples.json",generate(model,base,run["seed"],False)+generate(model,base,run["seed"],True)); del model,ck; gc.collect(); torch.cuda.empty_cache()
    write_json(root/"summary.json",{"runs":choices,"test":results,"test_tokens_per_run":524288,"scope":"Optimizer-preserving batch-ramp extension;151M tokens/model target; still undertrained, separate text weights","training_minutes":sum(v["wall_seconds"] for v in choices.values())/60}); print(json.dumps({"extended_text_evaluation_complete":True}),flush=True)


if __name__=="__main__": main()
