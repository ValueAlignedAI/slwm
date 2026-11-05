"""Matched broad shared-field learning with bounded GPU slices and frozen selection."""
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
import numpy as np
import torch
import torch.nn.functional as F
from models.slwm_stream import StreamWorld
from training.broad_batch import StreamCorpus,predict
from training.research_sources import file_sha
from training.t2_real_av import write_json

CONFIG="configs/t2/broad_stream.json"


def construct(cfg,run):
    return StreamWorld(variant=run["variant"],seed=run["seed"],**{**cfg["model"],"width":run.get("width",cfg["model"]["width"])})


def load_model(path,device="cuda"):
    ck=torch.load(path,map_location="cpu",weights_only=False); model=construct(ck["config"],ck["run"]); model.load_state_dict(ck["model"],strict=True); return model.to(device).eval(),ck


def masked_average(values,mask):
    return (values*mask).sum()/mask.sum().clamp_min(1)


def phone_loss(out,batch):
    error=F.cross_entropy(out["phone_logits"].transpose(1,2),batch["phones"],reduction="none"); weight=torch.where(batch["phones"]==0,.1,1.)*batch["valid"]*batch.get("phone_valid",torch.ones(batch["valid"].shape[0],device=batch["valid"].device,dtype=torch.bool))[:,None]
    return masked_average(error,weight)


def losses(out,batch,cfg,no_content=False):
    valid=batch["valid"]; hole=batch["holes"]; known=batch["a_mask"]; error=(out["mel"]-batch["audio"]).abs().mean(-1); nominal=torch.where(hole,1.,torch.where(known,.02,.2)); generation=~known.any(1); weight=torch.where(generation[:,None],torch.ones_like(nominal),nominal)*valid
    audio=masked_average(error,weight); video_mask=valid&batch["has_video"][:,None]; visual=masked_average((out["video"]-batch["video"]).square().mean(-1),video_mask); motion=masked_average(((out["video"][:,1:]-out["video"][:,:-1])-(batch["video"][:,1:]-batch["video"][:,:-1])).square().mean(-1),video_mask[:,1:]&video_mask[:,:-1]); phonetic=phone_loss(out,batch)
    eligible=torch.nonzero(batch.get("ctc_valid",torch.ones(valid.shape[0],device=valid.device,dtype=torch.bool)),as_tuple=False).flatten(); indices=eligible.cpu().tolist()
    if len(indices):
        target_lengths=torch.tensor([len(batch["texts"][i]) for i in indices],dtype=torch.long); input_lengths=valid[eligible].sum(1).long().cpu(); targets=torch.tensor([v for i in indices for v in batch["texts"][i]],device=valid.device,dtype=torch.long)
        ctc=F.ctc_loss(out["character_logits"][eligible].log_softmax(-1).transpose(0,1).contiguous(),targets,input_lengths,target_lengths,blank=0,zero_infinity=True)
    else: ctc=out["character_logits"].sum()*0
    future=[]; chunk=cfg["model"]["chunk"]
    for k in range(out["future"].shape[1]-1):
        start=(k+1)*chunk+2; end=start+8
        if end>valid.shape[1]: continue
        forecast=out["future"][:,k]; audio_error=(forecast[:,:,:400]-batch["audio"][:,start:end]).abs().mean(-1); value=masked_average(audio_error,valid[:,start:end]); vm=video_mask[:,start:end]; value=value+.2*masked_average((forecast[:,:,400:]-batch["video"][:,start:end]).square().mean(-1),vm); future.append(value)
    future_loss=torch.stack(future).mean() if future else audio*0
    activity=masked_average(F.cross_entropy(out["activity"].transpose(1,2),(batch["phones"]>0).long(),reduction="none"),valid)
    risk=masked_average((out["risk"]-(error.detach()+1e-4).log()).square(),valid&~known)
    lexical=0. if no_content else cfg["training"]["phoneme_weight"]*phonetic+cfg["training"]["ctc_weight"]*ctc
    total=audio+.3*visual+.1*motion+lexical+cfg["training"]["future_weight"]*future_loss+.03*activity+.02*risk
    return total,{"audio":audio,"video":visual,"motion":motion,"phone":phonetic,"ctc":ctc,"future":future_loss,"activity":activity,"risk":risk}


def subset(batch,index):
    result={}
    for key,value in batch.items(): result[key]=value[index] if isinstance(value,torch.Tensor) else [value[int(i)] for i in index.cpu()]
    return result


def alignment_loss(model,batch,reset=False):
    ids=torch.nonzero(batch["has_video"],as_tuple=False).flatten()[:8]
    if len(ids)<2: return batch["audio"].sum()*0
    small=subset(batch,ids); a=predict(model,small,reset,view="audio")["embedding"]; video=predict(model,small,reset,view="video")["embedding"]; wrong=predict(model,small,reset,view="wrong_video")["embedding"]; text=predict(model,small,reset,view="text")["embedding"]; labels=torch.arange(len(ids),device=a.device)
    return F.cross_entropy(a@torch.cat((video,wrong),0).T/.1,labels)+F.cross_entropy(a@text.T/.1,labels)


def validation(model,data,cfg,plans,reset=False):
    model.eval(); values={}
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        batch=data.batch(plans["libri_repair"]); out=predict(model,batch,reset); values["repair_mae"]=float(masked_average((out["mel"]-batch["audio"]).abs().mean(-1),batch["holes"]&batch["valid"])); values["phone_ce"]=float(phone_loss(out,batch))
        for mode in ("audio","video","text"):
            batch=data.batch(plans["ami_"+mode]); out=predict(model,batch,reset)
            value=masked_average((out["video"]-batch["video"]).square().mean(-1),batch["valid"]) if mode=="audio" else masked_average((out["mel"]-batch["audio"]).abs().mean(-1),batch["valid"])
            values[mode+"_prediction"]=float(value)
    values["selection_loss"]=values["repair_mae"]+.2*sum(values[m+"_prediction"] for m in ("audio","video","text"))+.05*values["phone_ce"]; return values


def freeze(cfg):
    root=Path(cfg["artifacts"]); root.mkdir(parents=True,exist_ok=True)
    names=(CONFIG,cfg["protocol"],"configs/t2/broad_stream_data.json","models/slwm_stream.py","training/broad_stream_train.py","training/broad_batch.py","training/stream_codec.py","training/broad_data.py","training/broad_sources.py","training/broad_annotation_correction.py","docs/research/broad_annotation_review_20261105.md","docs/research/ami_source_availability_review_20261105.md","docs/research/ami_decoded_clock_review_20261105.md","tests/test_broad_annotations.py","training/t1_torch_text.py","tests/test_stream_world.py","scripts/train/run_broad_stream_fedora.sh")
    hashes={}
    for name in names:
        target=root/"source_snapshot"/name; target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and file_sha(target)!=file_sha(name): raise ValueError("Registered broad source changed: "+name)
        if not target.exists(): shutil.copyfile(name,target)
        hashes[name]=file_sha(name)
    write_json(root/"source_hashes.json",hashes)
    if not(root/"environment.json").exists(): write_json(root/"environment.json",{"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,"cuda":torch.version.cuda,"packages":{n:importlib.metadata.version(n) for n in ("torchaudio","vocos","cmudict","numpy","scipy","soundfile","opencv-python-headless")}})
    return hashes


def train_slice(cfg,run,train,val,plans,sources):
    root=Path(cfg["artifacts"]); out=root/run["id"]; out.mkdir(exist_ok=True)
    if (out/"complete.json").exists(): return
    tc=cfg["training"]; torch.manual_seed(run["seed"]); model=construct(cfg,run).cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=tc["lr"],betas=(.9,.95),weight_decay=tc["weight_decay"],fused=True); rng=np.random.default_rng(run["seed"]+50000); begin=0; best=float("inf"); selected=0; seconds=0.; rises=0; curves=[]
    if (out/"resume.pt").exists():
        ck=torch.load(out/"resume.pt",map_location="cpu",weights_only=False); model.load_state_dict(ck["model"],strict=True); optimizer.load_state_dict(ck["optimizer"]); rng.bit_generator.state=ck["data_rng"]; torch.set_rng_state(ck["torch_rng"]); torch.cuda.set_rng_state_all(ck["cuda_rng"]); begin=ck["step"]; best=ck["best"]; selected=ck["selected"]; seconds=ck["seconds"]; rises=ck["rises"]; curves=json.loads((out/"curves.json").read_text()); del ck
    count=sum(p.numel() for p in model.parameters()); write_json(out/"config.json",{"matrix":cfg,"run":run,"parameters":count,"source_hashes":sources,"dataset_card_sha256":file_sha(Path(cfg["corpus"])/"dataset_card.json"),"warmstart":"None; all shared-field weights random; external decoder frozen/separate"})
    started=time.perf_counter(); torch.cuda.reset_peak_memory_stats(); totals=[]; end=min(begin+tc["eval_every"],tc["steps"])
    for step in range(begin,end):
        model.train(); optimizer.zero_grad(set_to_none=True); lr=tc["lr"]*min(1.,(step+1)/tc["warmup"])*(.1+.9*.5*(1+math.cos(math.pi*step/tc["steps"])))
        for group in optimizer.param_groups: group["lr"]=lr
        plan=train.sample(rng,tc["batch"]); batch=train.batch(plan,shuffled=run.get("shuffled_pairs",False)); results=[]
        for at in range(0,tc["batch"],tc["micro_batch"]):
            ids=torch.arange(at,min(at+tc["micro_batch"],tc["batch"]),device="cuda"); part=subset(batch,ids)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                out_model=predict(model,part,run.get("reset_memory",False)); loss,components=losses(out_model,part,cfg,run.get("no_content_loss",False)); align=alignment_loss(model,part,run.get("reset_memory",False)); loss=loss+tc["alignment_weight"]*align
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite broad loss")
            (loss*(len(ids)/tc["batch"])).backward(); results.append({k:float(v.detach()) for k,v in components.items()}); del out_model,part
        if step==begin:
            accounting={"total_neural_weights":count,"trainable_weights":sum(p.numel() for p in model.parameters() if p.requires_grad),"graph_connected_tensor_weights":sum(p.numel() for p in model.parameters() if p.grad is not None),"fitted_codecs_and_external_vocoder":"Counted separately in data/bundle manifests"}; write_json(out/"parameter_accounting.json",accounting)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),tc["clip"])
        if not torch.isfinite(norm): raise RuntimeError("Nonfinite broad gradient")
        optimizer.step(); totals.append({k:float(np.mean([r[k] for r in results])) for k in results[0]})
    value=validation(model,val,cfg,plans,run.get("reset_memory",False)); improved=value["selection_loss"]<best; rises=rises+1 if value["selection_loss"]>best*1.05 else 0
    if improved: best=value["selection_loss"]; selected=end
    seconds+=time.perf_counter()-started; record={"step":end,"training":{k:float(np.mean([r[k] for r in totals])) for k in totals[0]},"validation":value,"seconds":seconds,"peak_gpu_bytes":torch.cuda.max_memory_allocated(),"positive_windows":end*tc["batch"]}; curves.append(record); write_json(out/"curves.json",curves)
    saved={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"config":cfg,"run":run,"step":end,"best":best,"selected":selected,"seconds":seconds,"rises":rises,"data_rng":rng.bit_generator.state,"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),"source_hashes":sources,"parameters":count,"dataset_card_sha256":file_sha(Path(cfg["corpus"])/"dataset_card.json")}; torch.save(saved,out/"resume.pt")
    if improved: torch.save(saved,out/"best.pt")
    if end==tc["steps"] or rises>=2:
        torch.save({k:v for k,v in saved.items() if k!="optimizer"},out/"last.pt"); write_json(out/"complete.json",{"run":run,"parameters":count,"updates":end,"positive_windows":end*tc["batch"],"selected_step":selected,"selected_validation_loss":best,"overfit_stop":rises>=2,"wall_seconds":seconds,"checkpoint_sha256":file_sha(out/"best.pt"),"last_sha256":file_sha(out/"last.pt")})
    print(json.dumps({"run":run["id"],**record}),flush=True); del model,optimizer,saved,batch; gc.collect(); torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(); p.add_argument("--stage",choices=("smoke","train"),required=True); a=p.parse_args(); cfg=json.loads(Path(CONFIG).read_text()); root=Path(cfg["artifacts"]); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
    train=StreamCorpus(cfg,"train"); val=StreamCorpus(cfg,"validation"); rng=np.random.default_rng(93001); plans={"libri_repair":val.sample(rng,128,force_mode="repair",libri_only=True)}; common=val.sample(rng,32,ami_only=True)
    for mode in ("audio","video","text"): plans["ami_"+mode]=[{**p,"mode":mode} for p in common]
    if a.stage=="smoke":
        with Path("/tmp/slwm_research_gpu.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for run in cfg["runs"][:2]:
                model=construct(cfg,run).cuda(); batch=train.batch(train.sample(rng,128)); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
                with torch.autocast("cuda",dtype=torch.bfloat16): out=predict(model,batch); loss,values=losses(out,batch,cfg); loss=loss+.05*alignment_loss(model,batch)
                loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.); assert torch.isfinite(norm); torch.cuda.synchronize(); print(json.dumps({"smoke":run["variant"],"loss":float(loss),"parameters":sum(p.numel() for p in model.parameters()),"seconds":time.perf_counter()-start,"peak_gpu_bytes":torch.cuda.max_memory_allocated()}),flush=True); del model,batch,out,loss; gc.collect(); torch.cuda.empty_cache()
        return
    sources=freeze(cfg)
    if (root/"selection_plans.json").exists(): assert json.loads((root/"selection_plans.json").read_text())==plans
    else: write_json(root/"selection_plans.json",plans)
    corpus=Path(cfg["corpus"]); card=json.loads((corpus/"dataset_card.json").read_text())
    assert card["annotation_revision"]==2
    assert file_sha(corpus/"codec.npz")==card["codec_sha256"] and file_sha(corpus/"duration_model.json")==card["duration_model_sha256"]
    for relative,digest in json.loads((corpus/"corpus_hashes.json").read_text()).items(): assert file_sha(corpus/relative)==digest
    for manifest in corpus.glob("*/manifest.json"):
        for name,digest in json.loads(manifest.read_text())["files"].items(): assert file_sha(manifest.parent/name)==digest
    while any(not(root/r["id"]/"complete.json").exists() for r in cfg["runs"]):
        for run in cfg["runs"]:
            if (root/run["id"]/"complete.json").exists(): continue
            with Path("/tmp/slwm_research_gpu.lock").open("a") as lock:
                fcntl.flock(lock,fcntl.LOCK_EX); train_slice(cfg,run,train,val,plans,sources)
    choices={r["id"]:json.loads((root/r["id"]/"complete.json").read_text()) for r in cfg["runs"]}; write_json(root/"selection.json",choices); print(json.dumps({"broad_training_complete":True}),flush=True)


if __name__=="__main__": main()
