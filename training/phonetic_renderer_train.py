"""Train a counted common acoustic edge while G1 tests reusable state."""
from functools import lru_cache
import fcntl
import gc
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F
from models.phonetic_renderer import PhoneticRenderer,observed_condition
from training.temporal_masks import mask_support
from training.general_sources import write
from training.research_sources import file_sha

CONFIG="configs/t2/phonetic_renderer.json"


@lru_cache(maxsize=8)
def array(path):return np.load(path,mmap_mode="r")


def batch(rows,cfg,step,device,selection=False):
    rng=np.random.default_rng((9001 if selection else cfg["seed"])*1000003+step);b=64 if selection else cfg["batch"];t=cfg["packets"];groups={}
    for row in rows:groups.setdefault(row["group"],[]).append(row)
    names=sorted(groups);mels=[];phones=[];masks=[]
    for _ in range(b):
        current=groups[names[int(rng.integers(len(names)))]];row=current[int(rng.integers(len(current)))];start=int(rng.integers(row["length"]-t+1));a=row["offset"]+start
        mels.append(np.asarray(array(row["mel_file"])[a:a+t],np.float32).reshape(t*4,100));phones.append(np.asarray(array(row["phone_file"])[a:a+t],np.int64).reshape(-1))
        gap=.16 if selection else float(rng.uniform(.064,.320));center=t*1024/24000*.5;mask=mask_support(t*4,center-gap/2,center+gap/2)
        if not selection and rng.random()<cfg["style_dropout"]:mask[:]=False
        masks.append(mask)
    return tuple(torch.from_numpy(np.stack(v)).to(device) for v in (mels,phones,masks))


def objective(model,values,phones,present):
    target=(values-model.mean)/model.scale;style,prosody=observed_condition(target,present);prediction=model(phones,style,prosody);valid=phones>=0
    point=F.smooth_l1_loss(prediction,target,reduction="none").mean(-1);loss=(point*valid).sum()/valid.sum().clamp_min(1)
    delta=F.smooth_l1_loss(prediction[:,1:]-prediction[:,:-1],target[:,1:]-target[:,:-1],reduction="none").mean(-1);pair=valid[:,1:]&valid[:,:-1]
    return loss+.1*(delta*pair).sum()/pair.sum().clamp_min(1),((prediction-target).square().mean(-1)*valid).sum()/valid.sum().clamp_min(1)


def save(path,value):
    temporary=path.with_name(path.name+".tmp");torch.save(value,temporary);temporary.replace(path)


def main():
    cfg=json.loads(Path(CONFIG).read_text());root=Path(cfg["root"]);root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);train=json.loads(Path(cfg["training_rows"]).read_text());selection=json.loads(Path(cfg["selection_rows"]).read_text());names=json.loads(Path(cfg["phone_vocabulary"]).read_text());stats=np.load(cfg["normalizer"])
    files=(CONFIG,cfg["protocol"],"models/phonetic_renderer.py","training/phonetic_renderer_train.py","tests/test_phonetic_renderer.py","training/temporal_masks.py",cfg["training_rows"],cfg["selection_rows"],cfg["phone_vocabulary"],cfg["normalizer"],"artifacts/general_world/diverse_v1/prepared/speech/train_real_phones.npy","artifacts/general_world/diverse_v1/prepared/speech/legacy_selection_real_phones.npy")
    hashes={name:file_sha(name) for name in files}
    if (root/"source_hashes.json").exists():assert json.loads((root/"source_hashes.json").read_text())==hashes,"Frozen renderer source changed"
    write(root/"source_hashes.json",hashes);write(root/"config.json",cfg)
    for name in files[:5]:
        target=root/"source_snapshot"/name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copyfile(name,target)
    while not(root/"complete.json").exists():
        with open("/tmp/slwm_research_gpu.lock","a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);model=PhoneticRenderer(len(names),stats["audio_mean"],stats["audio_scale"]).cuda();opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"],fused=True)
            state={"updates":0,"best":float("inf"),"selected_update":0,"rises":0,"training_validation_seconds":0.}
            if (root/"resume.pt").exists():
                ck=torch.load(root/"resume.pt",map_location="cpu",weights_only=False);model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"]);state=ck["state"];del ck
            begin=state["updates"];end=min(begin+cfg["slice_updates"],cfg["updates"]);started=time.perf_counter();curves=json.loads((root/"curves.json").read_text()) if (root/"curves.json").exists() else []
            for step in range(begin,end):
                model.train();opt.zero_grad(set_to_none=True);lr=cfg["minimum_lr"]+(cfg["learning_rate"]-cfg["minimum_lr"])*.5*(1+math.cos(math.pi*step/max(1,cfg["updates"]-1)))
                for group in opt.param_groups:group["lr"]=lr
                values,phones,present=batch(train,cfg,step,"cuda")
                with torch.autocast("cuda",dtype=torch.bfloat16):loss,mse=objective(model,values,phones,present)
                assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step();state["updates"]=step+1
                if (step+1)%cfg["validation_every"]==0 or step+1==cfg["updates"]:
                    model.eval();scores=[]
                    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                        for i in range(8):
                            v,p,m=batch(selection,cfg,i,"cuda",True);_,value=objective(model,v,p,m);scores.append(float(value))
                    score=float(np.mean(scores));improved=score<state["best"];state["rises"]=state["rises"]+1 if score>state["best"]*1.1 else 0
                    if improved:state["best"]=score;state["selected_update"]=step+1;save(root/"best.pt",{"model":model.state_dict(),"config":cfg,"phone_count":len(names),"step":step+1,"source_hashes":hashes})
                    record={"update":step+1,"oracle_phone_selection_mse":score,"training_loss":float(loss.detach())};curves.append(record);write(root/"curves.json",curves);print(json.dumps(record),flush=True)
                    if state["rises"]>=2:break
            state["training_validation_seconds"]+=time.perf_counter()-started;save(root/"resume.pt",{"model":model.state_dict(),"optimizer":opt.state_dict(),"state":state,"config":cfg,"source_hashes":hashes})
            if state["updates"]==cfg["updates"] or state["rises"]>=2:
                save(root/"endpoint.pt",{"model":model.state_dict(),"config":cfg,"phone_count":len(names),"step":state["updates"],"source_hashes":hashes});write(root/"complete.json",{**state,"parameters":sum(p.numel() for p in model.parameters()),"phone_count":len(names),"training_context_hours":state["updates"]*cfg["batch"]*cfg["packets"]*1024/24000/3600,"training_speakers":len({r["actor"] for r in train}),"legacy_selection_speakers":len({r["actor"] for r in selection}),"new_final_speakers_used":0,"best_sha256":file_sha(root/"best.pt"),"endpoint_sha256":file_sha(root/"endpoint.pt")})
            del model,opt,values,phones,present,loss,mse;gc.collect();torch.cuda.empty_cache()


if __name__=="__main__":main()
