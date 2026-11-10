"""Train-only disposable gradient/fit check while larger held sources download.

Uses existing100h speech/AMI training observations plus new train-only scenes,
wearables and verified robot groups. Provisional numeric scales are for shape/
memory validation only. No optimizer update or research checkpoint is retained.
"""
import copy
import fcntl
import json
from pathlib import Path
import numpy as np
import tiktoken
import torch
from models.slwm_general import GeneralWorld
from training.general_batch import Sources
from training.general_data import CACHE,ROOT
from training.general_losses import batch_loss
from training.general_sources import write


def main():
    cfg=json.loads(Path("configs/t2/general_transfer.json").read_text());source=Sources.__new__(Sources);source.cfg=cfg;source.role="train";source.encoder=tiktoken.get_encoding("gpt2");source.rows={};source.robot_family=cfg["robot_family"]
    for family in ("coco","clotho","pamap2"):source.rows[family]=[r for r in json.loads((CACHE/family/"rows.json").read_text()) if r["role"]=="train"]
    source.rows["speech"]=json.loads((CACHE/"speech/train_rows.json").read_text())
    folder=Path("artifacts/t2_audio_visual/broad_stream_prepared_v1/ami_train");legacy=json.loads((folder/"rows.json").read_text());source.rows["ami"]=[]
    for r in legacy:
        meeting=r["id"].rsplit("-",1)[0];source.rows["ami"].append({**r,"group":"OLD-"+meeting,"mel_file":str(folder/"mel.npy"),"pixel_file":str(folder/(r["id"]+"_pixels.npy")),"other_speaker_spans":[(a,b) for other in legacy if other["id"].rsplit("-",1)[0]==meeting and other["actor"]!=r["actor"] for a,b,_ in other["spans"]]})
    source.rows["droid"]=[r for r in json.loads((ROOT/cfg["robot_family"]/"raw_metadata/scene_role_plan.json").read_text())["rows"] if r["role"]=="train"]
    normal=np.load("artifacts/t2_audio_visual/broad_stream_prepared_v1/codec.npz");source.stats={"audio_mean":normal["audio_mean"],"audio_scale":normal["audio_scale"],"numeric_mean":np.zeros(64,np.float32),"numeric_scale":np.ones(64,np.float32),"action_mean":np.zeros(7,np.float32),"action_scale":np.ones(7,np.float32)}
    torch.set_num_threads(4);records=[]
    with open("/tmp/slwm_research_gpu.lock","a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for variant in ("transformer","slwm"):
            model=GeneralWorld(variant=variant,phone_count=79).cuda();teacher=copy.deepcopy(model).eval().requires_grad_(False)
            for index,family in enumerate(cfg["family_schedule_100"]):
                model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();batch=source.make(family,81234+index,101,"cuda")
                with torch.autocast("cuda",dtype=torch.bfloat16):loss,metrics=batch_loss(model,teacher,batch,{"variant":variant},cfg,index)
                loss.backward();assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                record={"variant":variant,"family":family,"parameters":sum(p.numel() for p in model.parameters()),"peak_bytes":torch.cuda.max_memory_allocated(),"metrics":metrics,"no_optimizer_update":True,"train_sources_only":True};records.append(record);print(json.dumps(record),flush=True);del batch,loss
            del model,teacher;torch.cuda.empty_cache()
    write(Path(cfg["root"])/"development_gradient_preflight.json",{"checks":records,"scope":"Disposable gradients and fit only; provisional numeric scales; no selected/final source examples and no optimized weights"})


if __name__=="__main__":main()
