"""Trained-world train-only inference equivalence/latency; originals untouched."""
import argparse
import copy
from contextlib import nullcontext
import fcntl
import json
from pathlib import Path
import time
import numpy as np
import torch
from models.slwm_stream_folded import fold_signal_branches
from training.broad_stream_train import load_model
from training.broad_batch import StreamCorpus,predict
from training.research_sources import file_sha
from training.t2_real_av import write_json


def compare(a,b):
    return {k:{"max_absolute_error":float((a[k].float()-b[k].float()).abs().max()),"relative_l2_error":float((a[k].float()-b[k].float()).norm()/a[k].float().norm().clamp_min(1e-8))} for k in a}


def run(device):
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; cfg=json.loads(Path("configs/t2/broad_stream.json").read_text()); data=StreamCorpus(cfg,"train"); checkpoint=Path(cfg["artifacts"])/"EXP-T2-801/best.pt"; root=Path("experiments/multimodal/t2/folded-inference-20261105"); root.mkdir(exist_ok=True); original,ck=load_model(checkpoint,device); original.requires_grad_(False); folded=fold_signal_branches(copy.deepcopy(original)); plans=data.sample(np.random.default_rng(294),16 if device=="cpu" else 128); batch=data.batch(plans,device=device); outputs={"checkpoint_sha256":file_sha(checkpoint),"checkpoint_step":ck["step"],"training_only":True,"weights_updated":False,"model":folded.folding_metadata,"dtype_results":{}}
    for dtype in ("fp32",) if device=="cpu" else ("fp32","bf16"):
        amp=lambda:torch.autocast("cuda",dtype=torch.bfloat16) if dtype=="bf16" else nullcontext()
        with torch.inference_mode(),amp():
            a=predict(original,batch); b=predict(folded,batch); errors=compare(a,b); latency=[]
            for model in (original,folded):
                for _ in range(3): predict(model,batch)
                if device=="cuda": torch.cuda.synchronize()
                start=time.perf_counter()
                for _ in range(12): predict(model,batch)
                if device=="cuda": torch.cuda.synchronize()
                latency.append((time.perf_counter()-start)/12)
        outputs["dtype_results"][dtype]={"original_seconds_per_batch":latency[0],"folded_seconds_per_batch":latency[1],"speedup":latency[0]/latency[1],"batch":len(plans),"packets":128,"errors":errors,"all_heads_fp32_tolerance_pass":all(v["max_absolute_error"]<2e-4 and v["relative_l2_error"]<1e-4 for v in errors.values()) if dtype=="fp32" else None}; print(json.dumps({"folded_inference":device,"dtype":dtype,"speedup":latency[0]/latency[1],"worst_error":max(v["max_absolute_error"] for v in errors.values())}),flush=True)
    write_json(root/(device+".json"),outputs)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--device",choices=("cpu","cuda"),required=True); a=p.parse_args()
    if a.device=="cpu": run("cpu")
    else:
        with Path("/tmp/slwm_research_gpu.lock").open("a") as lock: fcntl.flock(lock,fcntl.LOCK_EX); run("cuda")


if __name__=="__main__": main()
