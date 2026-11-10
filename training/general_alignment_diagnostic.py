"""Selection-only collapse diagnostic; never updates a weight or opens final data."""
import fcntl
import gc
import json
from pathlib import Path

import numpy as np
import torch

from training.general_batch import Sources
from training.general_losses import without
from training.general_sources import write
from training.general_train import initialize
from training.research_sources import file_sha


def geometry(values):
    x=values.detach().float();center=x-x.mean(0);cov=center.T@center/max(1,len(x)-1)
    eigen=torch.linalg.eigvalsh(cov).clamp_min(0);total=eigen.sum()
    unit=torch.nn.functional.normalize(x,dim=-1)
    cosine=(unit.sum(0).square().sum()-len(x))/max(1,len(x)*(len(x)-1))
    return {"mean_norm":float(x.norm(dim=-1).mean()),"mean_feature_std":float(x.std(0,unbiased=False).mean()),
            "covariance_participation_ratio":float(total.square()/eigen.square().sum().clamp_min(1e-20)),
            "mean_off_diagonal_cosine":float(cosine),"largest_eigenvalue_fraction":float(eigen[-1]/total.clamp_min(1e-20))}


def main():
    torch.set_num_threads(4);cfg=json.loads(Path("configs/t2/general_transfer.json").read_text());root=Path(cfg["root"]);out=root/"selection_geometry";out.mkdir(exist_ok=True);source=Sources(cfg,"selection");records=[]
    with open("/tmp/slwm_research_gpu.lock","a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for run in cfg["runs"][:4]:
            checkpoint=root/run["id"]/"best.pt"
            if not checkpoint.exists():continue
            ck=torch.load(checkpoint,map_location="cpu",weights_only=False,mmap=True);position=ck["state"]["position"];model=initialize(cfg,run).cuda().eval().requires_grad_(False);model.load_state_dict(ck["model"]);del ck
            for index,family in enumerate(("coco","clotho","speech")):
                batch=source.make(family,10_300_000+index,9001,"cuda",64)
                with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
                    state=model(without(batch["obs"],"text"));text=model(batch["companion"])
                scores=state["embedding"]@text["embedding"].T
                positive=torch.tensor([[a==b for b in batch["ids"]] for a in batch["ids"]],device="cuda")
                hit=positive[torch.arange(len(scores),device="cuda"),scores.argmax(1)]
                item={"run":run["id"],"position":position,"family":family,"checkpoint_sha256":file_sha(checkpoint),"last_state":geometry(state["last"]),"raw_alignment":geometry(state["raw_embedding"]),"unit_alignment":geometry(state["embedding"]),"text_alignment":geometry(text["embedding"]),"paired_record_hit1":float(hit.float().mean()),"batch":len(scores),"selection_only":True,"no_optimizer_updates":True}
                records.append(item);print(json.dumps(item),flush=True)
            del model;gc.collect();torch.cuda.empty_cache()
    write(out/"diagnostic.json",{"records":records,"scope":"Early selection geometry, not a final capability test or evidence of task memorization","source_sha256":file_sha(__file__)})


if __name__=="__main__":main()
