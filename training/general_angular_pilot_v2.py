"""Selection-only controlled repair of radial/low-rank alignment collapse."""
import fcntl
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from training.general_alignment_diagnostic import geometry as _geometry
from training.general_batch import Sources
from training.general_losses import batch_loss,without
from training.general_sources import write
from training.general_train import initialize,atomic_save
from training.research_sources import file_sha


def _angular_regularizer(unit):
    z=unit.float();dimension=z.shape[-1];scaled=z*math.sqrt(dimension)
    std=scaled.std(0,unbiased=False);variance=2*F.relu(.5-std).mean()
    centered=scaled-scaled.mean(0);cov=centered.T@centered/max(1,len(z)-1)
    off=cov-torch.diag_embed(cov.diagonal());decorrelation=off.square().sum()/max(1,dimension*(dimension-1))
    mean=z.mean(0).square().sum()
    return variance+decorrelation+mean


def geometry(values):
    with torch.autocast(values.device.type,enabled=False):
        return _geometry(values.float())


def angular_regularizer(unit):
    with torch.autocast(unit.device.type,enabled=False):
        return _angular_regularizer(unit.float())


def validation(model,source,cfg):
    records={};model.eval()
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        for index,family in enumerate(("coco","clotho")):
            values=[]
            for batch_index in range(4):
                batch=source.make(family,11_000_000+index*100+batch_index,9001,"cuda",64)
                loss,metrics=batch_loss(model,None,batch,{},cfg,batch_index,True)
                a=model(without(batch["obs"],"text"));b=model(batch["companion"]);scores=a["embedding"]@b["embedding"].T
                positive=torch.tensor([[x==y for y in batch["ids"]] for x in batch["ids"]],device="cuda")
                hit=positive[torch.arange(len(scores),device="cuda"),scores.argmax(1)]
                values.append({"hit1":float(hit.float().mean()),"geometry":geometry(a["embedding"]),"text_geometry":geometry(b["embedding"]),"masked_error":metrics["picture_masked" if family=="coco" else "masked"],"alignment_loss":metrics["alignment"]})
            records[family]={"hit1":float(np.mean([v["hit1"] for v in values])),"masked_error":float(np.mean([v["masked_error"] for v in values])),"participation_ratio":float(np.mean([v["geometry"]["covariance_participation_ratio"] for v in values])),"text_participation_ratio":float(np.mean([v["text_geometry"]["covariance_participation_ratio"] for v in values])),"batches":values}
    return records


def main():
    torch.set_num_threads(4);cfg=json.loads(Path("configs/t2/general_transfer.json").read_text());root=Path(cfg["root"])/"angular_pilot_v2";parents=json.loads((root/"parents/parents.json").read_text());source=Sources(cfg,"train");selection=Sources(cfg,"selection")
    names=("training/general_angular_pilot_v2.py","training/general_alignment_diagnostic.py","docs/research/angular_alignment_pilot_v2_20261106.md","docs/research/angular_alignment_pilot_20261106.md","training/general_losses.py","training/general_batch.py","models/slwm_general.py")
    hashes={p:file_sha(p) for p in names};hashes["dataset_contract"]=file_sha(Path(cfg["data_root"])/"dataset_contract.json")
    existing=root/"source_hashes.json"
    if existing.exists():assert json.loads(existing.read_text())==hashes
    else:write(existing,hashes)
    for name in names:
        path=root/"source_snapshot"/name;path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():path.write_bytes(Path(name).read_bytes())
    results={}
    # One bounded lease gives the scientist an answer quickly; the main study
    # and text trainers retain their checkpoints and wait unchanged.
    with open("/tmp/slwm_research_gpu.lock","a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for parent_id in ("G1-101","G1-102"):
            parent=parents[parent_id];assert file_sha(parent["file"])==parent["sha256"]
            run=next(r for r in cfg["runs"] if r["id"]==parent_id)
            for coefficient in (0.,.2):
                name=parent_id+("-angular" if coefficient else "-control");out=root/name;out.mkdir(exist_ok=True)
                if (out/"complete.json").exists():results[name]=json.loads((out/"complete.json").read_text());continue
                ck=torch.load(parent["file"],map_location="cpu",weights_only=False,mmap=True);model=initialize(cfg,run).cuda();model.load_state_dict(ck["model"]);del ck
                optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=.1,fused=True);curves=[];started=time.perf_counter()
                curves.append({"updates":0,"selection":validation(model,selection,cfg)})
                for step in range(512):
                    family="coco" if step%2==0 else "clotho";batch=source.make(family,12_000_000+step,101,"cuda");model.train();optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        loss,metrics=batch_loss(model,None,batch,run,cfg,step)
                        a=model(without(batch["obs"],"text"));b=model(batch["companion"])
                        regularizer=(angular_regularizer(a["embedding"])+angular_regularizer(b["embedding"]))/2
                        objective=loss+coefficient*regularizer
                    assert torch.isfinite(objective);objective.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);optimizer.step()
                    if (step+1)%128==0:
                        row={"updates":step+1,"training_objective":float(objective.detach()),"angular_regularizer":float(regularizer.detach()),"selection":validation(model,selection,cfg)};curves.append(row);write(out/"curves.json",curves);print(json.dumps({"pilot":name,**row}),flush=True)
                    del batch,loss,objective,a,b,regularizer
                atomic_save(out/"endpoint.pt",{"model":model.state_dict(),"run":run,"coefficient":coefficient,"parent":parent,"updates":512,"sources":hashes})
                result={"name":name,"parent":parent,"coefficient":coefficient,"updates":512,"selection":curves[-1]["selection"],"active_seconds":time.perf_counter()-started,"endpoint_sha256":file_sha(out/"endpoint.pt"),"final_people_used":0,"parameters":sum(p.numel() for p in model.parameters())};write(out/"complete.json",result);results[name]=result
                del model,optimizer;gc.collect();torch.cuda.empty_cache()
    gates={}
    for parent in ("G1-101","G1-102"):
        a=results[parent+"-angular"]["selection"];b=results[parent+"-control"]["selection"]
        gates[parent]={family:bool(a[family]["hit1"]>b[family]["hit1"] and a[family]["participation_ratio"]>=8 and a[family]["masked_error"]<=b[family]["masked_error"]*1.1) for family in ("coco","clotho")}
    write(root/"summary.json",{"runs":results,"pilot_gates":gates,"passes_both_processors_both_families":all(v for row in gates.values() for v in row.values()),"scope":"Exploratory selection-only collapse repair; not final transfer evidence or a change to G1"})


if __name__=="__main__":main()
