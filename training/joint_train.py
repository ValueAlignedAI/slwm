"""Resumable equal-exposure G3 training, with immutable milestones and delta exports."""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from models.joint_transfer import JointWorld, objective
from training.joint_data import CORPUS, G2, JointData, read
from training.joint_sources import CONFIG, digest
from training.scale_text_train import initialize

SOURCE = [str(CONFIG),"docs/research/joint_transfer_20261109.md","models/joint_transfer.py",
          "training/joint_data.py","training/joint_sources.py","training/joint_train.py",
          "training/general_batch.py","training/general_data.py","training/general_readout_data.py",
          "training/scale_text_train.py","training/t1_torch_text.py"]
PARENTS = Path(os.getenv("SLWM_S5_ROOT","experiments/text/t1/500m-20261106"))


def atomic(path, value, binary=False):
    path.parent.mkdir(exist_ok=True,parents=True)
    tmp = path.with_name(path.name+".tmp")
    if binary:
        torch.save(value,tmp)
    else:
        tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
    tmp.replace(path)


@contextmanager
def gpu():
    with open("/tmp/slwm_research_gpu.lock","a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def seed_all(seed):
    torch.manual_seed(seed)
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = True


def load_parent(cfg, name, seed):
    profile = cfg["profiles"][name]
    parent, sha = cfg["parents"][f"{profile['core']}-{seed}"]
    path = PARENTS/parent/"best.pt"
    if digest(path)!=sha:
        raise ValueError(f"Unexpected parent for {name}/{seed}")
    saved = torch.load(path,map_location="cpu",mmap=True,weights_only=False)
    if saved["total_token_equivalent_step"]*16384 != 500006912:
        raise ValueError("Expected matched 500M-token parent")
    core = initialize(saved["config"],profile["core"],seed)
    core.load_state_dict(saved["model"])
    return JointWorld(core,profile,seed), sha


def tensors(batch,device):
    return {k:torch.as_tensor(v,device=device) for k,v in batch.items()}


def bind_inputs(cfg):
    paths = [*map(Path,SOURCE), CORPUS/"dataset_card.json",G2/"dataset_contract.json",G2/"normalizers.npz",
             Path(cfg["data"])/"sources/manifest.json"]
    record = {str(p):digest(p) for p in paths}
    path = Path(cfg["output"])/"training_lock.json"
    if path.exists() and read(path)!=record:
        raise ValueError("Joint sources or data contract changed")
    atomic(path,record)
    return digest(path)


def exposure_digest(last, family, keys, batch):
    h = hashlib.sha256(bytes.fromhex(last))
    h.update(family.encode())
    h.update(json.dumps(keys,separators=(",",":")).encode())
    for key in sorted(batch):
        value = np.ascontiguousarray(batch[key])
        h.update(key.encode()+str(value.shape).encode()+value.dtype.str.encode())
        h.update(value.data)
    return h.hexdigest()


def train(cfg, name, seed, target, provenance):
    seed_all(seed)
    output = Path(cfg["output"])/f"{name}-{seed}"
    endpoint = output/f"step-{target}.pt"
    card = output/f"step-{target}.json"
    if card.exists():
        if digest(endpoint)!=read(card)["delta_sha256"]:
            raise ValueError(f"Damaged {endpoint}")
        return
    model, parent_sha = load_parent(cfg,name,seed)
    model.cuda().train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg["lr"],
                            betas=(.9,.95),weight_decay=cfg["weight_decay"],fused=True)
    data = JointData(cfg,seed)
    counter, routes, losses = Counter(), {}, {}
    previous, elapsed, chain = 0, 0., "00"*32
    resume = output/"resume.pt"
    if resume.exists():
        saved = torch.load(resume,map_location="cpu",weights_only=False)
        if (saved["name"],saved["seed"],saved["parent"],saved["lock"])!=(name,seed,parent_sha,provenance):
            raise ValueError("Resume identity mismatch")
        model.apply_delta(saved["delta"])
        opt.load_state_dict(saved["opt"])
        previous,elapsed,chain = saved["step"],saved["seconds"],saved["chain"]
        counter.update(saved["calls"])
        losses, routes = saved["losses"],saved["routes"]
        del saved
    if previous>target:
        raise ValueError("Skipped an immutable milestone")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for step in range(previous,target):
        family, batch, ids = data.get(step)
        chain = exposure_digest(chain,family,ids,batch)
        learning_rate = cfg["lr"]*min(1.,(step+1)/20)*(.1+.9*(1+math.cos(math.pi*step/cfg["milestones"][-1]))/2)
        opt.param_groups[0]["lr"] = learning_rate
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            loss,metrics,counts = objective(model,tensors(batch,"cuda"))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        counter[family] += 1
        if counts:
            arr = torch.stack(counts).cpu().numpy()
            routes[family] = (np.asarray(routes.get(family,np.zeros_like(arr)))+arr).tolist()
        cumulative = losses.setdefault(family,{})
        for k, v in {"objective":loss, **metrics}.items():
            old, n = cumulative.get(k,[0.,0])
            cumulative[k] = [old+float(v.detach()),n+1]
        if (step+1)%25==0 or step==previous:
            atomic(output/"progress.json",{"step":step+1,"target":target,"seconds":elapsed+time.monotonic()-started,"family":family})
        if (step+1)%250==0 or step+1==target:
            atomic(resume,{"name":name,"seed":seed,"parent":parent_sha,"lock":provenance,"step":step+1,"delta":model.delta(),
                           "opt":opt.state_dict(),"seconds":elapsed+time.monotonic()-started,"chain":chain,
                           "calls":dict(counter),"routes":routes,"losses":losses},True)
    sizes = {k:len(rows) for k,rows in data.rows.items()}
    sizes.update(cifar=len(data.cifar),text=data.text.blocks)
    record = {"profile":name,"seed":seed,"step":target,"parent_sha256":parent_sha,"lock_sha256":provenance,
              "accounting":model.accounting(),"calls":dict(counter),"eligible_rows":sizes,
              "unique_rows":{k:min(counter[k]*cfg["batch"],n) for k,n in sizes.items()},
              "general_lm_targets":counter["text"]*cfg["batch"]*cfg["text_length"],
              "example_presentations":sum(counter.values())*cfg["batch"], "exposure_sha256":chain,
              "loss_means":{k:{m:s/n for m,(s,n) in v.items()} for k,v in losses.items()},
              "expert_counts_by_family_and_layer":routes,"active_train_seconds":elapsed+time.monotonic()-started,
              "peak_gpu_bytes":torch.cuda.max_memory_allocated(),"device":torch.cuda.get_device_name()}
    atomic(endpoint,{"profile":name,"seed":seed,"step":target,"parent_sha256":parent_sha,
                     "training_lock":provenance,"delta":model.delta()},True)
    record["delta_sha256"] = digest(endpoint)
    atomic(card,record)
    print(json.dumps({k:record[k] for k in ("profile","seed","step","accounting","active_train_seconds")}),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile",required=True)
    p.add_argument("--seed",required=True,type=int)
    p.add_argument("--target",required=True,type=int)
    a = p.parse_args()
    cfg = read(CONFIG)
    if a.profile not in cfg["profiles"] or a.seed not in cfg["seeds"] or a.target not in cfg["milestones"]:
        p.error("Unknown registered run")
    with gpu():
        train(cfg,a.profile,a.seed,a.target,bind_inputs(cfg))


if __name__=="__main__":
    main()
