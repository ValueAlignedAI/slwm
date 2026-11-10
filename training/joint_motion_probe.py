"""Matched-case diagnostic of historical command accessibility, outside G3 confirmation."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from training.general_data import CACHE
from training.general_readout_data import observed_robot_history
from training.joint_data import G2, mapped, read, thumbnails
from training.joint_g2_anchor import HASHES
from training.joint_sources import CONFIG,digest
from training.joint_train import atomic,gpu,load_parent,seed_all,tensors

ROBOT = Path(os.environ.get("SLWM_G2_ENDPOINTS","experiments/multimodal/t3/predictive-bottleneck-20261106"))/"readouts/features/droid"


def case_order(val):
    return hashlib.sha256(("G3-motion-split:"+val).encode()).digest()


def select(cases,limit):
    order = sorted(range(len(cases["ids"])),key=lambda i:case_order(cases["ids"][i]))
    per_episode,chosen = {},[]
    for i in order:
        episode = cases["ids"][i].rsplit("@",1)[0]
        if per_episode.get(episode,0)<4:
            per_episode[episode] = per_episode.get(episode,0)+1
            chosen.append(i)
            if len(chosen)==limit:
                break
    return sorted(chosen)


def cases(role,limit):
    folder = ROBOT/"G2-201"/role
    identifiers = read(folder/"cases.json")
    idx = select(identifiers,limit)
    original = read(Path("configs/t3/predictive_world.json"))["context"]
    stats = dict(np.load(G2/"normalizers.npz",allow_pickle=False))
    image,numeric,present = [],[],[]
    target = mapped(folder/"target_previous_command.npy")[idx]
    keys = [identifiers["ids"][i] for i in idx]
    for j,key in enumerate(keys):
        ep,at = key.rsplit("@",1)
        start = int(at)+original-24
        base = CACHE/"droid_g2"/ep
        with np.load(base/"trajectory.npz",allow_pickle=False) as traj:
            vals,mask,prev = observed_robot_history(traj,start,24,stats)
        np.testing.assert_allclose(prev[-1],target[j],rtol=1e-5,atol=1e-6)
        numeric.append(vals.clip(-10,10))
        present.append(mask)
        image.append(thumbnails(mapped(base/"rgb.npy")[start:start+24,0]))
    batch = {"image":np.stack(image),"numeric":np.stack(numeric),"present":np.stack(present)}
    return batch,tuple(idx),target,np.asarray(identifiers["groups"])[idx],keys


def features(model,batch,erase):
    rows = []
    for i in range(0,len(batch["image"]),96):
        b = tensors({k:v[i:i+96] for k,v in batch.items()},"cuda")
        if erase:
            b["present"][:,:,43:50] = False
        with torch.autocast("cuda",dtype=torch.bfloat16):
            z,_,_ = model.field(b)
        rows.append(z[:,-1].float().cpu().numpy())
    return np.concatenate(rows)


def evaluate(x,v,y,w,groups):
    x,v,y,w = [np.asarray(a,np.float64) for a in (x,v,y,w)]
    mu,sd = x.mean(0),x.std(0).clip(.001)
    x,v = (x-mu)/sd,(v-mu)/sd
    ym,ys = y.mean(0),y.std(0).clip(1e-6)
    y,w = (y-ym)/ys,(w-ym)/ys
    k = x.T@x/len(x)
    k.flat[::len(k)+1] += .01
    pred = v@np.linalg.solve(k,x.T@y/len(x))
    out = {}
    for g in np.unique(groups):
        take = groups==g
        out[str(g)] = (1-((w[take]-pred[take])**2).mean(0)/(w[take]**2).mean(0).clip(1e-9)).tolist()
    return {"mean_group_coordinate_r2":float(np.mean(list(out.values()))),"group_coordinate_r2":out}


def main():
    cfg = read(CONFIG)
    root = Path(cfg["output"])
    out = root/"motion_accessibility.json"
    if out.exists():
        return
    seed_all(81)
    fit,ix,y,_,fit_ids = cases("train",1024)
    held,jx,w,groups,held_ids = cases("selection",512)
    if {k.split("@")[0] for k in fit_ids} & {k.split("@")[0] for k in held_ids}:
        raise ValueError("Episode leak")
    result = {}
    for n in HASHES:
        for role,ids,j in (("train",fit_ids,ix),("selection",held_ids,jx)):
            recorded = read(ROBOT/n/role/"cases.json")["ids"]
            if ids!=[recorded[i] for i in j]:
                raise ValueError("Old state case order mismatch")
        x = mapped(ROBOT/n/"train/shared.npy")[list(ix)]
        v = mapped(ROBOT/n/"selection/shared.npy")[list(jx)]
        result[n] = evaluate(x,v,y,w,groups)
    for s in cfg["seeds"]:
        for n in ("gpt2-dense","signal-dense","signal-sparse"):
            model,parent = load_parent(cfg,n,s)
            model.cuda().eval()
            with torch.inference_mode():
                for step in (0,100):
                    if step:
                        ck = root/f"{n}-{s}"/"step-100.pt"
                        if digest(ck)!=read(ck.with_suffix(".json"))["delta_sha256"]:
                            raise ValueError("Unexpected joint delta")
                        model.apply_delta(torch.load(ck,map_location="cpu",weights_only=True)["delta"])
                    for erase in (False,True):
                        x = features(model,fit,erase)
                        v = features(model,held,erase)
                        result[f"{n}-{s}/{step}/"+("commands-erased" if erase else "observed")] = evaluate(x,v,y,w,groups)
            del model
    atomic(out,{"source_sha256":digest(Path(__file__)),"roles":["old train","old selection"],"columns":"three translation, three rotation, one gripper",
                "fit_windows":len(ix),"selection_windows":len(jx),"selection_groups":len(set(groups)),"case_digest":hashlib.sha256(json.dumps([fit_ids,held_ids]).encode()).hexdigest(),
                "metric":"mean over group-coordinate R2 versus train mean, full standardized ridge .01",
                "observation_scope":"G2 uses its cached 32-tick multiview and goal state. G3 sees the final 24 ticks, one camera and numeric history. Targets and endpoints match.",
                "scores":result})


if __name__=="__main__":
    with gpu():
        main()
