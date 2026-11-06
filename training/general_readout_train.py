"""Frozen-core transfer: cache observations, fit small readouts, lock final data."""
import argparse
import copy
import fcntl
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import tiktoken
import torch
from torch import nn
import torch.nn.functional as F

from training.general_batch import tensor_tree
from training.general_frozen_features import frozen_features, supplied_text, fit_projection, project
from training.general_readout_data import ReadoutData
from training.general_sources import write
from training.general_train import initialize, atomic_save
from training.research_sources import file_sha

CONFIG = "configs/t2/general_readouts.json"


def load_configuration():
    cfg = json.loads(Path("configs/t2/general_transfer.json").read_text())
    recipe = json.loads(Path(CONFIG).read_text())
    root = Path(cfg["root"])/"readouts"; root.mkdir(exist_ok=True)
    assert (Path(cfg["root"])/"training_complete.json").exists(), "Freeze every core before readout extraction"
    return cfg, recipe, root


def source_guard(root):
    names = (CONFIG, "training/general_readout_train.py", "training/general_readout_data.py", "training/general_frozen_features.py", "docs/research/general_transfer_evaluation_20261106.md", "docs/research/general_aligned_state_probe_20261106.md", "tests/test_general_readout_data.py", "tests/test_general_frozen_features.py", "tests/test_general_readout_train.py")
    hashes = {p:file_sha(p) for p in names}
    path = root/"source_hashes.json"
    if path.exists():assert json.loads(path.read_text()) == hashes, "Frozen readout recipe changed"
    for name in names:
        target = root/"source_snapshot"/name; target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():shutil.copyfile(name,target)
    write(path,hashes)
    return hashes


def target_arrays(batch, family):
    names = ("activity", "future", "future_present", "last_physical", "last_present", "previous_physical") if family == "mhealth" else ("target", "previous_command", "plans", "commanded_poses", "future_poses", "anchors", "goal_present")
    return {name:batch[name] for name in (*names,"common")}


def cache_split(cfg, recipe, root, run, family, role, final=False):
    if role=="test":assert final and (Path(cfg["root"])/"evaluation_lock.json").exists(),"Final observations remain locked"
    folder = root/"features"/family/run["id"]/role
    marker = folder/"complete.json"
    checkpoint = Path(cfg["root"])/run["id"]/"endpoint.pt"
    weight_hash = file_sha(checkpoint)
    if marker.exists():
        saved = json.loads(marker.read_text());assert saved["core_checkpoint_sha256"] == weight_hash
        return saved
    if final:
        assert (Path(cfg["root"])/"evaluation_lock.json").exists(), "Freeze all calibration rules before final observations"
        assert (root/"training_complete.json").exists()
    data = ReadoutData(cfg,family,role,allow_final=final)
    assert len(data)>0
    folder.mkdir(parents=True,exist_ok=True)
    encoder = tiktoken.get_encoding("gpt2"); files={}; buffers={}; targets={}; groups=[]; identifiers=[]
    include_raw = run["id"] == recipe["shared_raw_text_edge"]
    started = time.perf_counter()
    with open("/tmp/slwm_research_gpu.lock","a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        saved = torch.load(checkpoint,map_location="cpu",weights_only=False,mmap=True)
        model = initialize(cfg,run).cuda().eval().requires_grad_(False);model.load_state_dict(saved["model"])
        del saved
        for start in range(0,len(data),64):
            batch = data.batch(range(start,min(len(data),start+64)))
            if family == "droid":batch["obs"].update(supplied_text(batch["captions"],encoder))
            obs = tensor_tree(batch["obs"],"cuda")
            with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
                values = frozen_features(model,obs,family,include_raw,data.stats)
            for name,array in values.items():
                if name == "edge" and run["id"] not in recipe["edge_controls"]:continue
                if name not in buffers:
                    buffers[name] = np.lib.format.open_memmap(folder/(name+".npy"),mode="w+",dtype=np.float16,shape=(len(data),array.shape[1]))
                assert np.isfinite(array).all(), "Nonfinite frozen features"
                buffers[name][start:start+len(array)] = array
            for name,array in target_arrays(batch,family).items():
                if name not in targets:
                    targets[name] = np.lib.format.open_memmap(folder/("target_"+name+".npy"),mode="w+",dtype=array.dtype,shape=(len(data),*array.shape[1:]))
                targets[name][start:start+len(array)] = array
            groups.extend(batch["groups"]);identifiers.extend(batch["ids"])
        del model;gc.collect();torch.cuda.empty_cache()
    for array in [*buffers.values(),*targets.values()]:array.flush()
    write(folder/"cases.json",{"ids":identifiers,"groups":groups})
    for path in folder.glob("*.npy"):files[path.name]=file_sha(path)
    result = {"core_checkpoint_sha256":weight_hash,"run":run,"family":family,"role":role,"records":len(data),"feature_shapes":{k:list(v.shape) for k,v in buffers.items()},"seconds":time.perf_counter()-started,"files":files,"cases_sha256":file_sha(folder/"cases.json"),"no_core_updates":True,"data_cases_sha256":file_sha(Path(cfg["data_root"])/"evaluation_cases.json")}
    write(marker,result);print(json.dumps({"frozen_features":run["id"],"family":family,"role":role,"records":len(data),"seconds":result["seconds"]}),flush=True)
    return result


def comparisons(cfg, recipe):
    by_id = {r["id"]:r for r in cfg["runs"]}
    for family,specialist in [("mhealth",recipe["sensor_specialist"]),("droid",recipe["robot_specialist"])]:
        for name in [*recipe["core_comparisons"],specialist]:yield family,by_id[name]


def conditions(root, family, recipe):
    names = [*recipe["core_comparisons"],recipe["sensor_specialist"] if family=="mhealth" else recipe["robot_specialist"]]
    specialist=recipe["sensor_specialist"] if family=="mhealth" else recipe["robot_specialist"]
    result = [{"id":name,"run":name,"feature":"shared","baseline":name==specialist} for name in names]
    result += [{"id":name+"-aligned","run":name,"feature":"aligned","baseline":False,"secondary":True} for name in names]
    result += [{"id":name+"-edge","run":name,"feature":"edge","baseline":True} for name in recipe["edge_controls"]]
    result.append({"id":"raw-history","run":recipe["shared_raw_text_edge"],"feature":"raw","baseline":True})
    if family=="mhealth":result.append({"id":"physical-invariants","run":recipe["shared_raw_text_edge"],"feature":"invariant","baseline":True})
    return result


def feature_folder(root, family, condition, role):
    return root/"features"/family/condition["run"]/role


def load_array(folder, name):
    return np.load(folder/(name+".npy"),mmap_mode="r",allow_pickle=False)


def labels(folder, family, task, recipe):
    if task=="activity":
        y=np.asarray(load_array(folder,"target_activity"))-1
        return y,(y>=0)
    if family=="droid":
        y=np.asarray(load_array(folder,"target_target"),np.float32)
        return y,np.isfinite(y)
    future=np.asarray(load_array(folder,"target_future"),np.float32)[:,np.array(recipe["horizons"])-1]
    last=np.asarray(load_array(folder,"target_last_physical"),np.float32)
    present=np.asarray(load_array(folder,"target_future_present"))[:,np.array(recipe["horizons"])-1]&np.asarray(load_array(folder,"target_last_present"))[:,None]
    return (future-last[:,None]).reshape(len(future),-1),present.reshape(len(future),-1)


def fixed_predictions(cfg,recipe,root,family,task,role):
    folder=root/"features"/family/recipe["shared_raw_text_edge"]/role
    training=root/"features"/family/recipe["shared_raw_text_edge"] / "train"
    train,present=labels(training,family,task,recipe)
    target,keep=labels(folder,family,task,recipe)
    if task=="activity":
        majority=np.bincount(train[present],minlength=12).argmax();prediction=np.zeros((len(target),12),np.float32);prediction[:,majority]=1
        return {"training-majority":prediction},target,keep,None,None
    count=present.sum(0).clip(1);mean=(train*present).sum(0)/count
    scale=np.sqrt((((train-mean)**2)*present).sum(0)/count).clip(recipe["target_scale_floor"])
    if task=="actor":
        stats=np.load(Path(cfg["data_root"])/"normalizers.npz")
        common=np.asarray(load_array(folder,"target_common"));gripper=common[:,9]*stats["numeric_scale"][42]+stats["numeric_mean"][42]
        noop=np.zeros_like(target);noop[:,6]=gripper
        predictions={"previous-command":np.asarray(load_array(folder,"target_previous_command")),"no-op":noop,"training-mean":np.broadcast_to(mean,target.shape)}
    else:
        last=np.asarray(load_array(folder,"target_last_physical"));previous=np.asarray(load_array(folder,"target_previous_physical"));velocity=(last-previous)[:,None]*np.array(recipe["horizons"],np.float32)[None,:,None]
        predictions={"persistence":np.zeros_like(target),"last-velocity":velocity.reshape(target.shape),"training-mean":np.broadcast_to(mean,target.shape)}
    return {k:((v-mean)/scale).astype(np.float32) for k,v in predictions.items()},((target-mean)/scale).astype(np.float32),keep,mean,scale


class Readout(nn.Module):
    def __init__(self, inputs, outputs, hidden=128):
        super().__init__();self.net=nn.Sequential(nn.Linear(inputs,hidden),nn.SiLU(),nn.Linear(hidden,outputs))
    def forward(self, values):return self.net(values)


def score(prediction, target, present, task):
    if task=="activity":
        truth=target[present].astype(int);guess=prediction[present].argmax(1)
        confusion=np.bincount(truth*12+guess,minlength=144).reshape(12,12)
        recall=np.diag(confusion)/confusion.sum(1).clip(1)
        f1=2*np.diag(confusion)/(confusion.sum(1)+confusion.sum(0)).clip(1)
        return {"selection":float(1-recall.mean()),"balanced_accuracy":float(recall.mean()),"macro_f1":float(f1.mean()),"confusion":confusion.tolist(),"records":int(present.sum())}
    error=prediction-target;denominator=max(1,int(present.sum()))
    mae=float((np.abs(error)*present).sum()/denominator);mse=float((error**2*present).sum()/denominator)
    return {"selection":mae if task=="actor" else mse,"normalized_mae":mae,"normalized_mse":mse,"records":len(target)}


def fit_one(cfg, recipe, root, family, condition, task, seed, sources):
    out=root/"heads"/family/task/condition["id"]/str(seed);out.mkdir(parents=True,exist_ok=True)
    if (out/"complete.json").exists():return json.loads((out/"complete.json").read_text())
    train_folder=feature_folder(root,family,condition,"train");val_folder=feature_folder(root,family,condition,"selection")
    train_features=load_array(train_folder,condition["feature"]);val_features=load_array(val_folder,condition["feature"])
    projection=fit_projection(train_features,recipe["projection"])
    common=np.asarray(load_array(train_folder,"target_common"),np.float32)
    common_mean=common.mean(0);common_scale=common.std(0).clip(.01)
    def inputs(folder, values):
        extra=(np.asarray(load_array(folder,"target_common"),np.float32)-common_mean)/common_scale
        return np.concatenate([project(values,projection),extra],1)
    x=torch.from_numpy(inputs(train_folder,train_features));vx=torch.from_numpy(inputs(val_folder,val_features))
    y,present=labels(train_folder,family,task,recipe);vy,vpresent=labels(val_folder,family,task,recipe)
    if task=="activity":
        mean=np.zeros(1,np.float32);scale=np.ones(1,np.float32)
        count=np.bincount(y[present],minlength=12);weight=torch.from_numpy((count.sum()/np.maximum(1,count)/12).astype(np.float32))
        outputs=12
    else:
        count=present.sum(0).clip(1);mean=(y*present).sum(0)/count
        scale=np.sqrt((((y-mean)**2)*present).sum(0)/count).clip(recipe["target_scale_floor"])
        y=((y-mean)/scale).astype(np.float32);vy=((vy-mean)/scale).astype(np.float32);outputs=y.shape[1]
    target=torch.from_numpy(y);keep=torch.from_numpy(present)
    group_ids=json.loads((train_folder/"cases.json").read_text())["groups"]
    indices={g:np.flatnonzero(np.array(group_ids)==g) for g in sorted(set(group_ids))}
    if task=="activity":indices={g:a[present[a]] for g,a in indices.items()};indices={g:a for g,a in indices.items() if len(a)}
    groups=list(indices);rng=np.random.default_rng(seed)
    with torch.random.fork_rng():
        torch.manual_seed(seed);model=Readout(x.shape[1],outputs,recipe["hidden"])
    optimizer=torch.optim.AdamW(model.parameters(),lr=recipe["learning_rate"],weight_decay=recipe["weight_decay"])
    best=float("inf");rises=0;curves=[];selected=0;started=time.perf_counter()
    np.savez(out/"projection.npz",**{k:v for k,v in projection.items() if k!="fitted_coefficients"},common_mean=common_mean,common_scale=common_scale,target_mean=mean,target_scale=scale)
    for step in range(recipe["updates"]):
        chosen=[indices[groups[int(rng.integers(len(groups)))]] for _ in range(recipe["batch"])]
        batch=np.array([a[int(rng.integers(len(a)))] for a in chosen]);optimizer.zero_grad(set_to_none=True);prediction=model(x[batch])
        if task=="activity":loss=F.cross_entropy(prediction,target[batch],weight=weight)
        else:
            point=F.smooth_l1_loss(prediction,target[batch],reduction="none") if task=="actor" else (prediction-target[batch]).square()
            loss=(point*keep[batch]).sum()/keep[batch].sum().clamp_min(1)
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if (step+1)%recipe["validation_every"]:continue
        with torch.no_grad():prediction=model(vx).numpy()
        metrics=score(prediction,vy,vpresent,task);curves.append({"update":step+1,"training_loss":float(loss.detach()),**metrics});write(out/"curves.json",curves)
        if metrics["selection"]<best:
            best=metrics["selection"];selected=step+1;rises=0
            atomic_save(out/"selected.pt",{"model":model.state_dict(),"inputs":x.shape[1],"outputs":outputs,"hidden":recipe["hidden"],"step":step+1,"task":task,"condition":condition,"seed":seed,"sources":sources})
        else:rises+=1
        if rises>=recipe["patience"]:break
    atomic_save(out/"resume.pt",{"model":model.state_dict(),"optimizer":optimizer.state_dict(),"numpy_rng":rng.bit_generator.state,"torch_rng":torch.get_rng_state(),"step":step+1})
    result={"condition":condition,"family":family,"task":task,"seed":seed,"updates":step+1,"selected_update":selected,"selection":best,"parameters":sum(p.numel() for p in model.parameters()),"projection_fitted_coefficients":projection["fitted_coefficients"],"seconds":time.perf_counter()-started,"checkpoint_sha256":file_sha(out/"selected.pt"),"projection_sha256":file_sha(out/"projection.npz"),"training_features_sha256":file_sha(train_folder/(condition["feature"]+".npy")),"core_weights_unchanged":True}
    write(out/"complete.json",result);print(json.dumps({"readout_complete":condition["id"],**{k:v for k,v in result.items() if k!="condition"}}),flush=True);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",required=True,choices=("features","fit"));args=parser.parse_args()
    torch.set_num_threads(4);cfg,recipe,root=load_configuration();sources=source_guard(root)
    if args.stage=="features":
        for family,run in comparisons(cfg,recipe):
            for role in ("train","selection"):cache_split(cfg,recipe,root,run,family,role)
        write(root/"features_complete.json",{"source_hashes":sources,"final_observations_opened":False})
        return
    assert (root/"features_complete.json").exists();results=[];choices={}
    for family in ("mhealth","droid"):
        for task in (("activity","forecast") if family=="mhealth" else ("actor",)):
            current=[]
            for condition in conditions(root,family,recipe):
                for seed in recipe["seeds"]:current.append(fit_one(cfg,recipe,root,family,condition,task,seed,sources))
            result=defaultdict_scores(current)
            baselines={name:value for name,value in result.items() if next(r for r in current if r["condition"]["id"]==name)["condition"]["baseline"]}
            fixed,target,keep,_,_=fixed_predictions(cfg,recipe,root,family,task,"selection")
            fixed_scores={name:score(value,target,keep,task) for name,value in fixed.items()}
            baselines.update({name:value["selection"] for name,value in fixed_scores.items()})
            choices[family+"/"+task]={"strongest_baseline":min(baselines,key=baselines.get),"selection_by_condition":result,"fixed_baselines":fixed_scores};results.extend(current)
    write(root/"training_complete.json",{"heads":results,"choices":choices,"sources":sources,"all_readouts_frozen":True,"final_observations_opened":False})


def defaultdict_scores(rows):
    names=sorted({r["condition"]["id"] for r in rows})
    return {name:float(np.mean([r["selection"] for r in rows if r["condition"]["id"]==name])) for name in names}


if __name__=="__main__":main()
