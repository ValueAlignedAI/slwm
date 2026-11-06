"""Final new-source/new-task tests, after the global G1 evaluation lock."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from training.general_frozen_features import project
from training.general_readout_train import (CONFIG, Readout, cache_split, comparisons,
    conditions, feature_folder, fixed_predictions, labels, load_array,
    load_configuration, score, source_guard)
from training.general_sources import write
from training.research_sources import file_sha


def prediction(root,family,task,condition,seed,role="test",control=None):
    out=root/"heads"/family/task/condition["id"]/str(seed)
    complete=json.loads((out/"complete.json").read_text())
    assert file_sha(out/"selected.pt")==complete["checkpoint_sha256"]
    assert file_sha(out/"projection.npz")==complete["projection_sha256"]
    fitted=dict(np.load(out/"projection.npz",allow_pickle=False))
    folder=feature_folder(root,family,condition,role)
    feature=condition["feature"]+("__"+control if control else "")
    values=load_array(folder,feature)
    extra=(np.asarray(load_array(folder,"target_common"),np.float32)-fitted["common_mean"])/fitted["common_scale"]
    x=np.concatenate([project(values,fitted),extra],1).astype(np.float32)
    ck=torch.load(out/"selected.pt",map_location="cpu",weights_only=False)
    model=Readout(ck["inputs"],ck["outputs"],ck["hidden"]);model.load_state_dict(ck["model"]);model.eval()
    with torch.inference_mode():result=model(torch.from_numpy(x)).numpy()
    return result,fitted


def confusion_by_group(prediction,target,present,groups):
    result=[]
    for group in sorted(set(groups)):
        keep=present&(np.asarray(groups)==group);truth=target[keep].astype(int);guess=prediction[keep].argmax(1)
        result.append(np.bincount(truth*12+guess,minlength=144).reshape(12,12))
    return np.stack(result)


def paired_effect(candidate,baseline,target,present,groups,task,draws=10000):
    """Positive means candidate improves; group is the independent sampling unit."""
    unique=sorted(set(groups));n=len(unique);rng=np.random.default_rng(7391)
    indices=rng.integers(0,n,(draws,n))
    if task=="activity":
        a=confusion_by_group(candidate,target,present,groups)
        b=confusion_by_group(baseline,target,present,groups)
        def error(matrix):
            return 1-(np.diagonal(matrix,axis1=-2,axis2=-1)/matrix.sum(-1).clip(1)).mean(-1)
        candidate_error=float(error(a.sum(0)));baseline_error=float(error(b.sum(0)))
        values=error(b[indices].sum(1))-error(a[indices].sum(1))
    else:
        power=1 if task=="actor" else 2
        per_a=(np.abs(candidate-target)**power*present).sum(1)/present.sum(1).clip(1)
        per_b=(np.abs(baseline-target)**power*present).sum(1)/present.sum(1).clip(1)
        a=np.array([per_a[np.asarray(groups)==g].mean() for g in unique]);b=np.array([per_b[np.asarray(groups)==g].mean() for g in unique])
        candidate_error=float(a.mean());baseline_error=float(b.mean());values=(b-a)[indices].mean(1)
    low,high=np.quantile(values,[.025,.975]);relative=(baseline_error-candidate_error)/max(1e-12,baseline_error)
    return {"groups":n,"candidate_error":candidate_error,"baseline_error":baseline_error,
            "gain":baseline_error-candidate_error,"gain_95ci":[float(low),float(high)],
            "relative_error_reduction":float(relative),"passes_10pct_transfer_gate":bool(relative>=.1 and low>0),
            "resampling":"10000 paired independent-group bootstrap draws, seed7391"}


def physical_actor(predicted,target):
    angle=(Rotation.from_rotvec(predicted[:,3:6]).inv()*Rotation.from_rotvec(target[:,3:6])).magnitude()
    return {"translation_command_error_m":float(np.linalg.norm(predicted[:,:3]-target[:,:3],axis=1).mean()),
            "rotation_command_error_rad":float(angle.mean()),"gripper_absolute_error":float(np.abs(predicted[:,6]-target[:,6]).mean())}


def evaluate():
    torch.set_num_threads(4);cfg,recipe,root=load_configuration();source_guard(root)
    lock=json.loads((Path(cfg["root"])/"evaluation_lock.json").read_text())
    assert lock["readouts_sha256"]==file_sha(root/"training_complete.json")
    training=json.loads((root/"training_complete.json").read_text());assert training["all_readouts_frozen"]
    for family,run in comparisons(cfg,recipe):cache_split(cfg,recipe,root,run,family,"test",True)
    results={};gate={}
    for family in ("mhealth","droid"):
        for task in (("activity","forecast") if family=="mhealth" else ("actor",)):
            key=family+"/"+task;predictions={};seed_scores={};fitted_by_name={}
            fixed,target,present,mean,scale=fixed_predictions(cfg,recipe,root,family,task,"test")
            folder=root/"features"/family/recipe["shared_raw_text_edge"] / "test"
            groups=json.loads((folder/"cases.json").read_text())["groups"]
            for condition in conditions(root,family,recipe):
                current=[];metrics=[]
                for seed in recipe["seeds"]:
                    value,fitted=prediction(root,family,task,condition,seed);current.append(value)
                    metrics.append({"seed":seed,**score(value,target,present,task)})
                name=condition["id"];predictions[name]=np.mean(current,axis=0);seed_scores[name]=metrics;fitted_by_name[name]=fitted
            predictions.update(fixed)
            baseline_name=training["choices"][key]["strongest_baseline"];baseline=predictions[baseline_name]
            scores={};effects={}
            for name,value in predictions.items():
                scores[name]=score(value,target,present,task)
                if task=="actor":scores[name].update(physical_actor(value*scale+mean,target*scale+mean))
                effects[name]=paired_effect(value,baseline,target,present,groups,task)
            destination=root/"evaluation"/family/task;destination.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(destination/"predictions.npz",**predictions,target=target,present=present)
            write(destination/"groups.json",groups)
            result={"strongest_selection_baseline":baseline_name,"scores":scores,"per_readout_seed":seed_scores,
                    "paired_effects":effects,"records":len(target),"independent_groups":len(set(groups)),
                    "readout_seed_aggregation":"Mean logits/normalized predictions of the two preregistered readout seeds",
                    "scope":"Frozen-core transfer and offline imitation; no physical deployment"}
            if family=="droid":
                interventions={}
                for condition in conditions(root,family,recipe):
                    if condition.get("secondary"):continue
                    current={}
                    for control in ("missing-goal","shuffled-goal","last-image","no-images"):
                        value=np.mean([prediction(root,family,task,condition,seed,control=control)[0] for seed in recipe["seeds"]],axis=0)
                        current[control]={**score(value,target,present,task),**physical_actor(value*scale+mean,target*scale+mean)}
                    interventions[condition["id"]]=current
                result["same_head_input_interventions"]=interventions
            result["secondary_conditions"]=[r["id"] for r in conditions(root,family,recipe) if r.get("secondary")]
            write(destination/"summary.json",result);results[key]=result
            gate[key]=all(effects[name]["passes_10pct_transfer_gate"] for name in ("G1-101","G1-103"))
    write(root/"evaluation_summary.json",{"tasks":results,"transfer_gates":gate,"broad_gate_requires_speech_check":True,"no_core_updates":True,"evaluation_lock_sha256":file_sha(Path(cfg["root"])/"evaluation_lock.json")})


if __name__=="__main__":evaluate()
