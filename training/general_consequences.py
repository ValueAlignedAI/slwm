"""Held-environment action-conditioned prediction with strong servo controls."""
import fcntl
import gc
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from training.general_batch import tensor_tree
from training.general_data import CACHE
from training.general_evaluation_lock import register
from training.general_frozen_features import supplied_text
from training.general_readout_data import ReadoutData
from training.general_sources import write
from training.general_train import initialize
from training.research_sources import file_sha
import tiktoken


def rotation_matrix6d(values):
    columns=values.reshape(*values.shape[:-1],3,2)
    first=columns[...,0];second=columns[...,1]
    n1=np.linalg.norm(first,axis=-1,keepdims=True);u=first/n1.clip(1e-8)
    second=second-(u*second).sum(-1,keepdims=True)*u;n2=np.linalg.norm(second,axis=-1,keepdims=True);v=second/n2.clip(1e-8)
    matrix=np.stack([u,v,np.cross(u,v)],-1);invalid=(n1[...,0]<1e-6)|(n2[...,0]<1e-6)
    matrix[invalid]=np.eye(3)
    return matrix,invalid


def state_errors(prediction,truth):
    rotation,invalid=rotation_matrix6d(prediction[...,3:9])
    target=Rotation.from_euler("xyz",truth[...,3:6].reshape(-1,3)).as_matrix().reshape(*truth.shape[:-1],3,3)
    relative=np.swapaxes(rotation,-1,-2)@target
    angle=np.arccos(((np.trace(relative,axis1=-2,axis2=-1)-1)/2).clip(-1,1));angle[invalid]=np.pi
    return {"translation_m":np.linalg.norm(prediction[...,:3]-truth[...,:3],axis=-1),"rotation_rad":angle,"gripper_absolute":np.abs(prediction[...,9]-truth[...,6]),"invalid_rotation":invalid.astype(float)}


def physical_state(poses):
    matrix=Rotation.from_euler("xyz",poses[...,3:6].reshape(-1,3)).as_matrix().reshape(*poses.shape[:-1],3,3)
    return np.concatenate([poses[...,:3],matrix[...,:2].reshape(*poses.shape[:-1],6),poses[...,6:7]],-1)


def pose_controls(batch,horizons):
    b=len(batch["anchors"]);last=np.concatenate([batch["anchors"],batch["anchor_grippers"][:,None]],1)
    persistence=np.repeat(last[:,None],len(horizons),1)
    commanded=np.concatenate([batch["commanded_poses"][:,np.array(horizons)-1],batch["plans"][:,np.array(horizons)-1,6:]],-1)
    velocity=persistence.copy();delta=last[:,:3]-batch["previous_poses"][:,:3]
    velocity[...,:3]+=np.array(horizons)[None,:,None]*delta[:,None]
    initial=Rotation.from_euler("xyz",last[:,3:6]);previous=Rotation.from_euler("xyz",batch["previous_poses"][:,3:6]);step=(previous.inv()*initial).as_rotvec()
    for i,h in enumerate(horizons):velocity[:,i,3:6]=(initial*Rotation.from_rotvec(step*h)).as_euler("xyz")
    velocity[...,6]+=(last[:,6]-batch["previous_poses"][:,6])[:,None]*np.array(horizons)[None]
    return {"persistence":physical_state(persistence),"last-velocity":physical_state(velocity),"commanded-pose-as-state":physical_state(commanded)}


def images16(data,batch,horizons):
    future=[];last=[];previous=[]
    for identifier in batch["ids"]:
        key,start=identifier.rsplit("@",1);current=int(start)+data.cfg["context"]-1
        image=data.load(str(CACHE/data.cfg["robot_family"]/key/"rgb.npy"))
        indices=np.r_[current-1,current,current+np.asarray(horizons)]
        value=np.asarray(image[indices],np.float32).transpose(0,1,4,2,3)/127.5-1
        value=value.reshape(len(indices),3,3,16,4,16,4).mean((4,6))
        previous.append(value[0]);last.append(value[1]);future.append(value[2:])
    return np.stack(future),np.stack(last),np.stack(previous)


def evaluate_run(cfg,run,root):
    out=root/"consequences"/run["id"];out.mkdir(parents=True,exist_ok=True)
    if (out/"complete.json").exists():return
    horizons=[1,4,8];data=ReadoutData(cfg,"droid","test",True);stats=data.stats;encoder=tiktoken.get_encoding("gpt2");records=[]
    with open("/tmp/slwm_research_gpu.lock","a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);ck=torch.load(root/run["id"]/"endpoint.pt",map_location="cpu",weights_only=False,mmap=True);model=initialize(cfg,run).cuda().eval().requires_grad_(False);model.load_state_dict(ck["model"]);del ck
        for start in range(0,len(data),32):
            batch=data.batch(range(start,min(start+32,len(data))));batch["obs"].update(supplied_text(batch["captions"],encoder));obs=tensor_tree(batch["obs"],"cuda");n=len(batch["ids"])
            plans=(batch["plans"]-stats["action_mean"])/stats["action_scale"]
            future_times=np.broadcast_to(np.arange(cfg["horizon"],dtype=np.float32)+cfg["context"],(n,cfg["horizon"])).copy()
            truth=batch["future_poses"][:,np.array(horizons)-1];frames,last,previous=images16(data,batch,horizons)
            image_controls={"last-frame":np.repeat(last[:,None],len(horizons),1),"last-image-velocity":last[:,None]+(last-previous)[:,None]*np.array(horizons)[None,:,None,None,None,None]}
            controls=pose_controls(batch,horizons);predicted_images={};predictions={}
            with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
                state=model(obs);times=torch.from_numpy(future_times).cuda()
                for condition,plan in [("actual-plan",plans),("no-plan",None),("shuffled-plan",np.roll(plans,1,axis=0))]:
                    supplied=None if plan is None else torch.from_numpy(plan.astype(np.float32)).cuda()
                    query=model.query(state,times,obs["time_unit"],2,supplied)
                    normalized=obs["numeric"][:,-1:,33:43]+model.propose(query,"numeric")[...,33:43]
                    physical=normalized.cpu().numpy()*stats["numeric_scale"][33:43]+stats["numeric_mean"][33:43]
                    predictions[condition]=physical[:,np.array(horizons)-1]
                    views=[]
                    for view in range(3):
                        query=model.query(state,times,obs["time_unit"],1,supplied,view=view)
                        delta=2*model.propose(query,"image").tanh().cpu().numpy()[:,np.array(horizons)-1]
                        views.append(last[:,None,view]+delta)
                    predicted_images[condition]=np.stack(views,2)
            predictions.update(controls);predicted_images.update(image_controls)
            for condition,value in predictions.items():
                errors=state_errors(value,truth)
                for i,identifier in enumerate(batch["ids"]):records.append({"id":identifier,"group":batch["groups"][i],"condition":condition,"kind":"pose",**{k:v[i].tolist() for k,v in errors.items()}})
            duration=np.diff([0,*horizons]).astype(np.float32)
            target_motion=np.diff(np.concatenate([last[:,None],frames],1),axis=1)/duration[None,:,None,None,None,None]
            for condition,value in predicted_images.items():
                pixel=((value-frames)**2).mean((2,3,4,5))
                motion=np.diff(np.concatenate([last[:,None],value],1),axis=1)/duration[None,:,None,None,None,None]
                motion_error=((motion-target_motion)**2).mean((2,3,4,5))
                for i,identifier in enumerate(batch["ids"]):records.append({"id":identifier,"group":batch["groups"][i],"condition":condition,"kind":"image","pixel_mse":pixel[i].tolist(),"motion_velocity_mse":motion_error[i].tolist()})
            write(out/"records.json",records)
        del model;gc.collect();torch.cuda.empty_cache()
    summaries={}
    for kind in ("pose","image"):
        for condition in sorted({r["condition"] for r in records if r["kind"]==kind}):
            current=[r for r in records if r["kind"]==kind and r["condition"]==condition];keys=[k for k in current[0] if k not in ("id","group","condition","kind")]
            summaries[kind+"/"+condition]={k:np.mean([r[k] for r in current],axis=0).tolist() for k in keys}
    write(out/"summary.json",{"horizons_ordinal_steps":horizons,"metrics":summaries,"groups":len({r["group"] for r in records}),"scope":"Prediction of actual recorded outcomes; wrong-plan interventions do not supply physical counterfactual ground truth","rotation_invalid_penalty_rad":float(np.pi),"no_source_identity_or_future_observation_inputs":True})
    write(out/"complete.json",{"run":run["id"],"records":len(data),"source_sha256":file_sha(__file__),"new_fit_updates":0})
    print(json.dumps({"consequence_evaluated":run["id"]}),flush=True)


def main():
    torch.set_num_threads(4);cfg=json.loads(Path("configs/t2/general_transfer.json").read_text());root=Path(cfg["root"]);register(cfg);assert (root/"evaluation_lock.json").exists()
    for run in cfg["runs"]:
        if run["id"] in {"G1-101","G1-102","G1-103","G1-104","G1-105","G1-106","G1-107","G1-108","G1-109","G1-113"}:evaluate_run(cfg,run,root)


if __name__=="__main__":main()
