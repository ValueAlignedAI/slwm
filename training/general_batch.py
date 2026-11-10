"""Deterministic real-source batches for G1; targets never enter observation dicts."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import tiktoken
from training.general_data import CACHE,ROOT,pose_delta,order
from training.temporal_masks import mask_support

STEP=1024/24000
ROBOT_STATE=slice(33,43)
ROBOT_ACTION=slice(43,50)


def schedule(cfg,position):
    """Deterministic shuffled100-call blocks, identical across all paired runs."""
    names=[name for name,count in cfg["family_schedule_100"].items() for _ in range(count)]
    rng=np.random.default_rng(7391+position//100); rng.shuffle(names)
    return names[position%100]


def interpolate(values,present):
    """Observed-only fine mel[F,100] -> baseline[F,100]; never reads masked values."""
    known=np.flatnonzero(present); result=np.zeros_like(values,np.float32)
    if not len(known): return result
    clock=np.arange(len(values))
    for band in range(values.shape[1]): result[:,band]=np.interp(clock,known,values[known,band])
    return result


def rotation6d(observed,gripper):
    """World pose[N,6] + gripper[N,1] -> continuous numeric state[N,10]."""
    rotation=Rotation.from_euler("xyz",observed[:,3:]).as_matrix()[:,:,:2].reshape(-1,6)
    return np.concatenate([observed[:,:3],rotation,gripper],1).astype(np.float32)


def normalize(values,mean,scale): return (values-mean)/scale


def numeric_baseline(values,present,time):
    """Irregular-clock observed-only interpolation[B,T,C], absent channels ->0."""
    result=np.zeros_like(values)
    for b in range(len(values)):
        for c in range(values.shape[-1]):
            known=np.flatnonzero(present[b,:,c])
            if len(known):result[b,:,c]=np.interp(time[b],time[b,known],values[b,known,c])
    return result


def tensor_tree(value,device):
    if isinstance(value,np.ndarray): return torch.from_numpy(np.ascontiguousarray(value)).to(device,non_blocking=True)
    if isinstance(value,dict): return {k:tensor_tree(v,device) for k,v in value.items()}
    return value


def base_observation(batch,steps,delta,ordinal=False):
    """Empty typed observation with a genuine shared clock[B,T]."""
    clock=np.broadcast_to(np.arange(steps,dtype=np.float32)*delta,(batch,steps)).copy()
    return {"time":clock,"time_unit":np.full(batch,int(ordinal),np.int64),"valid":np.ones((batch,steps),bool)}


class Sources:
    """Memory-mapped caches; all source/identity metadata stay outside the model."""
    def __init__(self,cfg,role="train"):
        self.cfg=cfg; self.role=role; self.rows={}; self.cache={}; self.encoder=tiktoken.get_encoding("gpt2")
        for family in ("coco","clotho","pamap2","mhealth","speech","ami"):
            path=CACHE/family/"rows.json"
            if path.exists(): self.rows[family]=[r for r in json.loads(path.read_text()) if r["role"]==role]
        # Old AMI contributes training only; its prior selection/test never does.
        if role=="train":
            folder=Path("artifacts/t2_audio_visual/broad_stream_prepared_v1/ami_train")
            legacy=json.loads((folder/"rows.json").read_text())
            for row in legacy:
                meeting=row["id"].rsplit("-",1)[0]
                self.rows.setdefault("ami",[]).append({**row,"role":"train","group":"OLD-"+meeting,"meeting":meeting,"pixel_file":str(folder/(row["id"]+"_pixels.npy")),"mel_file":str(folder/"mel.npy"),"previously_exposed":True,"other_speaker_spans":[(a,b) for other in legacy if other["id"].rsplit("-",1)[0]==meeting and other["actor"]!=row["actor"] for a,b,_ in other["spans"]]})
        self.robot_family=cfg["robot_family"];plan=ROOT/self.robot_family/"raw_metadata/scene_role_plan.json"
        if plan.exists():
            content=json.loads(plan.read_text()); assert content["suitable_for_building_holdout"]
            self.rows["droid"]=[r for r in content["rows"] if r["role"]==role]
        self.stats=np.load(CACHE/"normalizers.npz",allow_pickle=False)

    @lru_cache(maxsize=96)
    def load(self,path): return np.load(path,mmap_mode="r",allow_pickle=False)

    @lru_cache(maxsize=32768)
    def text_ids(self,text): return self.encoder.encode(text,allowed_special=set(),disallowed_special=())[:64] or [self.encoder.eot_token]

    def text_observation(self,captions):
        encoded=[self.text_ids(t) for t in captions]; length=max(map(len,encoded)); values=np.zeros((len(captions),length),np.int64); present=np.zeros_like(values,bool)
        for i,ids in enumerate(encoded):
            if captions[i].strip():values[i,:len(ids)]=ids;present[i,:len(ids)]=True
        return {**base_observation(len(captions),1,1.),"text":values,"text_present":present}

    def pick(self,family,rng,n):
        rows=self.rows[family]
        if family=="coco":return [rows[i] for i in rng.integers(0,len(rows),n)]
        if not hasattr(self,"group_sampling"):self.group_sampling={}
        if family not in self.group_sampling:
            groups={}
            minimum=1 if family=="clotho" else self.cfg["context"]+self.cfg["horizon"]+(self.cfg["future_audio_guard_packets"] if family in ("speech","ami") else 0)
            for row in rows:
                required=minimum*(2 if family in ("pamap2","mhealth") and row["rate_hz"]==100 else 1)
                if row.get("length",row.get("steps",0))>=required:groups.setdefault(row["group"],[]).append(row)
            indexed=[]
            for group in sorted(groups):
                current=groups[group]; counts=np.array([1 if family=="clotho" else r.get("length",r.get("steps",0))-minimum*(2 if family in ("pamap2","mhealth") and r["rate_hz"]==100 else 1)+1 for r in current],np.float64);indexed.append((current,counts/counts.sum()))
            if not indexed:raise ValueError(f"No {family}/{self.role} records support the registered context and horizon")
            self.group_sampling[family]=indexed
        indexed=self.group_sampling[family]; selected=[]
        for _ in range(n):
            current,probabilities=indexed[int(rng.integers(len(indexed)))];selected.append(current[int(rng.choice(len(current),p=probabilities))])
        return selected

    def make(self,family,position,seed=101,device="cpu",batch=None):
        rng=np.random.default_rng(seed*1000003+position); t=self.cfg["context"]; q=self.cfg["horizon"]
        b=batch or self.cfg["batch_static" if family=="coco" else "batch_temporal"]; rows=self.pick(family,rng,b)
        result={"family":family,"ids":[r["id"] for r in rows],"groups":[r["group"] for r in rows],"captions":[],"physical_observed_seconds":0.}
        if family=="coco":
            pixels=np.stack([self.load(str(CACHE/"coco"/r["array"]))[r["index"]] for r in rows]).astype(np.float32).transpose(0,3,1,2)/127.5-1
            obs=base_observation(b,1,1.); patch=rng.random((b,1,1,4,4))>.5; patch[:,:,:,0,0]=True
            obs.update(images=pixels[:,None,None],image_present=np.ones((b,1,1),bool),patch_present=patch.reshape(b,1,1,16),pixel_present=np.repeat(np.repeat(patch,16,3),16,4)[:,:,:,None])
            result.update(obs=obs,picture_target=pixels.reshape(b,3,16,4,16,4).mean((3,5))[:,None],query_time=np.zeros((b,1),np.float32))
            obs["images"]=np.where(obs["pixel_present"],obs["images"],0)
            result["captions"]=[r["captions"][int(rng.integers(len(r["captions"])))] for r in rows]
        elif family=="clotho":
            packets=[]
            for row in rows:
                mel=np.asarray(self.load(str(CACHE/"clotho"/row["file"])),np.float32).reshape(-1,100)
                # All15–30s caption content is represented; never label a random
                # 1s excerpt with an event that may occur elsewhere in the clip.
                edges=np.linspace(0,len(mel),t*4+1).astype(int)
                packets.append(np.stack([mel[edges[i]:max(edges[i]+1,edges[i+1])].mean(0) for i in range(t*4)]).reshape(t,4,100))
            values=normalize(np.stack(packets),self.stats["audio_mean"],self.stats["audio_scale"]); mask=rng.random((b,t,4))>.25; mask[:,0,0]=True
            obs=base_observation(b,t,1.); obs["time"]=np.stack([np.linspace(0,r["duration_seconds"],t,endpoint=False) for r in rows]).astype(np.float32)
            obs.update(audio=np.where(mask[...,None],values,0),audio_present=mask); baseline=np.stack([interpolate(v.reshape(-1,100),m.reshape(-1)).reshape(t,4,100) for v,m in zip(values,mask)])
            result.update(obs=obs,mel_target=values.reshape(b,t,400),mel_baseline=baseline.reshape(b,t,400),mel_hidden=(~mask).repeat(100,2).reshape(b,t,400),query_time=obs["time"].copy())
            result["captions"]=[r["captions"][int(rng.integers(5))] for r in rows]; result["physical_observed_seconds"]=sum(r["duration_seconds"] for r in rows)
        elif family in ("speech","ami"):
            values=[]; phones=[]; baselines=[]; masks=[]; future=[]; images=[]; shifted=[]; starts=[]; reference=[]; eligible=[]; correspondence=[]; modes=[]
            guard=self.cfg["future_audio_guard_packets"]
            for row in rows:
                length=row["length"]; maximum=length-t-guard-q
                if maximum<0: raise ValueError("Speech record too short for registered context/horizon")
                start=int(rng.integers(maximum+1))
                if family=="ami" and row["spans"] and rng.random()<.75:
                    span=row["spans"][int(rng.integers(len(row["spans"])))]; start=int(np.clip((span[0]+span[1])*.5/STEP-t*.5,0,maximum))
                at=row["offset"]+start; mel=np.asarray(self.load(row["mel_file"])[at:at+t],np.float32).reshape(t,4,100); target=normalize(mel,self.stats["audio_mean"],self.stats["audio_scale"])
                u=float(rng.random()); mode="repair" if family=="ami" or u<.7 else "audio" if u<.9 else "text_prefix"; present=np.ones(t*4,bool)
                if mode=="repair":
                    duration=float(rng.uniform(.064,.320)); center=float(rng.uniform(.35,t*STEP-.35)); present=mask_support(t*4,center-duration/2,center+duration/2)
                elif mode=="text_prefix": present[16:]=False
                baseline=interpolate(target.reshape(-1,100),present).reshape(t,400)
                values.append(target); masks.append(present.reshape(t,4)); baselines.append(baseline); starts.append(start); modes.append(mode)
                future.append(normalize(np.asarray(self.load(row["mel_file"])[at+t+guard:at+t+guard+q],np.float32).reshape(q,4,100),self.stats["audio_mean"],self.stats["audio_scale"]))
                a=start*STEP; end=(start+t)*STEP; spans=[(x,y,w) for x,y,w in row["spans"] if x>=a and y<=end]; text=" ".join(w for _,_,w in spans if "<unk>" not in w)
                result["captions"].append(text); reference.append(text)
                if family=="speech": phones.append(np.asarray(self.load(row["phone_file"])[at:at+t],np.int64))
                else:
                    phones.append(np.full((t,4),-1,np.int64)); pixel=self.load(row["pixel_file"]); images.append(np.asarray(pixel[start:start+t],np.float32)/127.5-1)
                    offset=int(rng.choice([-70,-24,-8,-4,4,8,24,70])); wrong=int(np.clip(start+offset,0,len(pixel)-t)); shifted.append(np.asarray(pixel[wrong:wrong+t],np.float32)/127.5-1)
                    own=sum(max(0,min(y,end)-max(x,a)) for x,y,_ in row["spans"]); other=sum(max(0,min(y,end)-max(x,a)) for x,y in row["other_speaker_spans"])
                    correspondence.append(own>t*STEP*.5 and other<t*STEP*.1 and abs(wrong-start)>=4)
                eligible.append(row.get("ctc_alignment_usable",True) and bool(text))
            obs=base_observation(b,t,STEP); obs.update(audio=np.where(np.stack(masks)[...,None],np.stack(values),0),audio_present=np.stack(masks)); result.update(obs=obs,mel_target=np.stack(values).reshape(b,t,400),mel_baseline=np.stack(baselines),mel_hidden=(~np.stack(masks)).repeat(100,2).reshape(b,t,400),phones=np.stack(phones),reference=reference,lexical_eligible=np.asarray(eligible,bool),modes=modes,query_time=obs["time"].copy(),future_time=np.broadcast_to((np.arange(q)+t+guard).astype(np.float32)*STEP,(b,q)).copy(),future_mel=np.stack(future).reshape(b,q,400),alignment_ids=[r["id"]+":"+str(start) for r,start in zip(rows,starts)])
            future_obs=base_observation(b,q,STEP); future_obs["time"]=result["future_time"].copy(); future_obs.update(audio=np.stack(future),audio_present=np.ones((b,q,4),bool)); result["future_obs"]=future_obs
            if family=="ami":
                picture=np.stack(images)[:,:,None,None]; picture=np.repeat(picture,3,3); wrong=np.stack(shifted)[:,:,None,None]; wrong=np.repeat(wrong,3,3)
                obs.update(images=picture,image_present=np.ones((b,t,1),bool)); result.update(wrong_images=wrong,correspondence_eligible=np.asarray(correspondence,bool))
            # Plain transcript is supplied only in the explicit text-prefix mode.
            if family=="speech":
                supplied=self.text_observation(result["captions"]); obs["text_present"]=supplied["text_present"]&np.asarray([m=="text_prefix" for m in modes])[:,None];obs["text"]=np.where(obs["text_present"],supplied["text"],0)
            result["physical_observed_seconds"]=b*t*STEP
        elif family in ("pamap2","mhealth"):
            values=[]; future=[]; labels=[]; clocks=[]
            for row in rows:
                data=self.load(str(CACHE/family/row["file"])); step=2 if row["rate_hz"]==100 else 1
                maximum=len(data)-(t+q)*step; start=int(rng.integers(maximum+1)); indices=start+np.arange(t+q)*step
                raw=np.asarray(data[indices],np.float32); all_values=np.full((t+q,64),np.nan,np.float32); all_values[:,:33]=normalize(raw,self.stats["numeric_mean"][:33],self.stats["numeric_scale"][:33]); values.append(all_values[:t]);future.append(all_values[t:])
                times=np.asarray(self.load(str(CACHE/family/row["time_file"]))[indices]);clocks.append((times-times[0]).astype(np.float32))
                if "activity_file" in row: labels.append(np.asarray(self.load(str(CACHE/family/row["activity_file"]))[indices[:t]],np.int64))
            original=np.stack(values); present=np.isfinite(original); observed=present&(rng.random(present.shape)>.15); obs=base_observation(b,t,.02);obs["time"]=np.stack(clocks)[:,:t];obs.update(numeric=np.where(observed,np.nan_to_num(original),0),numeric_present=observed)
            target=np.stack(future); future_obs=base_observation(b,q,.02);future_obs["time"]=np.stack(clocks)[:,t:];future_obs.update(numeric=np.nan_to_num(target),numeric_present=np.isfinite(target))
            result.update(obs=obs,numeric_target=np.nan_to_num(original),numeric_target_present=present,numeric_hidden=present&~observed,query_time=obs["time"].copy(),future_time=future_obs["time"].copy(),future_numeric=np.nan_to_num(target),future_numeric_present=np.isfinite(target),future_obs=future_obs,activity=np.stack(labels))
            result["numeric_baseline"]=numeric_baseline(obs["numeric"],observed,obs["time"])
            result["physical_observed_seconds"]=b*t*.02
        elif family=="droid":
            values=[]; pictures=[]; future_values=[]; future_pictures=[]; plans=[]; current_actions=[]
            for row in rows:
                folder=CACHE/self.robot_family/row["id"]; trajectory=self.load(str(folder/"trajectory.npz")); state=rotation6d(trajectory["observed"],trajectory["gripper"]); start=int(rng.integers(len(state)-t-q+1)); state_segment=state[start:start+t+q]
                numeric=np.full((t+q,64),np.nan,np.float32); numeric[:,ROBOT_STATE]=normalize(state_segment,self.stats["numeric_mean"][ROBOT_STATE],self.stats["numeric_scale"][ROBOT_STATE])
                # Current target command is absent. Only previously issued
                # commands enter the observed history, expressed at known poses.
                indices=start+np.arange(t); previous=(indices-1).clip(0); past=np.concatenate([pose_delta(trajectory["observed"][indices],trajectory["commanded"][previous]),trajectory["action"][previous,6:]],1)
                numeric[:t,ROBOT_ACTION]=normalize(past,self.stats["action_mean"],self.stats["action_scale"]);values.append(numeric[:t]);future_values.append(numeric[t:])
                numeric[:t][indices==0,ROBOT_ACTION]=np.nan
                # Action plans use the last observed reference pose throughout;
                # no future observed pose is needed to express the planned inputs.
                anchor=np.repeat(trajectory["observed"][start+t-1:start+t],q,axis=0); command=trajectory["commanded"][start+t-1:start+t+q-1]; grip=trajectory["action"][start+t-1:start+t+q-1,6:]
                plan=np.concatenate([pose_delta(anchor,command),grip],1); plans.append(normalize(plan,self.stats["action_mean"],self.stats["action_scale"]))
                current_actions.append(trajectory["action"][start+t-1])
                image=self.load(str(folder/"rgb.npy")); picture=np.asarray(image[start:start+t+q],np.float32).transpose(0,1,4,2,3)/127.5-1; pictures.append(picture[:t]);future_pictures.append(picture[t:])
                result["captions"].append(row["captions"][int(rng.integers(len(row["captions"])))] if row["captions"] else "")
            original=np.stack(values); obs=base_observation(b,t,1.,True); obs.update(numeric=np.nan_to_num(original),numeric_present=np.isfinite(original),images=np.stack(pictures),image_present=np.ones((b,t,3),bool))
            # Hold one real view out for cross-view reconstruction.
            hidden_view=int(rng.integers(3));cross_view=bool(rng.random()<.5)
            if cross_view:obs["image_present"][:,:,hidden_view]=False;obs["images"][:,:,hidden_view]=0
            targets=np.stack(future_values); future_obs=base_observation(b,q,1.,True); future_obs["time"]+=t;future_obs.update(numeric=np.nan_to_num(targets),numeric_present=np.isfinite(targets),images=np.stack(future_pictures),image_present=np.ones((b,q,3),bool))
            caption=self.text_observation(result["captions"]);obs.update(text=caption["text"],text_present=caption["text_present"])
            image_target=np.stack(pictures)[:,:,hidden_view].reshape(b,t,3,16,4,16,4).mean((4,6)); future_image=np.stack(future_pictures)[:,:,hidden_view].reshape(b,q,3,16,4,16,4).mean((4,6))
            result.update(obs=obs,numeric_target=np.nan_to_num(original),numeric_target_present=np.isfinite(original),numeric_hidden=np.zeros_like(original,bool),query_time=obs["time"].copy(),picture_target=image_target,future_time=future_obs["time"].copy(),future_numeric=np.nan_to_num(targets),future_numeric_present=np.isfinite(targets),future_picture=future_image,future_obs=future_obs,planned_actions=np.stack(plans).astype(np.float32),current_action=np.stack(current_actions).astype(np.float32),hidden_view=hidden_view)
            result["numeric_baseline"]=np.nan_to_num(original).copy()
            result["picture_baseline"]=np.zeros_like(image_target) if cross_view else image_target.copy()
            result["picture_hidden"]=cross_view
        else: raise ValueError(family)
        if result["captions"]:
            result["companion"]=self.text_observation(result["captions"])
            result["caption_hashes"]=[hashlib.sha256(t.strip().lower().encode()).hexdigest() for t in result["captions"]]
        return tensor_tree(result,device)
