"""Read the same four GRID commands from frozen 128-D fields with the input modality swapped."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from models.g4_shared import Spec
from models.g5_shared import encode
from training.g4_train import sha_file,write
from training.g4_transfer import balanced,ridge,support
from training.g5_batches import JointData,load_pack

VIEWS={"audio":0,"video":1,"text":3}
WORDS=("bin","lay","place","set")


class GridViews(JointData):
    """Use only GRID from the paired pack. No action or impact confirmation pack is opened."""
    def __init__(self,data_root,role,stats):
        base,_=load_pack(data_root/"paired"/role,False)
        self.packs=[base]
        self.pools=[np.flatnonzero(base["family"]==0)]
        self.stats=stats


def evaluate(data_root,run,out,confirmation=None):
    receipt=json.loads((run/"verified.json").read_bytes())
    card=json.loads((run/"config.json").read_bytes())
    if receipt["step"]!=2048 or sha_file(run/"inference.npz")!=receipt["sha256"]["inference.npz"]:
        raise ValueError("Require the registered, verified 2048-update weights")
    spec=Spec(**card["spec"])
    with np.load(run/"inference.npz",allow_pickle=False) as f:
        params={k:jax.device_put(f[k]) for k in f if k!="stats" and not k.startswith("prior_")}
        stats=np.array(f["stats"])
    train=JointData(data_root,"train",verify=False,stats=stats)
    if confirmation is None:
        dev=JointData(data_root,"dev",verify=False,stats=stats)
    else:
        lock=json.loads(confirmation.read_bytes())
        if len(lock["sha256"])!=8 or lock["sha256"].get(card["run"])!=receipt["sha256"]["inference.npz"]:
            raise ValueError("Speaker confirmation requires all eight fixed weights")
        dev=GridViews(data_root,"test",stats)
        target=dev.packs[0]; ids=dev.pools[0]
        valid_people=set(target["group"][ids].tolist()); valid_sentences={tuple(s) for s in target["x"][ids]}
        if len(valid_people)!=8: raise ValueError("Unexpected confirmation subjects")
        for source in (train,GridViews(data_root,"dev",stats)):
            orig=source.packs[0]; rows=source.pools[0]
            if valid_people.intersection(orig["group"][rows].tolist()) or valid_sentences.intersection(tuple(s) for s in orig["x"][rows]):
                raise ValueError("A speaker or sentence leaked from support or development")

    selected={}; labels={}; groups={}; captions={}; arrays={}
    for role,pack in (("train",train),("dev",dev)):
        a=pack.packs[0]; idx=pack.pools[0]
        idx=idx[np.argsort(a["uid"][idx])]
        command=[bytes(int(t)-3 for t in a["x"][i] if t>=3).decode("utf8").split()[0].lower() for i in idx]
        y=np.array([WORDS.index(s) for s in command])
        if role=="train":
            take=support(y,32); idx=idx[take]; y=y[take]
        selected[role]=idx; labels[role]=y; groups[role]=np.asarray(a["group"][idx]); captions[role]=np.asarray(a["cap"][idx])
        arrays[role]=pack
    if np.intersect1d(groups["train"],groups["dev"]).size:
        raise ValueError("Support and development speakers overlap")
    @jax.jit
    def extract(b,src):
        z,_=encode(params,spec,b,src)
        z=z.astype(jnp.float32); valid=b["ok"].astype(jnp.float32)
        m=valid.reshape(len(z),4,16,1)
        parts=(z.reshape(len(z),4,16,128)*m).sum(2)/jnp.maximum(m.sum(2),1)
        vec=(z*valid[...,None]).sum(1)/jnp.maximum(valid.sum(1,keepdims=True),1)
        vec=vec*jax.lax.rsqrt((vec**2).sum(1,keepdims=True)+1e-8)
        return parts.reshape(len(z),512),vec
    rep={}; embed={}; raw={}
    for role,pack in arrays.items():
        ix=selected[role]
        for view,mode in VIEWS.items():
            features=[]; means=[]; observed=[]
            for start in range(0,len(ix),16):
                rows=ix[start:start+16]; pad=np.pad(rows,(0,16-len(rows)),mode="edge")
                b=pack.gather(np.zeros(16,int),pad,np.zeros(16,int))
                # The command label and every unselected stream are unavailable to the encoder.
                for key,i in (("a",0),("v",1),("input_x",3)):
                    if i!=mode: b[key].fill(0)
                b["n"].fill(0); b["u"].fill(0)
                for key in ("a","v"):
                    v=b[key].astype(np.float32)
                    b[key]=np.clip(v+.65*(v-np.concatenate((v[:,:1],v[:,:-1]),1)),0,255)
                source=np.zeros((16,4),np.uint8); source[:,mode]=1
                field,vec=extract(b,source)
                features.append(np.asarray(field)[:len(rows)]); means.append(np.asarray(vec)[:len(rows)])
                sensed=(np.eye(259,dtype=np.float32)[b["input_x"]] if mode==3 else
                        b["a" if mode==0 else "v"]/255).reshape(16,4,16,-1)
                mask=b["ok"].reshape(16,4,16,1)
                pooled=(sensed*mask).sum(2)/np.maximum(mask.sum(2),1)
                observed.append(pooled.reshape(16,-1)[:len(rows)])
            rep[role,view]=np.concatenate(features); embed[role,view]=np.concatenate(means); raw[role,view]=np.concatenate(observed)
    y=labels["dev"]; g=groups["dev"]; unique=np.unique(g)
    perm=np.arange(len(y)); rng=np.random.default_rng(61401)
    for group in unique:
        ids=np.flatnonzero(g==group)
        perm[ids]=rng.permutation(ids)
    result={}
    for shots in (8,32):
        take=support(labels["train"],shots)
        cells={}; raw_same={}; per_group={str(int(group)):[] for group in unique}
        for source in VIEWS:
            raw_pred=ridge(raw["train",source][take],labels["train"][take],raw["dev",source])
            raw_same[source]=balanced(y,raw_pred)
            for target in VIEWS:
                guess=ridge(rep["train",source][take],labels["train"][take],rep["dev",target])
                by={str(int(group)):balanced(y[g==group],guess[g==group]) for group in unique}
                cells[source+"-to-"+target]={"balanced_accuracy":balanced(y,guess),
                    "shuffled_within_speaker":balanced(y,guess[perm]),"by_speaker":by}
                if source!=target:
                    for group in by: per_group[group].append(by[group])
        result[str(shots)]={"cells":cells,"raw_within_view":raw_same,
            "mean_cross_view":float(np.mean([v["balanced_accuracy"] for k,v in cells.items() if k.split("-to-")[0]!=k.split("-to-")[1]])),
            "cross_view_by_speaker":{group:float(np.mean(v)) for group,v in per_group.items()}}
    retrieval={}
    cap=captions["dev"]
    for s in VIEWS:
        for t in VIEWS:
            if s==t: continue
            best=[]; chance=[]
            for group in unique:
                ids=np.flatnonzero(g==group); scores=embed["dev",s][ids]@embed["dev",t][ids].T
                best.extend(cap[ids[scores.argmax(1)]]==cap[ids])
                chance.extend((cap[ids,None]==cap[None,ids]).mean(1))
            retrieval[s+"-to-"+t]={"same_caption_top1":float(np.mean(best)),"conditioned_chance":float(np.mean(chance))}
    population=({"development_speakers":len(unique),"development_clips":len(y)} if confirmation is None else
                {"held_speakers":len(unique),"held_clips":len(y),"role":"speaker-and-sentence confirmation","lock_sha256":sha_file(confirmation)})
    write(out,{"run":card["run"],"scale_params":card["params"],"seed":card["seed"],"step":receipt["step"],
               "inference_sha256":receipt["sha256"]["inference.npz"],"support_speakers":len(np.unique(groups["train"])),
               **population,"classes":WORDS,"chance":.25,
               "shots":result,"same_speaker_retrieval":retrieval})
    print(card["run"],"8-shot cross-view",result["8"]["mean_cross_view"],"32-shot",result["32"]["mean_cross_view"],flush=True)


if __name__=="__main__":
    q=argparse.ArgumentParser(description=__doc__); q.add_argument("--data",type=Path,required=True)
    q.add_argument("--run",type=Path,required=True); q.add_argument("--out",type=Path,required=True)
    a=q.parse_args(); evaluate(a.data,a.run,a.out)
