"""Encode separately cropped physical sensor streams on the same clock for all alignment alternatives."""
import argparse
from pathlib import Path
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
from models.g5_shared import encode
from training.g7_policy import base_weights
from training.g7_units import put
from training.g5_batches import load_pack
from training.g4_data import digest


def pairing(src,family):
    n=len(src["uid"]);length=np.array(src.get("length",np.full(n,64)),int);group=src["group"]
    labels=src["phrase"] if family=="crema" else src["recording"]
    wrong=np.full(n,-1,int)
    for g in np.unique(group):
        members=np.flatnonzero(group==g)
        for lab in np.unique(labels[members]):
            other=members[(labels[members]!=lab)&(length[members]>20)]
            if not len(other):continue
            for i in members[labels[members]==lab]:
                dist=np.abs(length[other]-length[i]);best=other[dist==dist.min()]
                wrong[i]=best[int(src["uid"][i])%len(best)]
    keep=np.flatnonzero((wrong>=0)&(length>20));keep=keep[np.argsort(src["uid"][keep])]
    return keep,wrong,length


def raw_pair(audio,video,mask):
    a=np.maximum(0,np.diff(audio,axis=1,prepend=audio[:,:1])).mean(-1);v=np.abs(np.diff(video,axis=1,prepend=video[:,:1])).mean(-1)
    n=mask.sum(-1,keepdims=True);a=a-(a*mask).sum(-1,keepdims=True)/n
    v=v-(v*mask).sum(-1,keepdims=True)/n
    return ((a*v)*mask).sum(-1)/np.sqrt(((a*a*mask).sum(-1)*(v*v*mask).sum(-1)).clip(1e-8))


def build(a):
    tag=f"{a.core}-{a.seed}"
    if a.core=="raw":
        rng=np.random.default_rng(77551);pa=jnp.asarray(rng.normal(0,.1,(100,128)),jnp.float32);pv=jnp.asarray(rng.normal(0,.0625,(256,128)),jnp.float32)
        p={};spec=None;h="fixed-raw-projection-77551"
    else:spec,p,_,_,h=base_weights(a.parents,a.core,a.seed)
    @jax.jit
    def infer(p,audio,video,kind,mask,sense):
        av=jnp.clip(audio+.65*(audio-jnp.concatenate((audio[:,:1],audio[:,:-1]),1)),0,255)
        vv=jnp.clip(video+.65*(video-jnp.concatenate((video[:,:1],video[:,:-1]),1)),0,255)
        if a.core=="raw":
            return jnp.where((kind==0)[:,None,None],(av/48-7/3)@pa,(vv/127.5-1)@pv)*sense[...,None]
        b,t=mask.shape
        edge={"a":av,"v":vv,"n":jnp.zeros((b,t,17)),"u":jnp.zeros((b,t,2)),"input_x":jnp.zeros((b,t),jnp.int32),"ok":mask,"sense":sense}
        z,_=encode(p,spec,edge,jax.nn.one_hot(kind,4,dtype=jnp.uint8));return z.astype(jnp.float32)
    for family in ("crema","events"):
        dest=a.root/"physical-time/cache"/tag/family/a.role
        if (dest/"card.json").exists():continue
        origin=a.root/"physical-time"/family/"confirm" if a.role=="confirm" else a.root/("speech/crema" if family=="crema" else "fresh/events")/a.role
        src,_=load_pack(origin,True); rows,donors,lengths=pairing(src,family);rows=rows[:4096] if a.role=="train" else rows
        dest.parent.mkdir(exist_ok=True,parents=True)
        with tempfile.TemporaryDirectory(dir=dest.parent,prefix=".cache-") as t:
            tmp=Path(t)/"packet";tmp.mkdir()
            z=np.lib.format.open_memmap(tmp/"field.npy",mode="w+",dtype=np.float16,shape=(len(rows),5,48,128))
            valid=np.zeros((len(rows),48),np.uint8);raw=np.zeros((len(rows),4),np.float32)
            for j in range(0,len(rows),24):
                ix=rows[j:j+24];nn=len(ix);ix=np.pad(ix,(0,24-nn),mode="edge");donor=donors[ix]
                length=np.minimum(lengths[ix],lengths[donor])-16; keep=np.arange(48)[None]<length[:,None]
                audio=np.asarray(src["a"][ix,8:56],np.float32);vid=src["v"][ix].astype(np.float32)
                allv=[vid[:,8:56],vid[:,:48],vid[:,16:64],src["v"][donor,8:56].astype(np.float32)]
                stacked=np.concatenate([np.zeros_like(vid[:,:48]),*allv])
                all_a=np.concatenate([audio,*[np.zeros_like(audio) for _ in range(4)]])
                mask=np.tile(keep,(5,1));kind=np.repeat([0,1,1,1,1],24)
                fields=np.asarray(infer(p,all_a,stacked,kind,mask,mask)).reshape(5,24,48,128).transpose(1,0,2,3)
                z[j:j+nn]=fields[:nn];valid[j:j+nn]=keep[:nn]
                raw[j:j+nn]=np.stack([raw_pair(audio,x,keep) for x in allv],axis=1)[:nn]
                if not j:
                    blank=np.asarray(infer(p,all_a,stacked,kind,mask,np.zeros_like(mask))).reshape(5,24,48,128)
                    for alt in (2,3,4):np.testing.assert_allclose(blank[1],blank[alt],atol=0,rtol=0)
            z.flush();np.savez(tmp/"rows.npz",uid=src["uid"][rows],group=src["group"][rows],mask=valid,raw=raw,donor=src["uid"][donors[rows]])
            put(tmp/"card.json",{"role":a.role,"family":family,"rows":len(rows),"groups":len(np.unique(src["group"][rows])),
                "source":digest(origin/"card.json"),"parent":h,"physical_offsets":[0,-8,8],"sensor_erasure_alternatives_identical":True,"row_hash":digest(tmp/"rows.npz")})
            tmp.rename(dest)
        print("Clock-safe fields",tag,family,a.role,len(rows),flush=True)


if __name__=="__main__":
    q=argparse.ArgumentParser();q.add_argument("--root",type=Path,required=True);q.add_argument("--parents",type=Path,required=True)
    q.add_argument("--core",choices=("hybrid","gpt2","raw"),required=True);q.add_argument("--seed",type=int,required=True)
    q.add_argument("--role",choices=("train","confirm"),required=True);build(q.parse_args())
