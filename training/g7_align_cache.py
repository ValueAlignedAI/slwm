"""Cache exact clean and span-masked states of the matched frozen cores on genuinely present modalities."""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
from models.g5_shared import encode
from training.g7_policy import base_weights
from training.g4_data import digest
from training.g7_units import put

CAP={"train":{"libri":32768,"crema":20000,"events":20000,"oldimpact":8192},
     "dev":{"libri":1024,"crema":1000,"events":8000,"oldimpact":1800},
     "test":{"libri":1024,"crema":1000,"events":8000,"oldimpact":1800}}


def source(root,old,family,role):
    if family in ("libri","crema"):folder=root/"speech"/family/role
    elif family=="events":folder=root/"fresh/events"/role
    else:folder=old/"impact"/role
    card=json.loads((folder/"card.json").read_bytes())
    return {k:np.load(folder/(k+".npy"),mmap_mode="r",allow_pickle=False) for k in card["sha256"]},digest(folder/"card.json")


def mask_for(ids,length):
    # Reproducible spans make clean-versus-masked comparisons paired across architectures.
    ids=np.asarray(ids,np.uint64); size=np.maximum(1,np.rint(length*.25).astype(int))
    start=(ids%np.maximum(1,length-size+1).astype(np.uint64)).astype(int)
    return (np.arange(64)[None]>=start[:,None])&(np.arange(64)[None]<(start+size)[:,None])


def build(a):
    spec,p,_,_,h=base_weights(a.parents,a.core,a.seed)
    @jax.jit
    def infer(params,b,src):
        z,_=encode(params,spec,b,src);return z.astype(jnp.float32)
    for role in ("train","dev","test"):
        for family in ("crema","libri","events","oldimpact"):
            dest=a.root/"align-features"/f"{a.core}-{a.seed}"/family/role
            if (dest/"card.json").is_file():continue
            src,pin=source(a.root,a.old,family,role);uid=np.asarray(src.get("uid",np.arange(len(src["a"]),dtype=np.uint64)))
            order=np.argsort(uid) if "uid" in src else np.random.default_rng(71110).permutation(len(uid))
            ix=order[:CAP[role][family]]
            dest.parent.mkdir(exist_ok=True,parents=True)
            with tempfile.TemporaryDirectory(dir=dest.parent,prefix=".g7-view-") as td:
                tmp=Path(td)/"build";tmp.mkdir();np.save(tmp/"idx.npy",ix)
                kinds=[0]+([1] if "v" in src else [])+([3] if "x" in src else [])
                for mode in kinds:
                    for masked in (False,True):
                        file=tmp/f"z{mode}-{'mask' if masked else 'clean'}.npy"
                        result=np.lib.format.open_memmap(file,mode="w+",dtype=np.float16,shape=(len(ix),64,128))
                        for j in range(0,len(ix),64):
                            ids=ix[j:j+64];n=len(ids);ids=np.pad(ids,(0,64-n),mode="edge")
                            length=src["length"][ids].astype(int) if "length" in src else np.full(64,64)
                            keep=np.arange(64)[None]<length[:,None]
                            blank=mask_for(uid[ids],length) if masked else np.zeros_like(keep)
                            a0=np.zeros((64,64,100),np.float32);v0=np.zeros((64,64,256),np.float32);x=np.zeros((64,64),np.uint16)
                            if mode==0:
                                a0=src["a"][ids].astype(np.float32);a0[blank]=112
                                a0=np.clip(a0+.65*(a0-np.concatenate([a0[:,:1],a0[:,:-1]],1)),0,255)
                            if mode==1:
                                v0=src["v"][ids].astype(np.float32);v0[blank]=127.5
                                v0=np.clip(v0+.65*(v0-np.concatenate([v0[:,:1],v0[:,:-1]],1)),0,255)
                            if mode==3:
                                text=src["x"][ids];lengths=np.maximum(1,(text>0).sum(1));at=np.arange(64)[None]*(lengths[:,None]-1)//63
                                x=np.take_along_axis(text,at,axis=1);x[blank]=0
                            b={"ok":keep.astype(np.uint8),"sense":(keep&~blank).astype(np.uint8),"a":a0,"v":v0,"input_x":x,
                               "n":np.zeros((64,64,17),np.float32),"u":np.zeros((64,64,2),np.float32)}
                            source_mask=np.zeros((64,4),np.uint8);source_mask[:,mode]=1
                            result[j:j+n]=np.asarray(infer(p,b,source_mask))[:n].astype(np.float16)
                        result.flush()
                put(tmp/"card.json",{"role":role,"family":family,"rows":len(ix),"parent_sha256":h,"source_card":pin,"views":kinds,
                                    "sample_identity":digest(tmp/"idx.npy"),"mask_fraction":.25})
                tmp.rename(dest)
            print("Frozen views",a.core,a.seed,family,role,len(ix),flush=True)


if __name__=="__main__":
    q=argparse.ArgumentParser();q.add_argument("--root",type=Path,required=True);q.add_argument("--old",type=Path,required=True)
    q.add_argument("--parents",type=Path,required=True);q.add_argument("--core",choices=("hybrid","gpt2"),required=True)
    q.add_argument("--seed",choices=(711,712),type=int,required=True);build(q.parse_args())
