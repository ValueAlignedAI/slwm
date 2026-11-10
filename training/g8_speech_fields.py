"""Prepare fresh frozen fields for genuine, erased and same-speaker wrong sources on identical recipient clocks."""
import argparse
from pathlib import Path
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
from models.g5_shared import encode
from training.g7_align_cache import mask_for
from training.g7_policy import base_weights
from training.g7_units import put
from training.g8_acoustic_codec import DATA
from scripts.train.g7_lock_heads import sha
from scripts.train.g6_artifacts import roots
from training.g8_models import read


def donors(src):
    y=np.full(len(src["uid"]),-1,int)
    for i,(g,phrase) in enumerate(zip(src["group"],src["phrase"])):
        ids=np.flatnonzero((src["group"]==g)&(src["phrase"]!=phrase))
        if len(ids):
            gap=np.abs(src["length"][ids].astype(int)-int(src["length"][i]));best=ids[gap==gap.min()]
            y[i]=best[int(src["uid"][i])%len(best)]
    return y


def build(tag,data_root=DATA):
    core,seed=tag.split("-");parents,_=roots("g6")
    spec,p,_,_,h=base_weights(parents,core,int(seed))
    @jax.jit
    def infer(p,b,s):return encode(p,spec,b,s)[0].astype(jnp.float32)
    for family in ("libri","crema"):
        folder=data_root/"confirm"/family;dest=data_root/"fields"/tag/family
        if (dest/"card.json").exists():continue
        card=read(folder/"card.json");src={k:np.load(folder/(k+".npy"),mmap_mode="r") for k in card["sha256"] if k not in ("pcm","h","fine")}
        wrong=donors(src);length=np.array(src["length"],int);n=len(wrong);views=[0,3]+([1] if family=="crema" else [])
        dest.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dest.parent) as t:
            target=Path(t)/"set";target.mkdir();np.save(target/"donor.npy",wrong)
            for mode in views:
                for cond in ("correct","wrong","erased"):
                    result=np.lib.format.open_memmap(target/f"z{mode}-{cond}.npy",mode="w+",dtype=np.float16,shape=(n,64,128))
                    for j in range(0,n,32):
                        ix=np.arange(j,min(j+32,n));size=len(ix);ix=np.pad(ix,(0,32-size),mode="edge")
                        ref=np.where(wrong[ix]>=0,wrong[ix],ix) if cond=="wrong" else ix
                        valid=np.arange(64)[None]<length[ix,None]
                        at=np.rint(np.arange(64)[None]*(length[ref,None]-1)/np.maximum(length[ix,None]-1,1)).astype(int).clip(0,63)
                        blank=mask_for(src["uid"][ix],length[ix]) if mode==0 else np.zeros_like(valid)
                        av=np.zeros((32,64,100),np.float32);vv=np.zeros((32,64,256),np.float32);x=np.zeros((32,64),np.int32)
                        if mode in (0,1):
                            val=np.take_along_axis(src["a" if mode==0 else "v"][ref],at[...,None],axis=1).astype(np.float32)
                            if mode==0:val[blank]=112
                            val=np.clip(val+.65*(val-np.concatenate((val[:,:1],val[:,:-1]),1)),0,255)
                            if mode==0:av=val
                            else:vv=val
                        else:
                            ids=src["x"][ref];nt=np.maximum(1,(ids>0).sum(1))
                            x=np.take_along_axis(ids,np.arange(64)[None]*(nt[:,None]-1)//63,axis=1)
                        sense=valid&~blank if cond!="erased" else np.zeros_like(valid)
                        b={"a":av,"v":vv,"n":np.zeros((32,64,17),np.float32),"u":np.zeros((32,64,2),np.float32),"input_x":x,"ok":valid,"sense":sense}
                        route=np.tile(np.eye(4,dtype=np.uint8)[mode],(32,1));result[j:j+size]=np.asarray(infer(p,b,route))[:size].astype(np.float16)
                    result.flush()
            put(target/"card.json",{"rows":n,"parent":h,"views":views,"source":sha(folder/"card.json"),
                "sha256":{f.name:sha(f) for f in sorted(target.glob("*.npy"))}})
            target.rename(dest)
        print("Held source fields",tag,family,n,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("tag");p.add_argument("--data-root",type=Path,default=DATA);a=p.parse_args();build(a.tag,a.data_root)
