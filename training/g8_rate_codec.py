"""Allocate an exact fixed payload across train-fitted scalar PCA quantizers and verify physical bitstreams."""
import argparse
from pathlib import Path
import numpy as np
from training.g8_acoustic_codec import source_sample,codebook,OUT,OLD,DATA
from training.g7_units import put
from scripts.train.g7_lock_heads import sha

OUT_RATE=OUT.parent/"rate-codec"
HELD=DATA.parent/"speech-rate-confirm"
BUDGETS=(6,24,48,96,192)


def scalar_book(values,bits):
    x=np.sort(values.astype(np.float64));k=1<<bits
    prefix=np.r_[0.,np.cumsum(x)];prefix2=np.r_[0.,np.cumsum(x*x)]
    center=np.quantile(x,(np.arange(k)+.5)/k)
    for _ in range(48):
        bounds=np.r_[0,np.searchsorted(x,(center[:-1]+center[1:])*.5,side="right"),len(x)]
        count=np.diff(bounds);sums=np.diff(prefix[bounds]);updated=np.where(count>0,sums/np.maximum(1,count),center)
        if np.max(np.abs(updated-center))<1e-7:center=updated;break
        center=updated
    bounds=np.r_[0,np.searchsorted(x,(center[:-1]+center[1:])*.5,side="right"),len(x)]
    err=(np.diff(prefix2[bounds])-2*center*np.diff(prefix[bounds])+center*center*np.diff(bounds)).sum()/len(x)
    return center.astype(np.float32),max(0.,float(err))


def allocation(costs,budget):
    d,maxbits=costs.shape;maxbits-=1
    dp=np.full((d+1,budget+1),np.inf);dp[0,0]=0;back=np.zeros((d+1,budget+1),np.int16)
    for i in range(d):
        for bits in range(min(maxbits,budget)+1):
            alt=dp[i,:budget+1-bits]+costs[i,bits];view=dp[i+1,bits:];improves=alt<view
            view[improves]=alt[improves];back[i+1,bits:][improves]=bits
    remaining=budget;parts=np.zeros(d,np.int16)
    for i in range(d,0,-1):parts[i-1]=back[i,remaining];remaining-=parts[i-1]
    if remaining!=0 or parts.sum()!=budget:raise ValueError("Incorrect fixed payload allocation")
    return parts


def fit_codec():
    OUT_RATE.mkdir(exist_ok=True,parents=True)
    if (OUT_RATE/"complete.json").exists():return
    base=codebook();rng=np.random.default_rng(81208)
    x=np.concatenate([source_sample(f,rng) for f in ("libri","crema")]);pc=(x-base["mean"])@base["basis"]
    centers=np.zeros((64,9,256),np.float32);mse=np.zeros((64,9))
    for j in range(64):
        for bits in range(9):
            c,error=scalar_book(pc[:,j],bits);centers[j,bits,:len(c)]=c;mse[j,bits]=error
    payload={"centers":centers,"cost":mse,"mean":base["mean"],"basis":base["basis"],"std":base["std"]}
    draws={}
    for budget in BUDGETS:
        equal=np.full(64,budget//64,np.int16);equal[:budget%64]+=1
        optimal=allocation(mse,budget)
        for name,parts in (("equal",equal),("allocated",optimal)):
            payload[f"{name}{budget}"]=parts
            draws[f"{name}{budget}"]={"nominal_bps":budget*46.875,"bits_per_coordinate":parts.tolist(),"training_pc_mse":float(mse[np.arange(64),parts].sum()/200)}
        if draws[f"allocated{budget}"]["training_pc_mse"]>draws[f"equal{budget}"]["training_pc_mse"]+1e-9:raise ValueError("Allocation did not minimize training distortion")
    with (OUT_RATE/"weights.part").open("wb") as f:np.savez(f,**payload)
    (OUT_RATE/"weights.part").replace(OUT_RATE/"weights.npz")
    put(OUT_RATE/"complete.json",{"weights":sha(OUT_RATE/"weights.npz"),"pca_parent":sha(OUT/"codec.npz"),"train_frame_pairs":len(x),"rates":draws,
        "meaning":"Fixed-rate scalar payloads. Static codebooks and byte alignment are excluded from nominal bit rate."})
    print("Frozen rate allocator",sha(OUT_RATE/"weights.npz"),flush=True)


def load_book():
    with np.load(OUT_RATE/"weights.npz") as f:return {k:np.array(v) for k,v in f.items()}


def coefficients(fine,b):
    return (fine.reshape(-1,200).astype(np.float32)/16-12-b["mean"])@b["basis"]


def quantize(pc,b,alloc):
    x=pc.reshape(-1,64);codes=np.empty(x.shape,np.uint8);recon=np.empty_like(x,dtype=np.float32)
    for c,nb in enumerate(alloc):
        centers=b["centers"][c,nb,:(1<<int(nb))]
        code=np.searchsorted((centers[:-1]+centers[1:])*.5,x[:,c],side="right")
        codes[:,c]=code;recon[:,c]=centers[code]
    return codes,recon


def pack(codes,alloc):
    parts=[]
    for j,nb in enumerate(alloc):
        if nb:parts.append((codes[:,j,None]>>np.arange(nb-1,-1,-1))&1)
    bits=np.concatenate(parts,axis=1).astype(np.uint8)
    if bits.shape[1]!=sum(alloc):raise ValueError("Wrong payload width")
    return np.packbits(bits.reshape(-1)).tobytes()


def unpack(blob,alloc,n):
    width=int(sum(alloc));raw=np.unpackbits(np.frombuffer(blob,np.uint8))
    if len(blob)!=(n*width+7)//8 or raw[n*width:].any():raise ValueError("Malformed acoustic bitstream")
    bits=raw[:n*width].reshape(n,width);out=np.zeros((n,len(alloc)),np.uint8);at=0
    for j,nb in enumerate(alloc):
        if nb:
            out[:,j]=bits[:,at:at+nb]@(1<<np.arange(nb-1,-1,-1));at+=nb
    return out


def mel(pc,b,n):
    return (pc.reshape(-1,64)@b["basis"].T+b["mean"]).reshape(n,256,100).clip(-12,3.9375)


def verify_clip_payload(codes,b,alloc):
    payload=pack(codes,alloc);restored=unpack(payload,alloc,len(codes))
    np.testing.assert_array_equal(codes,restored)
    decoded=np.column_stack([b["centers"][j,k,restored[:,j]] for j,k in enumerate(alloc)])
    return payload,decoded


if __name__=="__main__":fit_codec()
