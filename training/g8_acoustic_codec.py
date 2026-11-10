"""Fit a nested acoustic product quantizer on training people and encode 46.875-Hz frame pairs."""
import argparse
from pathlib import Path
import numpy as np
from training.g7_units import fields,put
from training.g4_data import digest

OLD=Path("/mnt/slwm-data/SLWM/g7")
DATA=Path("/mnt/slwm-data/SLWM/g8/speech")
OUT=Path("/mnt/slwm-checkpoints-a/SLWM/g8/acoustic-codec")


def source_sample(family,rng):
    root=OLD/"speech"/family/"train";a=fields(root,("fine","length","group"));groups=np.unique(a["group"])
    people=rng.choice(groups,65536);lookup={g:np.flatnonzero(a["group"]==g) for g in groups}
    ids=np.array([rng.choice(lookup[g]) for g in people]);at=(rng.random(len(ids))*a["length"][ids]).astype(int)
    slots=rng.integers(2,size=len(ids));fine=a["fine"][ids,at]
    return fine[np.arange(len(ids))[:,None],2*slots[:,None]+np.arange(2)].reshape(-1,200).astype(np.float32)/16-12


def closest(x,centers):
    return (np.sum(centers**2,axis=1)-2*x@centers.T).argmin(1)


def fit():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"complete.json").is_file():return
    rng=np.random.default_rng(81208);x=np.concatenate([source_sample(f,rng) for f in ("libri","crema")])
    mean=x.mean(0,dtype=np.float64).astype(np.float32);centered=x-mean
    cov=np.zeros((200,200),np.float64)
    for block in np.array_split(centered,32):
        v=block.astype(np.float64);cov+=v.T@v
    eigen,basis=np.linalg.eigh(cov/len(x));basis=basis[:,-64:][:,::-1].astype(np.float32)
    sign=np.sign(basis[np.abs(basis).argmax(0),np.arange(64)]);basis*=sign
    std=np.sqrt(np.maximum(eigen[-64:][::-1],1e-5)).astype(np.float32)
    components=((centered@basis)/std).reshape(-1,8,8)
    books=[];counts=[]
    for g in range(8):
        values=np.ascontiguousarray(components[:,g]);centers=values[rng.choice(len(values),64,replace=False)].copy()
        for epoch in range(24):
            totals=np.zeros((64,8),np.float64);freq=np.zeros(64,np.int64)
            for block in np.array_split(values,16):
                codes=closest(block,centers);freq+=np.bincount(codes,minlength=64)
                for dim in range(8):totals[:,dim]+=np.bincount(codes,weights=block[:,dim],minlength=64)
            centers=np.where(freq[:,None]>0,totals/np.maximum(freq,1)[:,None],values[rng.integers(len(values),size=64)]).astype(np.float32)
        books.append(centers);counts.append(freq)
    with (OUT/"codec.part").open("wb") as f:np.savez(f,mean=mean,basis=basis,std=std,books=np.stack(books),counts=np.stack(counts))
    (OUT/"codec.part").replace(OUT/"codec.npz")
    put(OUT/"complete.json",{"hash":digest(OUT/"codec.npz"),"train_samples":len(x),"balanced_samples_per_corpus":65536,
        "groups":8,"centers_per_group":64,"paired_fine_frames":2,"symbols_per_second":46.875,"nominal_rates_bps":[281.25,1125.,2250.],
        "pca_explained_fraction":float(eigen[-64:].sum()/eigen.sum()),"training_sources":{f:digest(OLD/"speech"/f/"train/card.json") for f in ("crema","libri")}})
    print("Frozen acoustic quantizer",digest(OUT/"codec.npz"),flush=True)


def codebook():
    with np.load(OUT/"codec.npz") as f:return {k:np.array(v) for k,v in f.items()}


def encode(fine,book):
    shape=fine.shape[:-2];x=fine.reshape(-1,200).astype(np.float32)/16-12
    z=(((x-book["mean"])@book["basis"])/book["std"]).reshape(-1,8,8)
    result=np.stack([closest(np.ascontiguousarray(z[:,g]),book["books"][g]) for g in range(8)],axis=-1)
    return result.reshape(*shape,2,8).astype(np.uint8)


def decode(codes,book,groups=8):
    if groups not in (1,4,8):raise ValueError("Unregistered acoustic rate")
    values=book["books"][np.arange(8),codes.astype(int)].copy();values[...,groups:,:]=0
    flat=values.reshape(-1,64)*book["std"]
    return (flat@book["basis"].T+book["mean"]).reshape(len(codes),256,100).clip(-12,3.9375)


def encode_training():
    book=codebook()
    for family in ("libri","crema"):
        root=OLD/"speech"/family/"train";out=DATA/"codes"/family;out.mkdir(exist_ok=True,parents=True)
        if (out/"card.json").is_file():continue
        source=np.load(root/"fine.npy",mmap_mode="r");result=np.lib.format.open_memmap(out/"train.partial.npy",mode="w+",dtype=np.uint8,shape=(len(source),64,2,8))
        for j in range(0,len(source),96):result[j:j+96]=encode(source[j:j+96],book)
        result.flush();(out/"train.partial.npy").replace(out/"train.npy")
        put(out/"card.json",{"rows":len(source),"codes":digest(out/"train.npy"),"codec":digest(OUT/"codec.npz"),"source":digest(root/"card.json")})
        print("Acoustic training labels",family,len(source),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=("fit","labels","all"));a=p.parse_args()
    if a.phase in ("fit","all"):fit()
    if a.phase in ("labels","all"):encode_training()
