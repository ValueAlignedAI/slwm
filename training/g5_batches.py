"""Verified, family-stratified host batches. Only present views can condition a prediction."""
import hashlib
import json
from pathlib import Path

import numpy as np

from training.g4_data import digest

CONFIG=Path("configs/t5/joint_world.json")
FAMILIES=("grid","libri","coco","clotho","droid","text","ami","impact","pusht")


def load_pack(path,verify):
    card=json.loads((path/"card.json").read_bytes())
    arrays={}
    for k,expected in card["sha256"].items():
        file=path/(k+".npy")
        if verify and digest(file)!=expected:
            raise ValueError(f"Changed data array {file}")
        arrays[k]=np.load(file,mmap_mode="r",allow_pickle=False)
    return arrays,card


class JointData:
    def __init__(self,root,role,verify=True,stats=None):
        self.role=role
        self.packs=[]
        cards=[]
        for family in ("paired","impact","pusht"):
            arr,card=load_pack(root/family/role,verify)
            self.packs.append(arr)
            cards.append(card)
        base,imp,sim=self.packs
        self.pools=[np.flatnonzero(base["family"]==n) for n in range(7)]+[
            np.arange(len(imp["group"])),np.arange(len(sim["group"]))]
        if not all(map(len,self.pools)):
            raise ValueError("A registered modality family is missing")
        self.identity=hashlib.sha256(json.dumps(cards,sort_keys=True).encode()).hexdigest()
        if stats is None:
            if role!="train":
                raise ValueError("Use the training split numeric normalization")
            d=base["n"][self.pools[4]].astype(np.float32)
            mask=base["ok"][self.pools[4]].astype(bool)
            self.stats=np.stack((d[mask].mean(0),np.maximum(d[mask].std(0),.025)))
        else:
            self.stats=np.array(stats,dtype=np.float32)
        self.cfg=json.loads(CONFIG.read_bytes())
        self.epochs={}

    def requests(self,step,batch):
        counts=np.array(self.cfg["mixture_32"])*batch//32
        if counts.sum()!=batch:
            raise ValueError("Use a batch size divisible by 32")
        rng=np.random.default_rng([self.cfg["seed"],step,1901])
        families=np.repeat(np.arange(len(counts)),counts)
        rows=[]
        for f,count in enumerate(counts):
            idx=step*count+np.arange(count)
            pool=self.pools[f]
            selected=np.empty(count,dtype=np.int64)
            for epoch in np.unique(idx//len(pool)):
                key=(f,int(epoch))
                if key not in self.epochs:
                    self.epochs={k:v for k,v in self.epochs.items() if k[0]!=f}
                    self.epochs[key]=np.random.default_rng([self.cfg["seed"],f,epoch]).permutation(pool)
                use=idx//len(pool)==epoch
                selected[use]=self.epochs[key][idx[use]%len(pool)]
            rows.extend(selected)
        perm=rng.permutation(batch)
        return self.gather(families[perm],np.asarray(rows)[perm],rng.integers(3,size=batch),rng)

    def gather(self,family,indices,branch,rng=None):
        size=len(indices)
        def zeros(*shape,dtype=np.uint8):
            return np.zeros((size,*shape),dtype=dtype)
        b={"a":zeros(64,100),"v":zeros(64,256),"n":zeros(64,17,dtype=np.float32),
           "target_n":zeros(64,17,dtype=np.float32),"u":zeros(64,2,dtype=np.float32),
           "ok":zeros(64),"sense":zeros(64),"x":zeros(64,dtype=np.uint16),"input_x":zeros(64,dtype=np.uint16),
           "has":zeros(4),"family":np.asarray(family,dtype=np.uint8),"event":zeros(64),
           "outcome":zeros(16,2,dtype=np.float32),"source":zeros(4)}
        bm=np.flatnonzero(family<7)
        if len(bm):
            rows=indices[bm]
            base=self.packs[0]
            for k in ("a","v","x","ok","has"):
                b[k][bm]=base[k][rows]
            b["sense"][bm]=b["ok"][bm]
            num=np.asarray(base["n"][rows],np.float32)
            normalized=np.clip((num-self.stats[0])/self.stats[1],-8,8)
            b["target_n"][bm]=normalized
            b["n"][bm,:,:10]=normalized[:,:,:10]
            b["n"][bm,1:,10:]=normalized[:,:-1,10:]
            x=b["x"][bm]
            lengths=np.maximum(1,(x>0).sum(1))
            positions=(np.arange(64)[None,:]*(lengths[:,None]-1))//63
            b["input_x"][bm]=np.where(family[bm,None]==5,x,np.take_along_axis(x,positions,axis=1))
        im=np.flatnonzero(family==7)
        if len(im):
            rows=indices[im]
            imp=self.packs[1]
            for key in ("a","v","event"):
                b[key][im]=imp[key][rows]
            b["has"][im,:2]=1
            b["ok"][im]=1
            b["sense"][im]=1
        pm=np.flatnonzero(family==8)
        if len(pm):
            rows=indices[pm]
            sim=self.packs[2]
            b["n"][pm,:16,:6]=sim["n"][rows]
            b["v"][pm,:16]=sim["v"][rows]
            b["u"][pm,16:32]=sim["plan"][rows,branch[pm]]
            b["outcome"][pm]=sim["outcome"][rows,branch[pm]]
            b["has"][pm,1:3]=1
            b["ok"][pm,:32]=1
            b["sense"][pm,:16]=1
        if rng is not None:
            has=b["has"].astype(bool)
            ranks=np.where(has,rng.random(has.shape),-1)
            order=np.argsort(-ranks,axis=1)
            at=np.arange(size)
            b["source"][at,order[:,0]]=1
            add=((has.sum(1)>=3)|(family==8))&(rng.random(size)<.5)
            b["source"][at[add],order[add,1]]=1
        return b


def stack(batches):
    return {k:np.stack([b[k] for b in batches]) for k in batches[0]}
