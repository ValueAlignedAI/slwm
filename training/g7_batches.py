"""Paired fresh/replay inputs with the exact G6 family, presence, source and valid-tick masks."""
import hashlib
import json
from pathlib import Path
import numpy as np
from training.g5_batches import JointData,load_pack

FAMILIES={0:"crema",1:"libri",5:"text",7:"events",8:"sim"}


class ContinuationData:
    def __init__(self,base,fresh=None):
        self.base=base;self.extra={};self.pools={};self.orders={}; self.identity=base.identity
        if fresh is None:return
        cards=[]
        for f,name in FAMILIES.items():
            pack,card=load_pack(Path(fresh)/name/"train",False)
            n=len(next(iter(pack.values())))
            ix=np.arange(n)
            if "length" in pack:ix=ix[pack["length"][ix]>=64]
            if f in (0,1):ix=ix[(pack["x"][ix]>2).any(1)]
            if not len(ix):raise ValueError(f"No complete independent examples for {name}")
            self.extra[f]=pack;self.pools[f]=ix;cards.append(card)
        self.sim=object.__new__(JointData)
        self.sim.packs=[None,None,self.extra[8]];self.sim.stats=base.stats
        self.identity=hashlib.sha256((base.identity+json.dumps(cards,sort_keys=True)).encode()).hexdigest()

    def requests(self,step,batch):
        b=self.base.requests(step,batch)
        if not self.extra:return b
        replace=np.random.default_rng([71110,step,113])
        for f,pack in self.extra.items():
            members=np.flatnonzero(b["family"]==f)
            half=len(members)//2
            chosen=replace.permutation(members)[:half]
            required={0:[1,1,0,1],1:[1,0,0,1],5:[0,0,0,1],7:[1,1,0,0],8:[0,1,1,0]}[f]
            valid=32 if f==8 else 64
            ok=(b["has"][chosen]==required).all(1)&(b["ok"][chosen].sum(1)==valid)
            ordinal=(step-8192)*half+np.arange(half)[ok]
            chosen=chosen[ok];pool=self.pools[f]
            ids=np.empty(len(chosen),np.int64)
            for epoch in np.unique(ordinal//len(pool)):
                key=(f,int(epoch))
                if key not in self.orders:
                    self.orders={k:v for k,v in self.orders.items() if k[0]!=f}
                    self.orders[key]=np.random.default_rng([77113,f,int(epoch)]).permutation(pool)
                use=ordinal//len(pool)==epoch;ids[use]=self.orders[key][ordinal[use]%len(pool)]
            if not len(ids):continue
            if f==8:
                sb=self.sim.gather(np.full(len(ids),8),ids,replace.integers(3,size=len(ids)))
                for k in ("n","v","u","outcome","target_n"):b[k][chosen]=sb[k]
            else:
                for k in ("a","v","x","event"):
                    if k in pack:b[k][chosen]=pack[k][ids]
                if "x" in pack:
                    x=b["x"][chosen]
                    if f==5:b["input_x"][chosen]=x
                    else:
                        lengths=np.maximum(1,(x>0).sum(1));at=np.arange(64)[None,:]*(lengths[:,None]-1)//63
                        b["input_x"][chosen]=np.take_along_axis(x,at,axis=1)
        return b
