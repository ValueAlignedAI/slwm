"""Check causal timing, hidden-modality isolation, memory continuity, and initial physical predictions."""
import argparse
import json
from pathlib import Path

import jax
import numpy as np

from models.g4_shared import Spec
from models.g5_shared import encode, parameters
from models.g5_residual import initialize, world
from training.g4_train import write
from training.g5_batches import JointData, load_pack
from training.g5_eval import ridge


def check(root,out):
    train=JointData(root,"train",verify=False)
    dev=JointData(root,"dev",stats=train.stats,verify=False)
    cal={k:np.asarray(v) for k,v in np.load(root/"prior.npz").items()}
    sim=dev.packs[2]
    train_sim,_=load_pack(root/"pusht/train",False)
    expected=ridge(train_sim,sim)
    ids=np.array([0,1,2,3])
    branches=ids%3
    rows=dev.gather(np.full(4,8),ids,branches)
    rows["source"][:,1:3]=1
    results={}
    for core in ("hybrid","gpt2"):
        spec=Spec(core,128,4,"float32")
        p=initialize(spec,62)
        initial=world(p,spec,jax.device_put(rows),rows["source"],jax.device_put(cal))[2]
        prior=float(np.max(np.abs(np.asarray(initial)[:,-1,:]*64-expected[ids,branches])))
        p=parameters(spec,62)
        base=world(p,spec,jax.device_put(rows),rows["source"],jax.device_put(cal))
        changed={k:v.copy() for k,v in rows.items()}
        changed["u"][:,27:32]+=1.5
        changed["outcome"]+=100
        changed["target_n"]+=200
        altered=world(p,spec,jax.device_put(changed),rows["source"],jax.device_put(cal))
        future=float(np.max(np.abs(np.asarray(altered[2])[:,:11]-np.asarray(base[2])[:,:11])))
        vision=np.zeros_like(rows["source"]); vision[:,1]=1
        a=world(p,spec,jax.device_put(rows),vision,jax.device_put(cal))
        changed={k:v.copy() for k,v in rows.items()}
        changed["n"]+=10; changed["a"][:]=222; changed["input_x"][:]=78; changed["event"][:]=1
        b=world(p,spec,jax.device_put(changed),vision,jax.device_put(cal))
        hidden=max(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) for x,y in zip(a,b))
        impact=dev.gather(np.full(4,7),ids,branches)
        av=impact["has"].copy()
        whole,_=encode(p,spec,impact,av)
        def chunk(left,right):
            return {k:(v[:,left:right] if v.ndim>=2 and v.shape[1]==64 else v) for k,v in impact.items()}
        z,m=encode(p,spec,chunk(0,32),av)
        zz,_=encode(p,spec,chunk(32,64),av,memory=m,offset=32)
        continuous=float(np.abs(np.concatenate((np.asarray(z),np.asarray(zz)),1)-np.asarray(whole)).max())
        if prior>.001 or future>1e-5 or hidden>1e-5 or continuous>.001:
            raise AssertionError(dict(core=core,prior=prior,future=future,hidden=hidden,continuous=continuous))
        results[core]={"prior_equivalence_px":prior,"future_leak_max":future,"withheld_input_leak_max":hidden,
                       "chunk_vs_continuous_max":continuous}
    write(out,results)
    print(json.dumps(results))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",required=True,type=Path); p.add_argument("--out",required=True,type=Path)
    a=p.parse_args(); check(a.data,a.out)
