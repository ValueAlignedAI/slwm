"""Batched conditional generation, within-domain controls and group-level paired uncertainty."""
import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from models.g4_shared import Spec, encode, errors, unpack
from training.g4_data import FAMILIES
from training.g4_train import Data, backend, read, sha_file, write

MODES=("audio","visual","numeric","text")


def weights(run):
    cfg=read(run/"config.json")
    complete=read(run/"complete.json")
    path=run/"resume.npz"
    expected=complete["weight_file_sha256"]
    if not path.is_file() and "inference_sha256" in complete:
        path=run/"model.npz"
        expected=complete["inference_sha256"]
    if sha_file(path)!=expected:
        raise ValueError("Endpoint checksum changed")
    with np.load(path,allow_pickle=False) as f:
        identity=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        if str(f["identity"])!=identity or int(f["step"])!=cfg["steps"]:
            raise ValueError("Checkpoint identity or training budget changed")
        p={key[2:]:jax.device_put(f[key]) for key in f.files if key.startswith("p/")}
        stats=jax.device_put(f["stats"])
    return cfg,complete,Spec(**cfg["spec"]),p,stats


def wrong(rows,data):
    """Use another recording from the same group when available, falling back within source family."""
    group=data.group[rows]
    other=np.empty_like(rows)
    same=np.zeros(len(rows),bool)
    for val in np.unique(group):
        local=np.flatnonzero(group==val)
        pool=local if len(local)>1 else np.arange(len(rows))
        if len(pool)<2: raise ValueError("No independent control row")
        pool=pool[np.argsort(data.uid[rows[pool]])]
        lookup={int(k):i for i,k in enumerate(pool)}
        for k in local:
            start=lookup[int(k)]
            for shift in range(1,min(len(pool),64)):
                alt=pool[(start+shift)%len(pool)]
                if data.host["cap"][rows[k]]!=data.host["cap"][rows[alt]]:
                    break
            else:
                alt=pool[(start+1)%len(pool)]
            other[k]=rows[alt]
            same[k]=data.group[other[k]]==group[k]
    return other,same


def paired_ci(value,groups):
    ids=np.unique(groups)
    means=np.array([value[groups==g].mean() for g in ids])
    rng=np.random.default_rng(9123)
    if len(ids)<2: return [float(means.mean()),None,None]
    boot=np.concatenate([rng.choice(means,(min(100,1000-k),len(means))).mean(1) for k in range(0,1000,100)])
    return [float(means.mean()),*map(float,np.quantile(boot,[.025,.975]))]


@lru_cache(maxsize=12)
def compiled_metrics(spec):
    @jax.jit
    def calculate(p,stats,arrays,ids,wrong_ids,source):
        target=jax.tree.map(lambda x:x[ids],arrays)
        observed=jax.tree.map(lambda x:jnp.concatenate([x[ids],x[wrong_ids],x[ids]],0),arrays)
        expected=jax.tree.map(lambda x:jnp.tile(x,(3,)+(1,)*(x.ndim-1)),target)
        batch=ids.shape[0]
        observed_kind=jnp.full((3*batch,),source,jnp.int32)
        erase=jnp.arange(3*batch)>=2*batch
        z,_=encode(p,spec,observed,observed_kind,stats,erase=erase)
        err,diff,_,pred=errors(p,spec,z,expected,stats)
        _,_,numeric=unpack(expected,stats)
        valid=expected["ok"].astype(jnp.float32)
        act=pred[:,:,366:373]-numeric[:,:,10:17]
        action=((act*act).mean(-1)*valid).sum(1)/jnp.maximum(valid.sum(1),1)
        persistence=((jnp.diff(numeric[:,:,10:],axis=1)**2).mean(-1)*valid[:,1:]).sum(1)/jnp.maximum(valid[:,1:].sum(1),1)
        lag=pred[:,:-1,394:401]-numeric[:,1:,10:]
        future=((lag*lag).mean(-1)*valid[:,1:]).sum(1)/jnp.maximum(valid[:,1:].sum(1),1)
        return jnp.concatenate([err,diff,action[:,None],future[:,None],persistence[:,None]],-1).reshape(3,batch,11)
    return calculate


def check_lock(lock,run,complete):
    if lock is None:
        raise ValueError("Confirmation requires a frozen endpoint set")
    record=read(lock)
    identity=record["endpoints"].get(run.name)
    if identity!=complete["weight_file_sha256"] or record["evaluation_sha256"]!=sha_file(Path(__file__)):
        raise ValueError("Endpoint or evaluation code differs from the confirmation lock")


def evaluate(run,root,role,batch_size,device,lock=None,cap=0,data_cache=None):
    dev=backend(device)
    cfg,complete,spec,p,stats=weights(run)
    if role=="test": check_lock(lock,run,complete)
    data=Data(root,role) if data_cache is None else data_cache
    calc=compiled_metrics(spec)
    results={}; arrays={}
    count=0; elapsed=0; compile_s=None
    for f,name in enumerate(FAMILIES):
        for src,modality in enumerate(MODES):
            rows=data.pools[f]
            rows=rows[data.host["has"][rows,src].astype(bool)]
            if cap: rows=rows[:cap]
            if len(rows)<2: continue
            wrong_ids,same=wrong(rows,data)
            parts=[]
            for begin in range(0,len(rows),batch_size):
                ii=rows[begin:begin+batch_size]; jj=wrong_ids[begin:begin+batch_size]
                n=len(ii); ii=np.pad(ii,(0,batch_size-n),mode="edge"); jj=np.pad(jj,(0,batch_size-n),mode="edge")
                t=time.monotonic()
                value=np.asarray(calc(p,stats,data.device,ii,jj,jnp.int32(src)))[:,:n]
                dt=time.monotonic()-t
                if compile_s is None: compile_s=dt
                else: count+=n; elapsed+=dt
                parts.append(value)
            values=np.concatenate(parts,axis=1)
            for target,label in enumerate(MODES):
                use=data.host["has"][rows,target].astype(bool)
                if not use.any() or (src==target and not (f==5 or target==2)): continue
                key=f"{name}/{modality}-to-{label}"
                groups=data.group[rows[use]]
                vals=values[:,use,target]
                results[key]={"n":int(use.sum()),"groups":len(np.unique(groups)),"correct":float(vals[0].mean()),
                              "wrong":float(vals[1].mean()),"erased":float(vals[2].mean()),
                              "wrong_minus_correct_group_ci":paired_ci(vals[1]-vals[0],groups),
                              "erased_minus_correct_group_ci":paired_ci(vals[2]-vals[0],groups),
                              "negative_same_group_fraction":float(same[use].mean())}
                if target<3:
                    delta=values[:,use,target+4]
                    results[key]["difference_mse"]=delta.mean(1).tolist()
                arrays[key.replace("/","__")]=np.stack([data.uid[rows[use]],groups.astype(np.uint64)],axis=1)
                arrays[key.replace("/","__")+"__errors"]=vals.astype(np.float32)
            if f==4:
                results[f"{name}/{modality}-action"]={"current":values[:,:,8].mean(1).tolist(),
                    "next":values[:,:,9].mean(1).tolist(),"next_persistence":float(values[0,:,10].mean()),
                    "n":len(rows),"groups":len(np.unique(data.group[rows]))}
    suffix=role if not cap else role+"-smoke"
    write(run/f"{suffix}.json",{"run":cfg["name"],"role":role,"data_card_sha256":sha_file(data.dir/"card.json"),
         "weight_sha256":complete["weight_file_sha256"],"compiled_first_batch_s":compile_s,
         "warm_conditioned_examples_s":count/max(elapsed,1e-9),"validation_batch":batch_size,
         "controls_per_example":3,"device":dev.device_kind,"scores":results})
    np.savez(run/f"{suffix}-paired.npz",**arrays)
    print(json.dumps({"run":run.name,"role":suffix,"directions":len(results),"examples_s":count/max(elapsed,1e-9)}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True); p.add_argument("--data",type=Path,required=True)
    p.add_argument("--role",choices=["dev","test"],default="dev"); p.add_argument("--backend",default="tpu")
    p.add_argument("--batch",type=int,default=128); p.add_argument("--lock",type=Path); p.add_argument("--cap",type=int,default=0)
    a=p.parse_args(); evaluate(a.run,a.data,a.role,a.batch,a.backend,a.lock,a.cap)
