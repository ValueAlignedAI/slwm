"""Score recorded action forecasts against persistence with the identical information cutoff."""
import argparse
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from models.g4_shared import encode,unpack
from training.g4_eval import weights
from training.g4_train import Data,read,write


def score(run,data,role,batch_size):
    cfg,complete,spec,p,stats=weights(run)
    @jax.jit
    def predict(p,stats,b):
        z,_=encode(p,spec,b,jnp.full((len(b["ok"]),),2,jnp.int32),stats)
        pred=(z@p["readout"].astype(z.dtype)).astype(jnp.float32)[:,:,394:401]
        n=unpack(b,stats)[2][:,:,10:]
        # At decision tick t the model has state[t] and actions through t-1.
        # All candidates predict action[t+1] on the same valid triples, using t>=2.
        y=n[:,3:]; prev=n[:,1:-2]; older=n[:,:-3]
        candidates=(pred[:,2:-1],prev,2*prev-older,n[:,2:-1])
        valid=b["ok"][:,3:]*b["ok"][:,2:-1]*b["ok"][:,1:-2]*b["ok"][:,:-3]
        err=jnp.stack([((v-y)**2).mean(-1) for v in candidates],-1)
        return (err*valid[...,None]).sum(1)/jnp.maximum(valid.sum(1,keepdims=True),1)
    rows=data.pools[4]; results=[]
    for i in range(0,len(rows),batch_size):
        idx=rows[i:i+batch_size]; n=len(idx); idx=np.pad(idx,(0,batch_size-n),mode="edge")
        b={k:v[idx] for k,v in data.device.items()}
        results.append(np.array(predict(p,stats,b))[:n])
    vals=np.concatenate(results)
    record={"run":cfg["name"],"role":role,"weight_sha256":complete["weight_file_sha256"],"n":len(rows),
            "groups":len(np.unique(data.group[rows])),"normalized_action_mse":dict(zip(
            ("model","matched_last_available_t_minus_1","matched_constant_velocity","privileged_action_t"),map(float,vals.mean(0))))}
    write(run/"action-history-audit.json",record)
    print(cfg["name"],record["normalized_action_mse"],flush=True)


if __name__=="__main__":
    q=argparse.ArgumentParser(); q.add_argument("--out",type=Path,required=True); q.add_argument("--data",type=Path,required=True)
    q.add_argument("--lock",type=Path,required=True); q.add_argument("--batch",type=int,default=64)
    a=q.parse_args(); data=Data(a.data,"test"); lock=read(a.lock)["endpoints"]
    for tag,sha in sorted(lock.items()):
        if not any(x in tag for x in ("gpt2-cross","signal-cross","sparse-cross")): continue
        if read(a.out/tag/"complete.json")["weight_file_sha256"]!=sha: raise ValueError("Action audit endpoint mismatch")
        score(a.out/tag,data,"test",a.batch)
