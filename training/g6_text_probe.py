"""Held byte-level LM and same-last-byte context control inside the actual joint terminal models."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from models.g4_shared import Spec
from models.g5_residual import world
from training.g4_train import write,sha_file
from training.g5_batches import JointData


def raw_log_probs(pack):
    a=pack.packs[0]; counts=np.zeros((259,259),np.float64)
    for ix in np.array_split(pack.pools[5],64):
        x=np.asarray(a["x"][ix],np.int32); left=x[:,:-1]; right=x[:,1:]
        ok=(left>0)&(right>0)
        counts+=np.bincount(left[ok]*259+right[ok],minlength=259**2).reshape(259,259)
    gram=counts+.1; mono=counts.sum(0)+.1
    return -np.log2(gram/gram.sum(1,keepdims=True)),-np.log2(mono/mono.sum())


def score(a):
    cfg=json.loads((a.run/"config.json").read_bytes()); rec=json.loads((a.run/"verified.json").read_bytes())
    if rec["step"]!=8192 or sha_file(a.run/"inference.npz")!=rec["sha256"]["inference.npz"]:
        raise ValueError("Expect a verified terminal model")
    with np.load(a.run/"inference.npz") as f:
        p={k:jax.device_put(f[k]) for k in f if k!="stats" and not k.startswith("prior_")}
        cal={k[6:]:jax.device_put(f[k]) for k in f if k.startswith("prior_")}
        stats=np.asarray(f["stats"])
    train=JointData(a.data,"train",verify=False,stats=stats); dev=JointData(a.data,"dev",verify=False,stats=stats)
    gram,mono=raw_log_probs(train); spec=Spec(**cfg["spec"]); rows=dev.pools[5]; src=dev.packs[0]
    totals=dict(full=0.,learned_bigram=0.,raw_bigram=0.,raw_unigram=0.); nbytes=0; correct=[]; corrupted=[]
    @jax.jit
    def losses(b):
        pred,_,_=world(p,spec,b,b["source"],cal)
        ids=b["x"].astype(jnp.int32)
        bias=p["bigram"][b["input_x"].astype(jnp.int32)]
        def nll(logits):
            return -jnp.take_along_axis(jax.nn.log_softmax(logits[:,:-1],axis=-1),ids[:,1:,None],axis=-1)[...,0]/jnp.log(2.)
        return nll(pred[:,:,373:632]+bias),nll(bias)
    for start in range(0,len(rows),32):
        ix=rows[start:start+32]; n=len(ix); pad=np.pad(ix,(0,32-n),mode="edge")
        b=dev.gather(np.full(32,5),pad,np.zeros(32,int)); b["source"][:,3]=1
        b["a"].fill(0); b["v"].fill(0); b["n"].fill(0); b["u"].fill(0)
        x=b["x"][:n]; left=x[:,:-1]; target=x[:,1:]
        valid=(target>=3)&(left>0)&b["ok"][:n,:-1].astype(bool)
        full,bias=map(np.asarray,losses(b))
        for key,value in (("full",full[:n]),("learned_bigram",bias[:n]),("raw_bigram",gram[left,target]),("raw_unigram",mono[target])):
            totals[key]+=float(value[valid].sum(dtype=np.float64))
        nbytes+=int(valid.sum())
        altered=b.copy(); altered["input_x"]=b["input_x"].copy(); positions=[]
        for j,row in enumerate(ix):
            choices=np.flatnonzero(valid[j]&(np.arange(63)>=8))
            if not len(choices): continue
            k=int(src["uid"][row]); rng=np.random.default_rng([k,61110])
            t=int(rng.choice(choices))
            altered["input_x"][j,:t]=rng.permutation(altered["input_x"][j,:t])
            positions.append((j,t))
        loss2,bias2=map(np.asarray,losses(altered))
        for j,t in positions:
            if abs(float(bias2[j,t])-float(bias[j,t]))>1e-6: raise ValueError("The immediate-byte control changed")
            correct.append(full[j,t]); corrupted.append(loss2[j,t])
    delta=np.array(corrupted,dtype=np.float64)-np.array(correct,dtype=np.float64)
    draw=np.random.default_rng(61111).integers(0,len(delta),(2000,len(delta)))
    bpb={k:v/nbytes for k,v in totals.items()}
    write(a.out,{"run":cfg["run"],"params":cfg["params"],"seed":cfg["seed"],"step":rec["step"],"role":"FineWeb development",
                 "inference_sha256":rec["sha256"]["inference.npz"],"windows":len(rows),"scored_real_next_bytes":nbytes,
                 "bpb":bpb,"perplexity_per_byte_full":2**bpb["full"],"context_probe":{"eligible_windows":len(delta),
                 "correct_bpb":float(np.mean(correct)),"scrambled_prefix_bpb":float(np.mean(corrupted)),
                 "scrambled_minus_correct_bpb":float(delta.mean()),"paired_window_ci95":np.quantile(delta[draw].mean(1),[.025,.975]).tolist()}})
    print(cfg["run"],"held bits per byte",bpb,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--run",type=Path,required=True); p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True); score(p.parse_args())
