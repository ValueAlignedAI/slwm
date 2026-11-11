"""Score immutable G11 parents and matched instruction interventions on the same decision rows."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from models.g11_instruction import predict
from scripts.train.g9_fit import numpy_weights
from scripts.train.g11_fit import parent,read_bank

MODES=("correct","wrong","none","acoustic","asr_text","human","wrong_human","small","wrong_small")


@lru_cache(maxsize=80)
def kernel(s,mode):
    @jax.jit
    def run(base,d,r):
        y=predict(base,d,s,r,mode)
        return y
    return run


def infer(base,delta,s,data,sem,nv,nn,mode,ix):
    parts={};norm=sem/np.maximum(1e-7,np.linalg.norm(sem,axis=1,keepdims=True))
    keys=("a","v","n","ac","wc","aw","ww","ah","wh","small_words","wrong_small","geometry")
    for i in range(0,len(ix),128):
        subset=ix[i:i+128];row={k:jnp.asarray(data[k][subset]) for k in keys}
        out=kernel(s,mode)(base,delta,row)
        verb=np.asarray(out["verb"])[:,:nv];noun=np.asarray(out["noun"])[:,:nn]
        def collect(k,v):parts.setdefault(k,[]).append(np.asarray(v))
        collect("verb",verb.argmax(1).astype("i2"));collect("noun",noun.argmax(1).astype("i2"))
        collect("open_noun",(np.asarray(out["semantic"])@norm.T).argmax(1).astype("i2"))
        logits=verb-verb.max(1,keepdims=True);lp=logits-np.log(np.exp(logits).sum(1,keepdims=True));truth=data["verb"][subset]
        collect("verb_nll",np.where(truth>=0,-lp[np.arange(len(subset)),np.maximum(0,truth)],-np.log(1e-9)).astype("f4"))
        collect("verb_top3",np.any(np.argsort(verb,axis=1)[:,-3:]==truth[:,None],axis=1))
        collect("effect",np.asarray(out["effect"],dtype=np.float32))
    return {k:np.concatenate(v) for k,v in parts.items()}


def evaluate(banks,parents,runs,dest,stage,fold):
    vocab=json.loads((banks/(fold+"-vocabulary.json")).read_text());nv=len(vocab["verb"]);nn=len(vocab["noun"])
    full=stage==1024
    # Midpoint scores are diagnostics on development. The registered held cohorts use terminal weights.
    roles=("held","familiar","dev") if full else ("dev",)
    db={role:read_bank(banks,fold,role,physical=full) for role in roles}
    dest.mkdir(parents=True,exist_ok=True)
    for seed in (1001,1002,1003):
        for c in ("hybrid","gpt2","gated"):
            pmeta,p,s=parent(parents,c,seed)
            for treatment in ("paired","deranged","absent"):
                tag=f"g11-{fold}-{c}-{treatment}-s{seed}";job=runs/tag/f"step-{stage:05d}";ready=dest/(tag+f"-{stage}.json")
                if ready.exists() or not (job/"verified.json").exists():continue
                d=json.loads((job/"receipt.json").read_text())
                if d["parent_sha256"]!=pmeta["inference_sha256"]:raise ValueError("Source parent changed")
                with (job/"weights.npz").open("rb") as f:
                    if hashlib.file_digest(f,"sha256").hexdigest()!=d["inference_sha256"]:raise ValueError("Inference weights changed")
                theta=jax.tree.map(jnp.asarray,numpy_weights(job/"weights.npz"));out={};sizes={};start=time.monotonic()
                for role,(x,sem) in db.items():
                    ids=np.flatnonzero(x["cued"]);sizes[role]=len(ids)
                    for mode in (MODES if full and treatment=="paired" else ("correct","wrong","none")):
                        for k,val in infer(p,theta,s,x,sem,nv,nn,mode,ids).items():out[role+"|"+mode+"|"+k]=val
                output=dest/(tag+f"-{stage}.npz");temp=output.with_suffix(".part.npz");np.savez_compressed(temp,**out);temp.replace(output)
                ready.write_text(json.dumps(dict(tag=tag,step=stage,parent_sha256=d["parent_sha256"],delta_sha256=d["inference_sha256"],
                     output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),cued=sizes,backend=jax.default_backend(),seconds=time.monotonic()-start))+"\n")
                print(tag,stage,sizes,flush=True)


if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("banks",type=Path);a.add_argument("parents",type=Path);a.add_argument("runs",type=Path);a.add_argument("dest",type=Path)
    a.add_argument("--stage",type=int,choices=(256,1024),required=True);a.add_argument("--folds",nargs="+",default=("A","B"));a.add_argument("--cache");p=a.parse_args()
    if p.cache:jax.config.update("jax_compilation_cache_dir",p.cache)
    for f in p.folds:evaluate(p.banks,p.parents,p.runs,p.dest,p.stage,f)
