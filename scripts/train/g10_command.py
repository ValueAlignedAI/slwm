"""Test transient commands, then apply the preregistered matched correction if the gate fires."""
import argparse
import json
from dataclasses import asdict
from functools import lru_cache
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from evals.g10_control import load_stage,mission
from models.g9_shared import encode
from models.g10_context import continuous,brief_visibility,loss_from
from scripts.train.g9_fit import save_json
from scripts.train.g10_fit import adam,bank,restore,snapshot
from scripts.train.g10_store import mirror


@lru_cache(maxsize=8)
def kernel(spec,mode):
    def run(p,m,v,real,starts,step,seed):
        rows=continuous(real,starts,step,seed+76013)
        which=jax.random.bernoulli(jax.random.fold_in(jax.random.PRNGKey(seed+8761),step),.5,(len(rows["u"]),))&rows["is_sim"]
        visible=~which[:,None]|brief_visibility(rows["w"][:,:,2])
        corrected=rows if mode=="continuous" else {**rows,"w":jnp.where(visible[...,None],rows["w"],0)}
        def loss(params):
            z,_=encode(params,spec,corrected,rows["keep"],window=512)
            return loss_from(z,params,rows)
        (value,metrics),g=jax.value_and_grad(loss,has_aux=True)(p)
        p,m,v,norm=adam(p,m,v,g,step,768,5e-5)
        return p,m,v,jnp.r_[value,metrics,norm]
    return jax.jit(run,donate_argnums=(0,1,2))


def score(p,s,seed):
    return {k:mission(p,s,"full",0,seed,instruction=k) for k in ("continuous","brief","blank")}


def run(root,star,dest):
    dest.mkdir(exist_ok=True,parents=True);parents={}
    for seed in (1001,1002,1003):
        for c in ("hybrid","gpt2","gated"):
            tag=f"context-{c}-variable-s{seed}";stage=root/tag/"step-05120";f=dest/(tag+"-parent.json")
            if not f.exists():
                d,p,s=load_stage(stage);v=score(p,s,159071)
                save_json(f,dict(tag=tag,sha256=d["inference_sha256"],scores=v))
            parents[(c,seed)]=json.loads(f.read_text())
    gate={c:[float(np.mean(parents[c,s]["scores"]["continuous"]["occupancy"])-np.mean(parents[c,s]["scores"]["brief"]["occupancy"])) for s in (1001,1002,1003)] for c in ("hybrid","gpt2")}
    eligible=any(sum(v>=.10 for v in vals)>=2 for vals in gate.values());save_json(dest/"gate.json",dict(passed=eligible,losses=gate))
    if not eligible:return
    r,starts,names=bank(star)
    for seed in (1001,1002,1003):
        for core in ("hybrid","gpt2"):
            parent_tag=f"context-{core}-variable-s{seed}"
            parent,source,s=load_stage(root/parent_tag/"step-05120")
            # The sensor-only comparator uses the last actually visible command.
            reference=dest/"physical-reference.json"
            if not reference.exists():save_json(reference,{mode:mission(None,s,"raw",0,247667,instruction=mode) for mode in ("continuous","brief","blank")})
            for mode in ("continuous","brief"):
                tag=f"cue-{core}-{mode}-s{seed}";job=root/tag
                recipe=dict(tag=tag,spec=asdict(s),seed=seed,goal_steps=768,phase="cue",parent_sha256=parent["inference_sha256"],training_recordings=names,backend=jax.default_backend())
                if not (job/"done.json").exists():
                    step,p,m,v=restore(job,lambda:jax.tree.map(lambda x:jnp.array(x,copy=True),source),recipe);anchor=step;t=time.monotonic()
                    full={**recipe,"trainable_parameters":sum(x.size for x in p.values())}
                    while step<768:
                        p,m,v,metric=kernel(s,mode)(p,m,v,r,starts,jnp.int32(step),jnp.int32(seed));step+=1
                        if step==1 or step%64==0:
                            vals=np.asarray(metric)
                            if not np.isfinite(vals).all():raise FloatingPointError(tag)
                            with (job/"train.jsonl").open("a") as log:log.write(json.dumps(dict(step=step,loss=float(vals[0]),seconds=time.monotonic()-t))+"\n")
                        if step in (384,768):snapshot(job,step,p,m,v,full,t,anchor)
                mirror(tag,768);f=dest/(tag+".json")
                if not f.exists():
                    d,p,s=load_stage(job/"step-00768");save_json(f,dict(tag=tag,sha256=d["inference_sha256"],scores=score(p,s,247667)))
                    print(tag,round(np.mean(json.loads(f.read_text())["scores"]["brief"]["occupancy"]),4),flush=True)
    save_json(dest/"complete.json",dict(fits=12,mission_seed=247667))


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("runs",type=Path);ap.add_argument("real",type=Path);ap.add_argument("dest",type=Path);ap.add_argument("--cache")
    a=ap.parse_args()
    if a.cache:jax.config.update("jax_compilation_cache_dir",a.cache)
    run(a.runs,a.real,a.dest)
