"""Run the G9-lineage gated controls and downstream capacity checks on an actual assigned accelerator."""
import argparse
import json
from pathlib import Path
import time
import jax
from scripts.train.g9_fit import bank_data,one
from scripts.train.g11_fit import run,read_bank,parent,kernel
import jax.numpy as jnp
import numpy as np
from models.g11_instruction import initialize
from scripts.train.g9_fit import save_json

def benchmark(banks,parents,out,family,seed):
    d,sem=read_bank(banks,"A","train",False);meta,base,s=parent(parents,"gpt2",seed,family)
    names=("a","v","n","ac","wc","aw","ww","ah","wh","small_words","wrong_small","geometry","effect","valid","verb","noun","open_noun")
    data=jax.tree.map(jax.device_put,{key:d[key] for key in names});ci=jnp.array(np.flatnonzero(d["cued"]),dtype=jnp.int32);bg=jnp.array(np.flatnonzero(~d["cued"]),dtype=jnp.int32)
    labels=json.loads((banks/"A-vocabulary.json").read_text());fn=kernel(s,"correct",len(labels["verb"]),len(labels["noun"]),False)
    theta=initialize(s,seed);m=jax.tree.map(jnp.zeros_like,theta);v=jax.tree.map(jnp.zeros_like,theta)
    for i in range(8):
        theta,m,v,z=fn(theta,m,v,base,data,ci,bg,ci,jnp.asarray(sem),jnp.ones(3),jnp.int32(i),jnp.int32(seed))
    z.block_until_ready();start=time.monotonic()
    for i in range(8,72):theta,m,v,z=fn(theta,m,v,base,data,ci,bg,ci,jnp.asarray(sem),jnp.ones(3),jnp.int32(i),jnp.int32(seed))
    z.block_until_ready();elapsed=time.monotonic()-start
    save_json(out/f"throughput-{family}.json",dict(core="gpt2",family=family,seed=seed,updates=64,batch=128,history=32,seconds=elapsed,updates_per_second=64/elapsed,positions_per_second=64*128*32/elapsed,backend=jax.default_backend(),device=jax.devices()[0].device_kind,final_loss=float(z[0])))


def main(mode,data,parents,out,seeds):
    out.mkdir(exist_ok=True,parents=True);parents.mkdir(exist_ok=True,parents=True)
    if mode=="parents":
        real,starts,names=bank_data(data/"starss-train.npz")
        for seed in seeds:
            for size in ("exposure","confirm"):one((size,"gated",128,"gelu",seed,False),real,starts,names,parents)
    else:
        banks=data/"banks"
        for seed in seeds:
            for family in ("small","large"):
                if not (out/f"throughput-{family}.json").exists():benchmark(banks,parents,out,family,seed)
                for phase in ("action","joint"):
                    for fold in ("A","B"):run(banks,parents,out,fold,[seed],["hybrid","gpt2","gated"],["paired"],phase,family)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("parents","transfer"));ap.add_argument("data",type=Path);ap.add_argument("parents",type=Path);ap.add_argument("out",type=Path)
    ap.add_argument("--seeds",nargs="+",type=int,default=[911,912]);ap.add_argument("--cache");p=ap.parse_args()
    if p.cache:jax.config.update("jax_compilation_cache_dir",p.cache)
    main(p.mode,p.data,p.parents,p.out,p.seeds)
