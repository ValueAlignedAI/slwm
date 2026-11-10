"""Time complete, synchronized optimizer updates at equal effective exposure."""
import argparse
import gc
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from models.g4_shared import initialize
from training.g4_train import CFG, Data, Sampler, backend, config_spec, make_block, read, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    p.add_argument("--backend",choices=["cpu","gpu","tpu"],required=True)
    p.add_argument("--scale",default="pilot"); p.add_argument("--profile",default="signal-cross")
    p.add_argument("--micros",nargs="+",type=int,default=[32,64,128,256])
    p.add_argument("--batch",type=int,default=256); p.add_argument("--scan",type=int,default=4)
    p.add_argument("--reps",type=int,default=4); p.add_argument("--limit",type=int,default=0)
    args=p.parse_args()
    dev=backend(args.backend); cfg=read(CFG)
    started=time.monotonic()
    data=Data(args.data,"train",args.limit)
    stats=jax.device_put(data.stats()); sampler=Sampler(data,args.batch,cfg["seed"],cfg["mixture_16"])
    load_s=time.monotonic()-started
    spec,policy=config_spec(cfg,args.scale,args.profile,"bfloat16")
    record={"device":dev.device_kind,"platform":dev.platform,"jax":jax.__version__,"scale":args.scale,"profile":args.profile,
            "effective_batch":args.batch,"scan":args.scan,"valid_fraction":float(data.host["ok"].mean()),
            "examples_in_memory":len(data.host["family"]),"verify_load_device_s":load_s,"rows":[]}
    for micro in args.micros:
        weights=initialize(spec,81)
        opt,fn=make_block(spec,policy,micro,args.batch,32,.0004,1024)
        state=(weights,opt.init(weights),jnp.int32(0))
        ids,views=sampler.block(0,args.scan)
        jax.block_until_ready(state)
        begin=time.monotonic()
        try:
            executable=fn.lower(state,data.device,ids,views,stats).compile()
            compile_s=time.monotonic()-begin
            state,metrics=executable(state,data.device,ids,views,stats)
            jax.block_until_ready(metrics)
            durations=[]
            for r in range(args.reps):
                begin=time.monotonic()
                ids,views=sampler.block((r+1)*args.scan,args.scan)
                state,metrics=executable(state,data.device,ids,views,stats)
                jax.block_until_ready(metrics)
                durations.append(time.monotonic()-begin)
            if not np.isfinite(np.asarray(metrics)).all():
                raise ValueError("nonfinite optimizer metrics")
            rate=args.scan*args.batch/float(np.median(durations))
            row={"micro":micro,"compile_s":compile_s,"warm_block_s":durations,"examples_s":rate,
                 "input_positions_s":rate*64,"params":sum(v.size for v in weights.values()),"metrics":np.asarray(metrics).tolist()}
            del executable
        except (RuntimeError, ValueError) as e:
            row={"micro":micro,"error":str(e)[:650]}
        record["rows"].append(row); write(args.out,record); print(json.dumps(row),flush=True)
        del state,opt,fn,weights
        gc.collect(); jax.clear_caches()
    good=[r for r in record["rows"] if "examples_s" in r]
    if not good: raise RuntimeError("No viable registered batch")
    record["selected_micro"]=max(good,key=lambda r:r["examples_s"])["micro"]
    write(args.out,record)


if __name__=="__main__":
    main()
