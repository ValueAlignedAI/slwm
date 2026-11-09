"""Equal-budget scale, unique-data and temporal-objective comparisons in one accelerator process."""
import argparse
import gc
from pathlib import Path
import time

import jax

from training.g4_eval import evaluate
from training.g4_stages import monitor
from training.g4_train import Data,archive,parser as train_parser,read,run,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("data","out","spool"): p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--backend",default="tpu"); p.add_argument("--deadline",type=float,required=True)
    p.add_argument("--steps",type=int,default=256); p.add_argument("--label",default="v5e-fair")
    a=p.parse_args(); status=a.out/(a.label+"-scaling.json")
    order=[("medium","signal-cross",1.),("medium","signal-motion",1.),("medium","gpt2-cross",1.),("medium","gpt2-motion",1.),
           ("pilot","signal-cross",1.),("pilot","gpt2-cross",1.),("medium","signal-cross",.25),("medium","gpt2-cross",.25),
           ("anchor","signal-cross",1.),("anchor","gpt2-cross",1.)]
    data=Data(a.data,"train"); dev=Data(a.data,"dev"); stats=data.stats()
    done=[]
    for seed in (81,82):
        for scale,profile,fraction in order:
            label=a.label+("-quarter" if fraction<1 else "-all")
            tag=f"g4-{label}-{scale}-{profile}-{seed}"; output=a.out/tag
            micro=32 if scale=="anchor" else 128
            reserve={"pilot":55,"medium":100,"anchor":180}[scale]
            if time.time()+reserve>a.deadline:
                write(status,{"stage":"reserved_time_for_validation_and_backup","completed":done,"next":tag,"at":time.time()})
                return
            if not (output/"complete.json").exists():
                args=train_parser().parse_args(["--data",str(a.data),"--output",str(a.out),"--spool",str(a.spool),
                     "--backend",a.backend,"--label",label,"--scale",scale,"--profile",profile,"--seed",str(seed),
                     "--steps",str(a.steps),"--schedule-steps",str(a.steps),"--save-every",str(a.steps),"--batch","256",
                     "--micro",str(micro),"--scan","8","--negatives","32","--data-fraction",str(fraction),"--fixed-lr",".0003"])
                with monitor(status,output/"progress.json",tag+"/train",a.deadline):
                    run(args,data,stats)
            if not (output/"dev.json").exists():
                with monitor(status,None,tag+"/dev",a.deadline):
                    evaluate(output,a.data,"dev",min(micro,128),a.backend,data_cache=dev)
            archive(tag+"-evidence",a.steps,{s:output/s for s in ("complete.json","dev.json","dev-paired.npz")},a.spool)
            done.append(tag); gc.collect()
        jax.clear_caches()
    write(status,{"stage":"complete","completed":done,"at":time.time()})


if __name__=="__main__": main()
