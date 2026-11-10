"""Test the decoded-state alignment intervention at the same medium-scale budget as G4 fair-all."""
import argparse
from pathlib import Path
import time

from training.g4_eval import evaluate
from training.g4_checkpoint_budget import admit,parameters
from training.g4_stages import monitor
from training.g4_train import Data,archive,parser,run,write


def main():
    q=argparse.ArgumentParser()
    for key in ("data","out","spool"): q.add_argument("--"+key,type=Path,required=True)
    q.add_argument("--deadline",type=float,required=True); a=q.parse_args()
    status=a.out/"tied-status.json"; done=[]
    data=Data(a.data,"train"); dev=Data(a.data,"dev"); stats=data.stats()
    for seed in (81,82):
        for profile in (("signal-tied","gpt2-tied") if seed==81 else ("gpt2-tied","signal-tied")):
            tag=f"g4-v5e-tied-medium-{profile}-{seed}"; dest=a.out/tag
            if not (dest/"complete.json").exists() and not admit(a.out,a.spool,a.deadline,80,parameters(512,8),status):
                write(status,{"stage":"budget_boundary","completed":done}); return
            if not (dest/"complete.json").is_file():
                args=parser().parse_args(["--data",str(a.data),"--output",str(a.out),"--spool",str(a.spool),"--backend","tpu",
                    "--label","v5e-tied","--scale","medium","--profile",profile,"--seed",str(seed),"--steps","256",
                    "--fixed-lr",".0003","--schedule-steps","256","--save-every","256","--batch","256","--micro","128"])
                with monitor(status,dest/"progress.json",tag+"/train",a.deadline): run(args,data,stats)
            with monitor(status,None,tag+"/dev",a.deadline): evaluate(dest,a.data,"dev",128,"tpu",data_cache=dev)
            archive(tag+"-evidence",256,{k:dest/k for k in ("complete.json","dev.json","dev-paired.npz")},a.spool)
            done.append(tag)
    write(status,{"stage":"complete","completed":done})


if __name__=="__main__": main()
