"""Batched held-out confirmation after endpoint binding. Keep data on the accelerator across models."""
import argparse
from pathlib import Path
import time

from training.g4_eval import evaluate
from training.g4_train import Data,archive,read,write
from training.g4_transfer import transfer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ("data","out","lock","probes","spool"): p.add_argument("--"+field,type=Path,required=True)
    p.add_argument("--backend",default="cpu"); p.add_argument("--batch",type=int,default=128)
    p.add_argument("--phase",choices=["generation","transfer"],required=True)
    a=p.parse_args(); rows=read(a.lock)["endpoints"]
    data=Data(a.data,"test") if a.phase=="generation" else None
    order=sorted(rows,key=lambda s:("cross" not in s,s.endswith("82"),s))
    stat=a.out/("confirmation-"+a.phase+".json"); done=[]
    for tag in order:
        run=a.out/tag; step=read(run/"complete.json")["step"]
        write(stat,{"stage":a.phase,"active":tag,"completed":done,"time":time.time()})
        if a.phase=="generation":
            if not (run/"test.json").exists(): evaluate(run,a.data,"test",a.batch,a.backend,a.lock,data_cache=data)
            files={n:run/n for n in ("test.json","test-paired.npz")}
        else:
            if not (run/"transfer-test.json").exists(): transfer(run,a.data,a.probes,"test",a.batch,a.backend,a.lock)
            files={n:run/n for n in ("transfer-test.json","transfer-test.npz")}
        archive(tag+"-confirmation-"+a.phase,step,files,a.spool)
        done.append(tag)
    write(stat,{"stage":"complete","completed":done,"time":time.time()})


if __name__=="__main__": main()
