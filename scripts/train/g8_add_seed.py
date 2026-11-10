"""Attach a second seed to an existing G8 lane when the other concurrent allocation is unavailable."""
import argparse
import fcntl
from pathlib import Path
from g7_inputs import parent,shell

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("name");p.add_argument("seed",type=int,choices=(511,512));p.add_argument("control",type=Path);a=p.parse_args()
    with (a.control/(a.name+".transport")).open("a") as fd:
        fcntl.flock(fd,fcntl.LOCK_EX)
        shell(a.name,"test -s /content/g8-base-ready && test -s /content/g8-fresh/.verified")
        shell(a.name,f"touch /content/g8-out/seed-{a.seed}")
        for core in (("gpt2","hybrid") if a.seed==512 else ("hybrid","gpt2")):
            parent(a.name,a.seed,"anchor",core)
    print("Second seed ready",a.name,a.seed,flush=True)
