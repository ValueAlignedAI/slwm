"""Interleave registered replay/fresh continuations for two sizes and both cores on one native chip."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(p,v):
    q=p.with_suffix(".writing");q.write_text(json.dumps(v,sort_keys=True)+"\n");q.replace(p)


def main(a):
    cfg=json.loads(Path("configs/t7/study.json").read_bytes());a.out.mkdir(exist_ok=True,parents=True)
    orders=("hybrid","gpt2") if a.seed==511 else ("gpt2","hybrid")
    arms=("replay","fresh") if a.seed==511 else ("fresh","replay")
    combos=[(n,size,core,arm) for n in cfg["rungs"] for size in ("anchor","larger") for core in orders for arm in arms]
    def status(phase,**extra):write(a.out/"lane.json",dict(phase=phase,seed=a.seed,generation=a.generation,utc_unix=time.time(),**extra))
    def call(cmd,log):
        with log.open("ab") as fd:subprocess.run(cmd,stdout=fd,stderr=subprocess.STDOUT,check=True)
    try:
        while time.time()<a.deadline-1200:
            finished=0; job=None
            for n,size,core,arm in combos:
                name=f"g7-{a.generation}-{size}-{core}-{a.seed}-{arm}";folder=a.out/name;stage=folder/f"step-{n:06d}"
                if (stage/"dev-metrics.json").is_file():finished+=1;continue
                if arm=="fresh" and not (a.fresh/".verified").is_file():continue
                parent=a.parents/f"g6-v5e-{size}-{core}-{a.seed}"/"step-008192"
                if not (parent/".received").is_file():continue
                previous={10240:8192,12288:10240,16384:12288}[n]
                if previous!=8192 and not (folder/f"step-{previous:06d}/dev-metrics.json").is_file():continue
                job=(n,size,core,arm,name,folder,stage);break
            if finished==len(combos):status("draining",reason="all matched stages finished");return
            if job is None:
                status("waiting_inputs",completed=finished);time.sleep(15);continue
            n,size,core,arm,name,folder,stage=job
            log=a.out/(name+".log");folder.mkdir(exist_ok=True,parents=True)
            if not (stage/"complete.json").is_file():
                rate=2700. if size=="anchor" else 1650.
                if time.time()+1200+1.3*(n-({10240:8192,12288:10240,16384:12288}[n]))*cfg["effective_batch"]/rate>=a.deadline:break
                status("training",run=name,target=n,completed=finished)
                before=time.time()
                call([sys.executable,"-u","-m","training.g7_continue","--data",str(a.data),"--parents",str(a.parents),
                      "--out",str(a.out),"--fresh",str(a.fresh),"--generation",a.generation,"--core",core,"--scale",size,
                      "--seed",str(a.seed),"--arm",arm,"--until",str(n),"--micro","64"],log)
                wall=time.time()-before
            else:
                wall=max(0.,(stage/"complete.json").stat().st_mtime-(folder/"config.json").stat().st_mtime) if n==10240 else None
            status("evaluating",run=name,target=n,completed=finished)
            before=time.time()
            call([sys.executable,"-u","-m","training.g5_eval","--run",str(stage),"--data",str(a.data),"--role","dev","--backend","tpu","--out",str(stage)],log)
            write(stage/"wall.json",{"training_invocation_s":wall,"evaluation_invocation_s":time.time()-before,"run":name,"step":n})
        status("draining",reason="archive wall reserve")
    except BaseException as e:
        status("failed",reason=str(e)[:300]);raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--seed",type=int,required=True);p.add_argument("--generation",required=True)
    for k,v in (("out","/content/g7-out"),("data","/content/g7-data"),("fresh","/content/g7-fresh"),("parents","/content/g5-landing")):
        p.add_argument("--"+k,type=Path,default=Path(v))
    p.add_argument("--deadline",type=float,required=True);main(p.parse_args())
