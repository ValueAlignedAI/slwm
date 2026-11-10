"""Run a same-chip architecture pair at increasing registered exposures inside one paid session."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path,data):
    tmp=path.with_suffix(".writing"); tmp.write_text(json.dumps(data,sort_keys=True)+"\n"); tmp.replace(path)


def main(a):
    cfg=json.loads(Path("configs/t6/tpu_scale.json").read_bytes())
    out=a.output; out.mkdir(exist_ok=True,parents=True)
    status=out/"lane.json"
    def state(phase,**values):
        write(status,dict(phase=phase,generation=a.generation,seed=a.seed,utc_unix=time.time(),**values))
    def invoke(command,log):
        with log.open("ab") as fd:
            return subprocess.run(command,stdout=fd,stderr=subprocess.STDOUT).returncode
    def base(scale,core):
        return [sys.executable,"-u","-m","training.g6_train","--generation",a.generation,"--scale",scale,
                "--core",core,"--seed",str(a.seed),"--output",str(out),"--data",str(a.data),"--checked"]
    def optimize(scale,core):
        successful=[]
        for micro in cfg["micro_candidates"][a.generation]:
            prefix=f"bench-{a.generation}-{scale}-{core}-{micro}"
            result=out/(prefix+".json"); log=out/(prefix+".log")
            if not result.exists() and not (out/(prefix+".oom")).is_file():
                state("benchmark",scale=scale,core=core,micro=micro)
                rc=invoke(base(scale,core)+["--probe","--micro",str(micro),"--until","12"],log)
                if rc:
                    tail=log.read_bytes()[-12000:].lower()
                    if any(w in tail for w in (b"out of memory",b"resource_exhausted",b"ran out of memory")):
                        (out/(prefix+".oom")).touch()
                        continue
                    raise RuntimeError(f"Benchmark failed, see {log.name}")
            if result.is_file():
                successful.append((json.loads(result.read_bytes())["warm_examples_per_second"],micro))
        if not successful:
            raise RuntimeError(f"No verified microbatch for {scale}/{core}")
        return max(successful)
    order=cfg["cores"] if a.seed==511 else cfg["cores"][::-1]
    try:
        for scale,step in cfg["stages"]:
            for core in order:
                tag=f"g6-{a.generation}-{scale}-{core}-{a.seed}"
                folder=out/tag
                stage=folder/f"step-{step:06d}"
                if (stage/"dev-metrics.json").is_file():
                    continue
                if time.time()+a.reserve_seconds>=a.deadline:
                    state("draining",reason="budgeted archive reserve"); return
                if not (stage/"complete.json").is_file():
                    speed,micro=optimize(scale,core)
                    previous=max((int(p.parent.name[5:]) for p in folder.glob("step-*/complete.json")),default=0)
                    eta=max(0,step-previous)*cfg["batch"]/speed
                    if time.time()+eta*1.15+a.reserve_seconds>=a.deadline:
                        state("draining",reason="remaining stage exceeds time reserve",step=step,estimate_s=eta); return
                    state("training",core=core,scale=scale,target=step,initial_step=previous,micro=micro,estimated_s=eta)
                    log=out/(tag+".log")
                    if invoke(base(scale,core)+["--micro",str(micro),"--until",str(step)],log):
                        raise RuntimeError(f"Training failed, see {log.name}")
                state("evaluating",core=core,scale=scale,step=step)
                if invoke([sys.executable,"-u","-m","training.g5_eval","--backend","tpu","--role","dev",
                           "--data",str(a.data),"--run",str(stage),"--out",str(stage)],out/(tag+"-eval.log")):
                    raise RuntimeError(f"Evaluation failed for {tag}/{step}")
        state("draining",reason="registered stages finished")
    except Exception as exc:
        state("failed",reason=str(exc)); raise


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--generation",choices=["v5e","v6e"],required=True); p.add_argument("--seed",type=int,required=True)
    p.add_argument("--deadline",type=int,required=True); p.add_argument("--reserve-seconds",type=int,default=1200)
    main(p.parse_args())
