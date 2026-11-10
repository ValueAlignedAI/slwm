"""Numerically bridge each old corner and train the two missing arms at fixed exposure."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path("/content/g8-out")
G7=Path("experiments/multimodal/t7/continuation-alignment-control-20261110/metrics/checkpoint-inventory.json")
CONFIG=Path("configs/t8/factorial.json")


def write(path,doc):
    tmp=path.with_suffix(".new");tmp.write_text(json.dumps(doc,sort_keys=True)+"\n");tmp.replace(path)


def call(args,log):
    with log.open("ab") as out:subprocess.run([sys.executable,"-u",*args],stdout=out,stderr=subprocess.STDOUT,check=True)


def main(deadline):
    plan=json.loads(CONFIG.read_bytes());old=json.loads(G7.read_bytes())["rungs"]
    ROOT.mkdir(exist_ok=True,parents=True)
    def status(phase,**fields):write(ROOT/"lane.json",dict(phase=phase,utc_unix=time.time(),**fields))
    common=["--data","/content/g8-data","--parents","/content/g5-landing","--fresh","/content/g8-fresh","--generation","v5e","--scale","anchor","--micro","64"]
    try:
        while time.time()<deadline-900:
            seeds=[s for s in plan["seeds"] if (ROOT/f"seed-{s}").exists()]
            pending=False;all_done=True;worked=False
            for seed in seeds:
                for core in (("hybrid","gpt2") if seed==511 else ("gpt2","hybrid")):
                    folder=Path("/content/g5-landing")/f"g6-v5e-anchor-{core}-{seed}"/"step-008192"
                    bridge=ROOT/f"bridge-{core}-{seed}.json";arm=plan["bridge"][str(seed)][core]
                    if not (folder/".received").is_file():
                        pending=True;all_done=False;continue
                    if not bridge.exists():
                        name=f"g7-v5e-anchor-{core}-{seed}-{arm}";ref=old[name+"/step-010240"]
                        status("numerical_bridge",core=core,seed=seed,arm=arm)
                        call(["-m","training.g7_continue",*common,"--core",core,"--seed",str(seed),"--arm",arm,"--until","10240","--out","/content/g8-bridge"],ROOT/(name+".log"))
                        result=json.loads((Path("/content/g8-bridge")/name/"step-010240/complete.json").read_bytes())["sha256"]
                        exact=result["inference.npz"]==ref["inference_sha256"] and result["resume.npz"]==ref["optimizer_sha256"]
                        write(bridge,{"core":core,"seed":seed,"arm":arm,"exact":exact,"observed":result,
                                      "expected_weights":ref["inference_sha256"],"expected_optimizer":ref["optimizer_sha256"]})
                        worked=True;break
                if worked:break
            if worked:continue
            for stage in plan["rungs"]:
                for seed in seeds:
                    for core in (("hybrid","gpt2") if seed==511 else ("gpt2","hybrid")):
                        bridge=ROOT/f"bridge-{core}-{seed}.json"
                        if not bridge.is_file():continue
                        arms=["sim","other"] if seed==511 else ["other","sim"]
                        if not json.loads(bridge.read_bytes())["exact"]:arms+=["replay","fresh"]
                        for arm in arms:
                            name=f"g8-v5e-anchor-{core}-{seed}-{arm}";out=ROOT/name/f"step-{stage:06d}"
                            if (out/"wall.json").exists():continue
                            prev={10240:8192,12288:10240,16384:12288}[stage]
                            if prev>8192 and not (out.parent/f"step-{prev:06d}/wall.json").exists():continue
                            all_done=False;status("training",run=name,step=stage)
                            t=time.time()
                            call(["-m","training.g8_factorial",*common,"--core",core,"--seed",str(seed),"--arm",arm,"--until",str(stage),"--out",str(ROOT)],ROOT/(name+".log"))
                            train_time=time.time()-t;status("development",run=name,step=stage);t=time.time()
                            call(["-m","training.g5_eval","--run",str(out),"--data","/content/g8-data","--role","dev","--backend","tpu","--out",str(out)],ROOT/(name+".log"))
                            write(out/"wall.json",{"training_invocation_s":train_time,"evaluation_invocation_s":time.time()-t,"step":stage,"run":name})
                            worked=True;break
                        if worked:break
                    if worked:break
                if worked:break
            if worked:continue
            if seeds and all_done and not pending:
                status("draining",seeds=seeds);return
            status("waiting_inputs",seeds=seeds);time.sleep(12)
        status("wall_reserve")
    except BaseException as e:
        status("failed",exception=repr(e)[:350]);raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--deadline",required=True,type=float);main(p.parse_args().deadline)
