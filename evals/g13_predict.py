"""Seal model selection and residual strengths on development, then open two independent source cohorts."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from scripts.train.g13_robot_fit import ROOT,SEEDS,Spec,read,prediction,errors
from scripts.train.g13_optim import roots
from scripts.train.g9_storage import sha,write
from training.g13_physics import prepare,group_mean
from training.g13_adaptive import decorate

def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def study(kind):
    return ROOT/"work"/("repaired" if kind=="repaired" else "robot")


def records(kind):
    work=study(kind)
    docs=[json.loads(f.read_bytes()) for f in sorted((work/"dev").glob("*.json"))]
    if kind=="original":
        recipes=json.loads((work/"recipe-lock.json").read_bytes())["recipes"]
        docs=[d for d in docs if d["recipe"] in recipes]
    if len(docs)!=54:raise ValueError(f"{kind} lacks multiple independent seeds: {len(docs)}")
    return docs


def bank(kind,role,external=False):
    work=study(kind);co=arrays(work/"system-id.npz");r=prepare(read(role,kind=="repaired",external),co)
    if kind=="repaired":
        best=json.loads((work/"adaptive-lock.json").read_bytes())["best"];c=arrays(work/"adaptive.npz");r=decorate(r,best,c,c["mass"])
    return r


def parameters(d,kind):
    folder=roots("robot-corrected" if kind=="repaired" else "robot")[0]/d["tag"]/f"step-{d['best']['step']:05d}"
    if sha(folder/"weights.npz")!=d["weights_sha"]:raise ValueError("Model changed since selection")
    return {k:jnp.asarray(x) for k,x in arrays(folder/"weights.npz").items()}


def active_ref(r,recipe,mode):
    prefix="adapt" if recipe in ("adaptive","anchored") else "ref"
    suffix={"wrong_plan":"wrong","zero_plan":"zero","reset":"reset"}.get(mode,"active")
    return r[prefix+"_"+suffix]


def develop():
    out=ROOT/"work/score";out.mkdir(exist_ok=True,parents=True);lock={}
    for kind in ("original","repaired"):
        r=bank(kind,"dev")
        for d in records(kind):
            q=d["cfg"];p=parameters(d,kind);s=Spec(**q["spec"]);o=prediction(p,s,d["core"],d["recipe"],r)
            ref=active_ref(r,d["recipe"],"correct");delta=o["active"][:,-1]-ref[:,-1]
            valid=r["mover"]>=0;score=errors(o,r);confidence=.5*(o["mover"].max(1)+o["confidence"][:,0])
            opt=[]
            for a in (0.,.1,.25,.5,.75,1.):
                err=np.linalg.norm(ref[:,-1]+a*delta-r["future"][:,-1],axis=1)*1000;opt.append((group_mean(err,r["group"]),a))
            val,alpha=min(opt)
            rec=dict(core=d["core"],recipe=d["recipe"],seed=d["seed"],step=d["best"]["step"],sha256=d["weights_sha"],
                     alpha=alpha,dev_calibrated_mm=val,curve=d["curve"],parameters=q["parameters"],objective=d["best"]["objective"],
                     thresholds=[float(np.quantile(confidence[r["first"]],.6)),float(np.quantile(o["confidence"][:,1],.6))])
            lock[kind+"/"+d["tag"]]=rec
        print("Sealed decisions",kind,flush=True)
    write(out/"model-lock.json",lock)
    (out/"frozen").touch()


def confirm():
    out=ROOT/"work/score";lock=json.loads((out/"model-lock.json").read_bytes())
    for kind in ("original","repaired"):
        for cohort in ("hold","fresh"):
            r=bank(kind,"test",cohort=="fresh");path=out/kind/cohort;path.mkdir(exist_ok=True,parents=True)
            for d in records(kind):
                tag=d["tag"];doc=lock[kind+"/"+tag];file=path/(tag+".npz")
                if file.is_file() and file.with_suffix(".json").exists():continue
                p=parameters(d,kind);q=d["cfg"];s=Spec(**q["spec"])
                modes=("correct","wrong","none")
                if doc["seed"]==1301 and kind=="repaired" and doc["recipe"]=="anchored":
                    modes+=("no_vision","text_only","reset","zero_plan","wrong_plan")
                saved={}
                for mode in modes:
                    o=prediction(p,s,d["core"],d["recipe"],r,mode);ref=active_ref(r,d["recipe"],mode)
                    for k in ("active","passive"):saved[mode+"_"+k]=o[k][:,-1]
                    saved[mode+"_cal"]=ref[:,-1]+doc["alpha"]*(o["active"][:,-1]-ref[:,-1])
                    saved[mode+"_cmd"]=o["action"].sum(1);saved[mode+"_mover"]=o["mover"]
                    saved[mode+"_confidence"]=o["confidence"]
                tmp=file.with_suffix(".new.npz");np.savez_compressed(tmp,**saved);tmp.replace(file)
                write(file.with_suffix(".json"),dict(hash=sha(file),model=doc["sha256"],modes=modes))
            print("Confirmatory predictions",kind,cohort,flush=True)
    (out/"evaluated").touch()


if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("stage",choices=("develop","confirm"));args=a.parse_args()
    jax.config.update("jax_compilation_cache_dir",str(ROOT/"cache"));(develop if args.stage=="develop" else confirm)()
