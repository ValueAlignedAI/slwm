"""One frozen four-command replication on four new whole shards and two new initialization seeds."""
import argparse
import json
from pathlib import Path
import jax
import numpy as np
from evals.g13_predict import bank,parameters,arrays
from evals.g13_stats import Groups
from scripts.train.g13_robot_fit import ROOT,Spec,prediction
from scripts.train.g13_corpus_reference import fit,apply
from scripts.train.g13_optim import save_npz
from scripts.train.g9_storage import write,sha
from training.g13_physics import prepare
from training.g13_adaptive import decorate

CORES=("gpt2","dense_spatial","associative","transport")
SEEDS=(1301,1302,1303,1311,1312)
def choices(out):
    work=ROOT/"work/repaired";locks={}
    for seed in SEEDS:
        for core in CORES:
            tag=f"{core}-linear-{seed}";d=json.loads((work/"dev"/(tag+".json")).read_bytes())
            locks[tag]=dict(step=d["best"]["step"],sha256=d["weights_sha"],dev_four_command_mm=d["best"]["action"])
    ref=fit(bank("repaired","train"),bank("repaired","dev"));save_npz(out/"raw-action.npz",ref)
    write(out/"lock.json",dict(models=locks,ridge=float(ref["lam"]),raw_dev_mm=float(ref["development_mm"]),raw_sha256=sha(out/"raw-action.npz")))

def confirm(out):
    lock=json.loads((out/"lock.json").read_bytes());dst=ROOT/"release/metrics";work=ROOT/"work/repaired";root=ROOT/"data/robot-prospective"
    r=prepare(arrays(root/"corrected/test.npz"),arrays(work/"system-id.npz"))
    co=arrays(work/"adaptive.npz");best=json.loads((work/"adaptive-lock.json").read_bytes())["best"];r=decorate(r,best,co,co["mass"])
    if sha(out/"raw-action.npz")!=lock["raw_sha256"]:raise ValueError("Raw reference differs from the lock")
    raw=apply(r,arrays(out/"raw-action.npz"));y=r["plan"].sum(1);g=r["group"];bs=Groups(g)
    results={};obs=dict(truth=y,group=g,episode=r["episode"],tick=r["tick"],novel=r["novel"],raw=raw.astype("f4"))
    errors={}
    def err(x):return np.linalg.norm(x-y,axis=1)*1000
    controls=dict(raw=err(raw),zero=err(np.zeros_like(y)),repeat=err(4*r["kin"][:,-1,4:6]))
    for seed in SEEDS:
        for core in CORES:
            tag=f"{core}-linear-{seed}";d=json.loads((work/"dev"/(tag+".json")).read_bytes())
            if d["weights_sha"]!=lock["models"][tag]["sha256"]:raise ValueError("Changed prospective model")
            pp=parameters(d,"repaired");outp=prediction(pp,Spec(),core,"linear",r)
            pred=outp["action"].sum(1);obs[tag]=pred;errors[tag]=err(pred)
    delta=[]
    for seed in SEEDS:
        tr=errors[f"transport-linear-{seed}"];row={}
        for name in CORES:
            er=errors[f"{name}-linear-{seed}"];row[name]=dict(mm=bs.score(er),gain_vs_raw=bs.score(controls["raw"]-er))
        for rival in ("gpt2","dense_spatial","associative"):
            diff=errors[f"{rival}-linear-{seed}"]-tr;row["over_"+rival]=bs.score(diff);row["novel_over_"+rival]=bs.score(np.where(r["novel"],diff,np.nan))
        delta.append(errors[f"gpt2-linear-{seed}"]-tr);results[str(seed)]=row
    diff=np.mean(delta,axis=0);summary=bs.score(diff)
    ids,inv,count=np.unique(g,return_inverse=True,return_counts=True)
    means=np.bincount(inv,weights=diff)/count
    rng=np.random.default_rng(608141);draws=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=10000)@means/len(ids)
    ci=np.quantile(draws,[.005,.995]);summary["lo99"],summary["hi99"]=map(float,ci)
    passed=bool(ci[0]>0 and all(results[str(s)]["over_gpt2"]["value"]>0 for s in (1311,1312)) and sum(results[str(s)]["over_gpt2"]["lo"]>0 for s in (1301,1302,1303))>=2)
    write(dst/"prospective.json",dict(passed=passed,rows=len(g),episodes=len(np.unique(r["episode"])),parent_groups=len(ids),novel_windows=int(r["novel"].sum()),
        five_seed_gain=summary,seed_results=results,controls={k:bs.score(v) for k,v in controls.items()},
        source= json.loads((root/"banks/data.json").read_bytes())["counts"],test_sha256=sha(root/"corrected/test.npz")))
    np.savez_compressed(dst/"prospective-actions.npz",**obs);print("Prospective gate",passed,summary,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=("develop","test"));a=p.parse_args();out=ROOT/"work/prospective";out.mkdir(exist_ok=True)
    jax.config.update("jax_compilation_cache_dir",str(ROOT/"cache"));(choices if a.phase=="develop" else confirm)(out)
