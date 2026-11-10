"""Run all locked G8 physical and retention evaluations, preserving each complete case for restart."""
import argparse
from functools import lru_cache
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np
from models.g5_residual import world
from training.g5_eval import average_precision
from training.g7_fresh_eval import signal_scores
from training.g7_units import put
from training.g8_models import CONFIG,G7_DATA,G8_DATA,load_model,sealed,read
from training.g8_pushing import panel,evaluate,Geometry
from training.g8_reacher import score as reacher
from training.g8_live import ScenePool,rollout
from scripts.train.g6_artifacts import roots
from scripts.train.g7_lock_heads import sha

OLD=G7_DATA.parent/"g5"


@lru_cache(maxsize=2)
def event_kernel(spec):
    @jax.jit
    def apply(p,cal,v,sense):
        n=len(v);b={"a":jnp.zeros((n,64,100)),"v":v,"n":jnp.zeros((n,64,17)),"u":jnp.zeros((n,64,2)),
            "input_x":jnp.zeros((n,64),jnp.int32),"ok":jnp.ones((n,64),jnp.uint8),"sense":sense,"family":jnp.full((n,),7)}
        source=jnp.broadcast_to(jnp.array([0,1,0,0],jnp.uint8),(n,4));return world(p,spec,b,source,cal)[1]
    return apply


def retention(name,rec,doc,spec,p,cal,dest):
    out=dest/"retention"/name;out.mkdir(parents=True,exist_ok=True);report=out/"metrics.json"
    if report.exists():
        if read(report)["weight"]!=rec["weights"]:raise ValueError("Retention model changed")
        return
    for f,pin in doc["previous_panels"].items():
        if sha(G7_DATA/"fresh"/f/"test/card.json")!=pin:raise ValueError("Reused diagnostic data changed")
    stage=Path(rec["root"])
    if rec["study"] in ("g6","g7"):
        g7,_=roots("g7");folder=g7/"scale-eval"/rec["run"]/stage.name
        signal,events=read(folder/"test-signals.json"),read(folder/"test-metrics.json")
        if signal["checkpoint"]!=rec["weights"] or events["checkpoint"]!=rec["weights"]:raise ValueError("Different historical diagnostic")
        ap=events["impact"]["average_precision"];method="verified archived G7 diagnostics"
    else:
        signal_scores(stage,G7_DATA,out,"test");signal=read(out/"test-signals.json")
        if signal["checkpoint"]!=rec["weights"]:raise ValueError("Retained model changed")
        data=G7_DATA/"fresh/events/test";v=np.load(data/"v.npy",mmap_mode="r");y=np.load(data/"event.npy")[:,8:].flatten()
        ap={};apply=event_kernel(spec)
        for condition in ("correct","shift","erase","raw_motion"):
            values=[]
            for j in range(0,len(v),64):
                chunk=np.asarray(v[j:j+64]);n=len(chunk);chunk=np.pad(chunk,((0,64-n),(0,0),(0,0)),mode="edge")
                if condition=="raw_motion":z=np.abs(chunk[:,8:].astype(np.float32)-chunk[:,7:-1].astype(np.float32)).mean(-1)
                else:
                    if condition=="shift":chunk[:,8:]=chunk[:,:-8].copy()
                    sense=np.full((64,64),condition!="erase",np.uint8);z=np.asarray(apply(p,cal,chunk,sense))[:,8:]
                values.append(z[:n].reshape(-1))
            ap[condition]=average_precision(np.concatenate(values),y,np.ones(len(y)))
        method="same frozen G7 examples rescored"
    put(report,{"model":name,"weight":rec["weights"],"method":method,"signals":signal["metrics"],"held_event_ap":ap})


def run(sim_site):
    if jax.devices()[0].platform!="gpu":raise RuntimeError("Expected the Fedora GPU")
    sys.path.append(str(sim_site))
    cfg=read(CONFIG);doc=sealed();dest,_=roots("g8");prior=Geometry(OLD)
    tasks=("pusht","puck_barrier");panels={task:panel(task,doc) for task in tasks}
    pools={}
    try:
        for task in tasks:
            pools[task]=ScenePool(task,*cfg["panels"]["live_"+task]);evaluate(task,"raw",None,None,None,panels[task],prior,dest,"original_train_ridge")
            for baseline in ("geometry","uniform","first_proposal","passive"):
                rollout(pools[task],baseline,None,None,None,prior,dest,"fixed-"+baseline)
        for baseline in ("physical","raw-511","raw-512"):reacher(baseline,doc)
        for name,rec in sorted(doc["models"].items()):
            spec,p,cal,_=load_model(rec)
            for task in tasks:
                evaluate(task,name,spec,p,cal,panels[task],prior,dest,rec["weights"])
                rollout(pools[task],name,spec,p,cal,prior,dest,rec["weights"])
            reacher(name,doc);retention(name,rec,doc,spec,p,cal,dest)
            print("Held model complete",name,flush=True)
    finally:
        for pool in pools.values():pool.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--sim-site",type=Path,required=True);run(parser.parse_args().sim_site)
