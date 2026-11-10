"""Rank candidate waypoint actions using a verified G8 world checkpoint and an observed scene."""
import argparse
import json
from pathlib import Path
import numpy as np
from training.g8_models import read,load_model,sealed,G8_DATA
from training.g8_pushing import predictions,panel,summarize
from scripts.train.g6_artifacts import roots
from scripts.train.g7_lock_heads import sha
from training.g7_units import put


def decide(spec,p,cal,n,v,plans,goal,condition="joint"):
    n=np.asarray(n,np.float32);v=np.asarray(v);plans=np.asarray(plans,np.float32);goal=np.asarray(goal,np.float64)
    if n.shape!=(16,6) or v.shape!=(16,256) or v.dtype!=np.uint8 or goal.shape!=(2,) or plans.ndim!=3 or plans.shape[1:]!=(16,2) or not len(plans):
        raise ValueError("Expected a 16-tick history, 16×256 uint8 images, a two-coordinate displacement goal and at least one 16×2 action plan")
    if not all(np.isfinite(x).all() for x in (n,plans,goal)):raise ValueError("Non-finite state, plan or goal")
    if condition not in ("joint","state","video"):raise ValueError("Unsupported source combination")
    outputs=[]
    # The registered evaluator uses 192 forward rows. Preserve its numerical kernel for single-scene inference.
    for offset in range(0,len(plans),192):
        block=plans[offset:offset+192];size=len(block);padded=np.pad(block,((0,192-size),(0,0),(0,0)),mode="edge")
        outputs.append(predictions(spec,p,cal,n[None],v[None],padded[None],condition)[0,:size])
    output=np.concatenate(outputs);dist=np.linalg.norm(output-goal[None],axis=-1);pick=int(dist.argmin())
    return {"chosen_plan":pick,"predicted_displacements_px":output.tolist(),"predicted_distances_to_goal_px":dist.tolist(),
            "selected_waypoints_xy_px":((plans[pick]+1)*256).tolist(),"horizon_physical_ticks":16,"sensor_condition":condition}


def sample_files():
    doc=sealed();a,_=roots("g8");out=a/"public/examples";out.mkdir(exist_ok=True,parents=True);summary={}
    for task in ("pusht","puck_barrier"):
        scen=panel(task,doc);f=out/(task+"-input.npz")
        with f.with_suffix(".partial").open("wb") as stream:np.savez_compressed(stream,n=scen["n"][0],v=scen["v"][0],plan=scen["plan"][0],goal_displacement_px=scen["goals"][0,0])
        f.with_suffix(".partial").replace(f)
        summary[task]={"first_reserved_start":int(scen["group"][0]),"input_sha256":sha(f),"controllers":{}}
        for core in ("hybrid","gpt2"):
            tag=f"{core}-511-other";rec=doc["models"][tag];spec,p,cal,_=load_model(rec)
            result=decide(spec,p,cal,scen["n"][0],scen["v"][0],scen["plan"][0],scen["goals"][0,0])
            measured=scen["outcome"][0,:,-1]*64;cost=np.linalg.norm(measured-scen["goals"][0,0],axis=-1)
            measured_row=summarize(np.asarray(result["predicted_displacements_px"])[None],measured[None],scen["goals"][:1])
            with np.load(a/"confirmation"/task/tag/"rows.npz") as registered:
                for field in ("endpoint_px","regret_px"):
                    np.testing.assert_allclose(measured_row[field][0],registered["joint_"+field][0],rtol=0,atol=1e-4)
            summary[task]["controllers"][tag]={**result,"weights":rec["weights"],"measured_distances_to_goal_px":cost.tolist(),
                "realized_regret_px":float(cost[result["chosen_plan"]]-cost.min())}
    put(a/"public/decision-examples.json",summary)


def main():
    args=argparse.ArgumentParser(description=__doc__);args.add_argument("--checkpoint",type=Path);args.add_argument("--input",type=Path)
    args.add_argument("--source",choices=("joint","video","state"),default="joint");args.add_argument("--examples",action="store_true")
    a=args.parse_args()
    if a.examples:sample_files();return
    if not a.input or not a.checkpoint:args.error("Supply a checkpoint directory and an input NPZ")
    meta=read(a.checkpoint/"complete.json");rec={"root":str(a.checkpoint),"weights":meta["sha256"]["inference.npz"]}
    spec,p,cal,_=load_model(rec)
    with np.load(a.input,allow_pickle=False) as f:output=decide(spec,p,cal,f["n"],f["v"],f["plan"],f["goal_displacement_px"],a.source)
    print(json.dumps({**output,"weights_sha256":rec["weights"]},indent=2,allow_nan=False))


if __name__=="__main__":main()
