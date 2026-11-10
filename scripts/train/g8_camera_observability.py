"""Measure direct visibility of changed physics at fixed development poses without changing the held protocol."""
import json
from pathlib import Path
import numpy as np
from training.g8_scenes import physics,initial,execute_prefix
from training.g7_units import put
from scripts.train.g6_artifacts import roots


def snapshots(env):
    result={}
    for size in (16,512):
        env.observation_width=size;env.observation_height=size
        rgb=env._render(visualize=False);result[size]=np.rint(rgb@np.array([.299,.587,.114])).astype(np.int16)
    result["area16"]=np.rint(result[512].reshape(16,32,16,32).mean((1,3))).astype(np.int16)
    env.observation_width=16;env.observation_height=16
    return result


def main():
    cases={"wall":[],"object":[]}
    for seed in range(1156000,1156064):
        env=physics("puck_barrier")
        _,state,prefix,_,_=initial("puck_barrier",seed);execute_prefix(env,seed,state,prefix,False)
        orig=snapshots(env);walls=[shape for shape in env.space.shapes if shape.collision_type==22]
        if len(walls)!=1:raise ValueError("Unexpected barrier geometry")
        env.space.remove(walls[0]);no_wall=snapshots(env);env.space.add(walls[0])
        now=env.block.position;env.block.position=(-1024,-1024);env.space.reindex_shapes_for_body(env.block)
        no_object=snapshots(env);env.block.position=now;env.close()
        for kind,blank in (("wall",no_wall),("object",no_object)):
            record={"seed":seed}
            for k in (16,512,"area16"):
                difference=np.abs(orig[k]-blank[k]);record[str(k)]={"changed_pixels":int((difference>0).sum()),"maximum_gray_delta":int(difference.max())}
            cases[kind].append(record)
    summary={}
    for k,records in cases.items():
        summary[k]={v:{"visible_scenes":sum(r[v]["changed_pixels"]>0 for r in records),"mean_changed_pixels":float(np.mean([r[v]["changed_pixels"] for r in records])),
            "mean_max_gray_delta":float(np.mean([r[v]["maximum_gray_delta"] for r in records]))} for v in ("16","512","area16")}
    for d in roots("g8"):
        (d/"public").mkdir(exist_ok=True);put(d/"public/camera-observability.json",{"role":"Supplementary development renderer audit after inspecting the sampling geometry, before confirmation outcomes were opened.",
            "independent_seeds":[1156000,1156064],"direct_render":"Original 16-pixel simulator camera.","area16":"Reference 512-pixel scene area-averaged to 16 pixels. This processing was not used by the registered models.",
            "interventions":"Remove only the barrier shape or move only the puck outside the view; do not advance the simulator between images.","summary":summary,"scenes":cases})
    print(json.dumps(summary),flush=True)


if __name__=="__main__":main()
