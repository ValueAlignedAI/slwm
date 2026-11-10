"""Scene-paired factorial contrasts for decision quality, zero-shot transfer and long rollouts."""
import argparse
from pathlib import Path
import numpy as np
from training.g8_models import read,sealed,key,CONFIG
from scripts.train.g6_artifacts import roots
from scripts.train.g7_lock_heads import sha
from training.g7_units import put

CONTRASTS={"other_at_replayed_sim":{"other":1,"replay":-1},"other_at_fresh_sim":{"fresh":1,"sim":-1},
    "sim_at_replayed_other":{"sim":1,"replay":-1},"sim_at_fresh_other":{"fresh":1,"other":-1},
    "other_main":{"other":.5,"replay":-.5,"fresh":.5,"sim":-.5},
    "sim_main":{"sim":.5,"replay":-.5,"fresh":.5,"other":-.5},
    "interaction":{"fresh":1,"sim":-1,"other":-1,"replay":1},"additional_replay":{"replay":1,"parent":-1}}


def estimate(x,draw):
    means=x[draw].mean(1)
    return {"mean":float(x.mean()),"paired_ci95":np.quantile(means,[.025,.975]).tolist(),"simultaneous_upper_one_sided":float(np.quantile(means,.9875))}


def effect(cells,weights):
    return sum(w*cells[arm] for arm,w in weights.items())


def metrics_file(root,task,model,live):
    path=root/("live" if live else "confirmation")/task/model;meta=read(path/"metrics.json")
    reps=path/("trajectory.npz" if live else "rows.npz");pin=meta.get("rows_sha256") or meta.get("row_sha256")
    if sha(reps)!=pin:raise ValueError("Changed held measurements")
    with np.load(reps) as f:
        ids=np.array(f["seed" if live or task=="reacher" else "group"])
        if live:vals={"terminal_distance_px":np.array(f["final_px"]),"final_success":np.array(f["success"]).astype(float),"ever_success":np.array(f["ever_success"]).astype(float)}
        elif task=="reacher":vals={k:np.array(f[k]) for k in ("regret_cm","endpoint_cm","hit")}
        else:vals={k:np.array(f[k]) for k in f.files if k!="group"}
    return ids,vals,meta


def compare_task(root,task,live):
    cfg=read(CONFIG);expected=np.arange(*cfg["panels"][("live_"+task) if live else "reacher_test" if task=="reacher" else task])
    draw=np.random.default_rng(cfg["bootstrap_seed"]).integers(len(expected),size=(cfg["draws"],len(expected)),dtype=np.int32)
    results={};data={};allrows={"start":expected}
    for core in cfg["cores"]:
        for seed in cfg["seeds"]:
            vals={};means={}
            for arm in ("parent","replay","sim","other","fresh"):
                name=key(core,seed,arm);ix,scores,meta=metrics_file(root,task,name,live);np.testing.assert_array_equal(expected,ix)
                vals[arm]=scores;means[arm]={k:float(v.mean()) for k,v in scores.items()}
                for metric,v in scores.items():
                    if live or task=="reacher" or metric.startswith("joint_") and not metric.startswith("joint_wrong"):
                        allrows[name+"__"+metric]=v
            comparisons={}
            for metric in vals["fresh"]:
                series={arm:v[metric].astype(float) for arm,v in vals.items()}
                if live or task=="reacher" or metric in ("joint_regret_px","joint_endpoint_px","joint_best_plan_rate","state_regret_px","video_regret_px"):
                    comparisons[metric]={k:estimate(effect(series,w),draw) for k,w in CONTRASTS.items()}
            results[f"{core}-{seed}"]={"cells":means,"paired_contrasts":comparisons};data[core,seed]=vals
    interactions={};architecture={}
    for seed in cfg["seeds"]:
        h,d=data["hybrid",seed],data["gpt2",seed];interactions[str(seed)]={};architecture[str(seed)]={}
        names=("terminal_distance_px","final_success") if live else ("regret_cm","endpoint_cm") if task=="reacher" else ("joint_regret_px","joint_endpoint_px")
        for metric in names:
            differences={arm:h[arm][metric].astype(float)-d[arm][metric].astype(float) for arm in h}
            interactions[str(seed)][metric]={k:estimate(effect(differences,w),draw) for k,w in CONTRASTS.items() if k in ("other_main","sim_main","interaction")}
            architecture[str(seed)][metric]={arm:estimate(v,draw) for arm,v in differences.items()}
    return {"unit":"one physical start","starts":len(expected),"series":results,"hybrid_minus_dense_interventions":interactions,
            "hybrid_minus_dense_by_cell":architecture},allrows


def analyze(out):
    root,_=roots("g8");doc=sealed();out.mkdir(parents=True,exist_ok=True)
    results={};packed={};controls={}
    for task,live in (("pusht",False),("puck_barrier",False),("reacher",False),("pusht",True),("puck_barrier",True)):
        label=("live_" if live else "")+task
        result,rows=compare_task(root,task,live);results[label]=result
        for k,v in rows.items():packed[label+"__"+k]=v
        if live:
            public={n:read(root/"live"/task/n/"metrics.json") for n in ("geometry","first_proposal","uniform","passive")}
            for n in sorted(doc["models"]):
                public[n]=read(root/"live"/task/n/"metrics.json")
            controls[label]=public
        elif task=="reacher":controls[label]={k:read(root/"confirmation"/"reacher"/k/"metrics.json") for k in ("physical","raw-511","raw-512")}
        else:controls[label]=read(root/"confirmation"/task/"raw/metrics.json")
    gates=[]
    for task in ("pusht","puck_barrier"):
        for seed in (511,512):
            d=results[task]["series"][f"hybrid-{seed}"]["paired_contrasts"]["joint_regret_px"]["other_main"]
            gates.append({"task":task,"seed":seed,**d,"improves":d["simultaneous_upper_one_sided"]<0})
    with (out/"paired-scenes.partial").open("wb") as f:np.savez_compressed(f,**packed)
    (out/"paired-scenes.partial").replace(out/"paired-scenes.npz")
    put(out/"factorial.json",{"primary_gate":all(v["improves"] for v in gates),"familywise_upper_quantile":.9875,"primary":gates,"panels":results,
        "differences":"Negative favors new data for error, distance and regret. Positive favors new data for success or optimal choice.",
        "row_file_sha256":sha(out/"paired-scenes.npz"),"lock_sha256":sha(root/"confirmation-lock.json")})
    put(out/"raw-and-live.json",controls)
    put(out/"retention.json",{n:read(root/"retention"/n/"metrics.json") for n in doc["models"]})
    print("Completed paired evidence; primary non-simulator transfer gate",all(v["improves"] for v in gates),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("out",type=Path);analyze(p.parse_args().out)
