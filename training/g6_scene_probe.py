"""Independent simulator interventions distinguishing visual evidence, plan information and action choice."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

LO=600000
HI=600512
CONDS={"video_correct":([1],"correct"),"video_erased":([1],"erase"),"video_wrong":([1],"wrong"),
       "video_delayed":([1],"delay"),"state":([2],"correct"),"joint_correct":([1,2],"correct"),
       "joint_wrong":([1,2],"wrong"),"joint_delayed":([1,2],"delay")}


def sha(p):
    with p.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()


def write(path,data):
    tmp=path.with_suffix(".new"); tmp.write_text(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n"); tmp.replace(path)


def prepare(folder):
    from training.g5_physical_data import simulate
    folder.mkdir(exist_ok=True,parents=True); dest=folder/"scenes.npz"
    if (folder/"card.json").is_file():
        if sha(dest)!=json.loads((folder/"card.json").read_bytes())["sha256"]:
            raise ValueError("Independent scenes changed")
        return
    records=folder/"records"; records.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        list(pool.map(simulate,[(seed,records) for seed in range(LO,HI)],chunksize=3))
    fields={k:[] for k in ("n","v","plan","outcome","group")}
    for seed in range(LO,HI):
        with np.load(records/f"{seed}.npz") as f:
            if int(f["group"][0])!=seed: raise ValueError("Unexpected simulator scene")
            for k in fields: fields[k].append(np.array(f[k]))
    arrays={k:np.concatenate(v) for k,v in fields.items()}
    rng=np.random.default_rng(610733)
    order=rng.permutation(HI-LO); wrong=np.empty(len(order),np.int32); wrong[order]=np.roll(order,1)
    if np.any(wrong==np.arange(len(order))): raise ValueError("A visual foil matched its source")
    theta=rng.uniform(-np.pi,np.pi,len(order)); radius=rng.uniform(40,100,len(order))
    arrays.update(wrong=wrong,goal=np.column_stack((np.cos(theta),np.sin(theta)))*radius[:,None])
    with dest.with_suffix(".partial").open("wb") as f: np.savez_compressed(f,**arrays)
    dest.with_suffix(".partial").replace(dest)
    write(folder/"card.json",{"sha256":sha(dest),"start":LO,"stop_exclusive":HI,"scenes":HI-LO,"plans_per_scene":3,
                              "generator_sha256":sha(Path("training/g5_physical_data.py")),"goal_and_foil_seed":610733})
    print("Prepared",HI-LO,"independent scenes",flush=True)


def score(a):
    import jax
    from models.g4_shared import Spec
    from models.g5_residual import world
    from training.g5_batches import JointData,load_pack
    from training.g5_eval import ridge
    card=json.loads((a.scenes/"card.json").read_bytes())
    if sha(a.scenes/"scenes.npz")!=card["sha256"]: raise ValueError("Changed scene manifest")
    with np.load(a.scenes/"scenes.npz") as f: sim={k:np.array(v) for k,v in f.items()}
    config=json.loads((a.run/"config.json").read_bytes()); rec=json.loads((a.run/"verified.json").read_bytes())
    if rec["step"]!=8192 or sha(a.run/"inference.npz")!=rec["sha256"]["inference.npz"]:
        raise ValueError("Expect verified terminal weights")
    spec=Spec(**config["spec"])
    with np.load(a.run/"inference.npz") as f:
        p={k:jax.device_put(f[k]) for k in f if k!="stats" and not k.startswith("prior_")}
        cal={k[6:]:jax.device_put(f[k]) for k in f if k.startswith("prior_")}
        dataset=JointData(a.data,"dev",verify=False,stats=f["stats"])
    dataset.packs[2]=sim
    train,_=load_pack(a.data/"pusht/train",False)
    @jax.jit
    def apply(batch):
        _,_,delta=world(p,spec,batch,batch["source"],cal)
        return delta[:,-1]*64
    n=len(sim["group"]); rows=np.repeat(np.arange(n),3); branches=np.tile(np.arange(3),n)
    predictions={}
    for name,(views,change) in CONDS.items():
        preds=[]
        for i in range(0,len(rows),64):
            ix=rows[i:i+64]; br=branches[i:i+64]
            b=dataset.gather(np.full(len(ix),8),ix,br)
            b["source"][:,views]=1
            if 2 not in views: b["n"][:]=0
            if 1 not in views or change=="erase": b["v"][:]=0
            if change=="erase": b["sense"][:]=0
            if change=="wrong": b["v"][:,:16]=sim["v"][sim["wrong"][ix]]
            if change=="delay":
                v=b["v"][:,:16].copy()
                b["v"][:,:16]=np.concatenate((np.repeat(v[:,:1],4,axis=1),v[:,:12]),axis=1)
            preds.append(np.asarray(apply(b)))
        predictions[name]=np.concatenate(preds).reshape(n,3,2)
    truth=sim["outcome"][:,:,-1,:].astype(np.float64)*64
    predictions["prior"]=ridge(train,sim)
    predictions["persistence"]=np.zeros_like(truth)
    actual=np.linalg.norm(truth-sim["goal"][:,None],axis=-1)
    oracle=actual.min(1); scores={}; samples={}; indices=np.random.default_rng(610734).integers(0,n,(2000,n))
    for name,values in predictions.items():
        error=values-truth
        endpoint=np.linalg.norm(error,axis=-1).mean(1)
        dif=np.mean([np.linalg.norm(error[:,l]-error[:,r],axis=-1) for l,r in ((0,1),(0,2),(1,2))],axis=0)
        target=np.linalg.norm(values-sim["goal"][:,None],axis=-1)
        best=np.isclose(target,target.min(1,keepdims=True),rtol=0,atol=1e-9)
        choices=best/best.sum(1,keepdims=True)
        regret=(choices*actual).sum(1)-oracle
        hit=(choices*np.isclose(actual,oracle[:,None],rtol=0,atol=1e-9)).sum(1)
        samples[name]={"endpoint_px":endpoint,"regret_px":regret,"branch_difference_px":dif,"optimal_rate":hit}
        scores[name]={k:float(v.mean()) for k,v in samples[name].items()}
    contrasts={}
    for better,control in (("video_correct","video_erased"),("video_correct","video_wrong"),("video_correct","video_delayed"),
                           ("joint_correct","state"),("joint_correct","joint_wrong"),("joint_correct","joint_delayed"),
                           ("joint_correct","prior"),("video_correct","persistence")):
        contrasts[better+"-over-"+control]={}
        for k in ("endpoint_px","regret_px"):
            delta=samples[control][k]-samples[better][k]
            contrasts[better+"-over-"+control][k]={"reduction":float(delta.mean()),
                "ci95":np.quantile(delta[indices].mean(1),[.025,.975]).tolist()}
    passed=all(contrasts[key][m]["ci95"][0]>0 for key in ("video_correct-over-video_erased","video_correct-over-video_wrong")
               for m in ("endpoint_px","regret_px"))
    out=a.output; out.mkdir(exist_ok=True,parents=True)
    with (out/"scene-rows.pending").open("wb") as f:
        flat={name+"_"+k:v for name,ms in samples.items() for k,v in ms.items()}
        np.savez_compressed(f,scene=sim["group"],**flat)
    (out/"scene-rows.pending").replace(out/"scene-rows.npz")
    write(out/"scene-controls.json",{"run":config["run"],"inference_sha256":rec["sha256"]["inference.npz"],"scenes":card,
          "role":"fresh independent simulator scenes","conditions":scores,"paired_contrasts":contrasts,
          "positive_visual_contribution":passed,"actual_oracle_goal_distance_px":float(oracle.mean()),
          "paired_rows_sha256":sha(out/"scene-rows.npz")})
    print(config["run"],"visual-control gate",passed,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("phase",choices=["prepare","score"])
    p.add_argument("--scenes",required=True,type=Path); p.add_argument("--data",type=Path); p.add_argument("--run",type=Path); p.add_argument("--output",type=Path)
    a=p.parse_args()
    if a.phase=="prepare": prepare(a.scenes)
    else: score(a)
