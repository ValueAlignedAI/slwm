"""Calibrate vision, validate a geometrical controller, and record executed candidate effects."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import cv2
import numpy as np
from data.g14_table import Table,ROOT,TOL,OTHER,CHUNK,LIMIT,COLORS,typed,options,geometric,utility,evaluate_path
from g14_runtime import json_file

def obs_pack(table):
    h=table.history
    return {k:np.stack([v[k] for v in h]) for k in ("q","seen","p","motor","px","u")}

def calibrate():
    table=Table();xs=[];ys=[]
    for i in range(128):
        table.reset(11000+i);xs.append(typed(table.render()));ys.append(table.positions())
    x=np.stack(xs);y=np.stack(ys);H=[];stats={}
    for c in range(4):
        fit=x[:96,c,4]>0;h,mask=cv2.findHomography(x[:96,c,:2][fit].astype("f8"),y[:96,c][fit].astype("f8"),cv2.RANSAC,.009)
        if h is None or mask.sum()<25:raise ValueError("Training camera calibration failed")
        H.append(h)
        valid=x[96:,c,4]>0;uv=x[96:,c,:2][valid];p=np.column_stack((uv,np.ones(len(uv))))@h.T
        err=1000*np.linalg.norm(p[:,:2]/p[:,2:]-y[96:,c][valid],axis=1)
        stats[COLORS[c]]=dict(train=int(fit.sum()),train_inliers=int(mask.sum()),held_seen=int(valid.sum()),median_mm=float(np.median(err)),p90_mm=float(np.quantile(err,.9)))
    np.savez_compressed(ROOT/"data/camera.npz",H=np.stack(H));json_file(ROOT/"data/camera.json",stats)
    print("Independent calibration starts",stats,flush=True)

def prepare(seed,shift,table):
    signature=table.reset(seed,shift);stations=table.stations(seed)
    return signature,stations

def basic_roll(table,c,target,mode):
    initial=table.positions();first=table.physical_signature();best=1.;hold=0;arrived=False;selective=False;worst=0.;effort=0.;final=0.;tr=[]
    for tick in range(LIMIT):
        obs=table.observation()
        if mode=="oracle":obs={**obs,"q":table.positions(),"seen":np.ones(4)}
        if mode=="silent":cmd=np.zeros(2)
        else:cmd=geometric(obs,c,target,mode!="straight",int(mode=="slow"))
        q,_=table.execute(cmd);effort+=np.linalg.norm(cmd)
        distance,others=evaluate_path(q,initial,c,target)
        for dd,other in zip(distance,others):
            worst=max(worst,float(other));hold=hold+1 if dd<=TOL else 0
            if hold>=3:
                arrived=True;selective|=worst<=OTHER
        best=min(best,distance.min());final=float(distance[-1]);tr.append([tick,final,worst,*cmd])
        if arrived:break
    return dict(success=int(selective),arrival=int(arrived),min_error=float(best),final_error=final,other=worst,
                macros=tick+1,effort=float(effort),start=first,end=table.positions().tolist(),trace=tr)

def init_worker():
    global ENV
    cv2.setNumThreads(1);ENV=Table();ENV.cal=np.load(ROOT/"data/camera.npz")["H"]

def pilot_one(seed):
    t=ENV;sig,stations=prepare(seed,False,t);rows=[]
    for c,target in ((seed%3,stations[0]),((seed+1)%3,stations[1])):
        scene=tuple(t.positions().ravel())
        for mode in ("straight","geometric","slow","oracle"):
            t.restore(t.root_state);res=basic_roll(t,c,target,mode);res.pop("trace")
            if res["start"]!=sig:raise ValueError("Counterfactual start changed")
            rows.append(dict(seed=seed,mode=mode,color=c,target=target.tolist(),**res))
        # Compare two independently restored repetitions of the exact same command sequence.
        t.restore(t.root_state);commands=np.random.default_rng(seed).uniform(-.02,.02,(5,2));a=[t.execute(u)[0] for u in commands]
        t.restore(t.root_state);b=[t.execute(u)[0] for u in commands]
        if np.max(np.abs(np.array(a)-b))>1e-5:raise ValueError("Nonrepeatable Bullet continuation from snapshot")
    return rows

def collect_world(args):
    kind,seed=args;t=ENV
    folder=ROOT/"data"/kind;file=folder/f"{seed}.npz"
    if file.exists():return (seed,0)
    shift=kind=="shift-train" or kind=="development" and seed%4==0
    signature,stations=prepare(seed,shift,t);rng=np.random.default_rng(seed+3314)
    selection=json.loads((ROOT/"work/geometry-choice.json").read_bytes())["variant"]
    rows=[];notes=[]
    for trial in range(2):
        t.restore(t.root_state)
        if t.physical_signature()!=signature:raise ValueError("Training snapshot drift")
        color=int((seed+trial)%3);target=stations[(seed//3+trial)%4];initial=t.positions()
        if trial==1 and seed%19==0:target=initial[color].copy()
        for tick in range(LIMIT):
            obs=t.observation();choices=options(obs,color,target,selection);gx=t.positions();ep=t.effector()
            if tick==0 or rng.random()<.42:
                saved=t.save();effect=[];gain=[];dwell=[]
                try:
                    for action in choices:
                        t.restore(saved);q,ee=t.execute(action)
                        effect.append(np.concatenate(((ee[-1]-ep)[None,:],q[-1]-gx),0))
                        gain.append(utility(gx,ep,q[-1],ee[-1],color,target,action))
                        dwell.append(np.linalg.norm(q[-1,color]-target))
                finally:t.restore(saved);t.p.removeState(saved[0])
                rows.append(dict(**obs_pack(t),color=color,target=target,choices=choices,effect=np.stack(effect),gain=np.asarray(gain),measured=np.asarray(dwell),world=seed,trial=trial,clock=tick))
            # Fixed training uses demonstrations and controlled exploration. Development holds out every scene.
            policy=1
            if seed%4==0 and rng.random()<.54:policy=int(rng.integers(1,len(choices)))
            elif trial==1 and seed%5==0:policy=6
            cmd=choices[policy]
            q,ee=t.execute(cmd)
            if (np.linalg.norm(q[:,color]-target,axis=1)<TOL).all():break
        notes.append(dict(goal=color,macros=tick+1,last_mm=float(1000*np.linalg.norm(t.positions()[color]-target))))
    data={k:np.stack([r[k] for r in rows]) for k in rows[0]}
    tmp=file.with_suffix(".part.npz");np.savez_compressed(tmp,**data);tmp.replace(file)
    return seed,len(rows)

def main(mode,count):
    if mode=="calibrate":calibrate();return
    workers=min(8,count)
    if mode=="pilot":
        with ProcessPoolExecutor(workers,initializer=init_worker) as pool:rows=[r for chunk in pool.map(pilot_one,range(12100,12100+count)) for r in chunk]
        stats={c:dict(success=float(np.mean([q["success"] for q in rows if q["mode"]==c])),arrival=float(np.mean([q["arrival"] for q in rows if q["mode"]==c])),
                      mean_mm=float(1000*np.mean([q["final_error"] for q in rows if q["mode"]==c]))) for c in ("geometric","slow","straight","oracle")}
        chosen=min(("geometric","slow"),key=lambda c:(-stats[c]["success"],stats[c]["mean_mm"]))
        json_file(ROOT/"work/geometry-choice.json",dict(variant=int(chosen=="slow"),selected=chosen,pilot=stats,starts=count,rows=rows))
        print("Geometric development",stats,chosen,flush=True);return
    start={"training":140000,"development":150000,"fresh-teacher":155000,"shift-train":158000}[mode]
    folder=ROOT/"data"/mode;folder.mkdir(exist_ok=True)
    with ProcessPoolExecutor(workers,initializer=init_worker) as pool:
        for i,(seed,n) in enumerate(pool.map(collect_world,((mode,k) for k in range(start,start+count))),1):
            if i%16==0:print(mode,i,count,flush=True)
    parts=sorted(folder.glob("*.npz"));blocks=[]
    for path in parts:
        with np.load(path) as z:blocks.append({k:z[k] for k in z.files})
    dst=ROOT/"data"/(mode+".npz");tmp=dst.with_suffix(".part.npz")
    np.savez_compressed(tmp,**{k:np.concatenate([r[k] for r in blocks]) for k in blocks[0]});tmp.replace(dst)
    print("Complete executed branch bank",mode,len(parts),sum(len(x["world"]) for x in blocks),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("calibrate","pilot","training","development","fresh-teacher","shift-train"));p.add_argument("--count",type=int,default=32);a=p.parse_args();main(a.mode,a.count)
