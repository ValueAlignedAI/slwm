"""Measure genuine one-decision physics controls with and without a hold action on fresh episodes."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from training.g8_scenes import physics,initial,normalized
from training.g8_live import commands
from training.g7_units import put
from scripts.train.g6_artifacts import roots
from scripts.train.g7_lock_heads import sha

RANGES={"pusht":(1160000,1160064),"puck_barrier":(1170000,1170064)}
POLICIES=("first_proposal","passive","oracle_three","oracle_plus_hold")


def replay(env,seed,state,prefix,history):
    env.reset(seed=int(seed),options={"reset_to_state":state.copy()})
    for step in prefix:pose=env.step(step)[0]
    for sequence in history:
        for step in sequence:pose=env.step(step)[0]
    return pose


def evaluate(job):
    task,seed,mode=job;env=physics(task);probe=physics(task)
    # Gym's goal-geometry intersection is an unused reward-only operation; verify the full original dynamics on the chosen branch below.
    probe._get_coverage=lambda:0.
    _,state,prefix,_,_=initial(task,seed);rng=np.random.default_rng([851733,seed]);phi=rng.uniform(-np.pi,np.pi);radius=rng.uniform(75,120)
    goal=np.array([512-state[2],state[3]+rng.uniform(-65,65)]) if task=="puck_barrier" else state[2:4]+radius*np.array([np.cos(phi),np.sin(phi)])
    goal=np.clip(goal,42,470);history=[];pose=replay(env,seed,state,prefix,history);dist=[];choices=[];maximum_error=0.
    for t in range(24):
        n=np.repeat(normalized(np.array([pose]))[None],16,axis=1)
        bank=commands(n,goal[None],np.array([seed]),t,task=="puck_barrier")[0]
        hold=np.repeat(pose[None,:2],16,axis=0)
        if mode=="passive":plan=hold;pick=3
        elif mode=="first_proposal":plan=bank[0];pick=0
        else:
            if mode=="oracle_plus_hold":bank=np.concatenate((bank,hold[None]))
            future=[]
            for alternative in bank:
                chk=replay(probe,seed,state,prefix,history);np.testing.assert_allclose(chk,pose,atol=1e-6,rtol=0)
                for command in alternative:tip=probe.step(command)[0][2:4]
                future.append(tip.copy())
            pick=int(np.linalg.norm(np.stack(future)-goal[None],axis=1).argmin());plan=bank[pick]
        for command in plan:pose=env.step(command)[0]
        if mode.startswith("oracle"):
            error=float(np.max(np.abs(pose[2:4]-future[pick])));maximum_error=max(maximum_error,error)
            np.testing.assert_allclose(pose[2:4],future[pick],atol=1e-6,rtol=0)
        history.append(plan.copy());choices.append(pick);dist.append(float(np.linalg.norm(pose[2:4]-goal)))
    env.close();probe.close()
    return {"task":task,"seed":seed,"mode":mode,"distance":dist,"choices":choices,"max_replay_difference":maximum_error}


def main():
    a,b=roots("g8");out=a/"live-oracle";out.mkdir(exist_ok=True)
    if (out/"complete.json").exists():return
    wanted=[(task,s,mode) for task,(lo,hi) in RANGES.items() for s in range(lo,hi) for mode in POLICIES]
    with ProcessPoolExecutor(max_workers=5) as pool:rows=list(pool.map(evaluate,wanted,chunksize=2))
    results={};arrays={};rng=np.random.default_rng(81109);draw=rng.integers(64,size=(4000,64))
    for task in RANGES:
        group={};vals={}
        for mode in POLICIES:
            cases=sorted([r for r in rows if r["mode"]==mode and r["task"]==task],key=lambda r:r["seed"])
            if [r["seed"] for r in cases]!=list(range(*RANGES[task])):raise ValueError("Incomplete paired oracle bank")
            dist=np.asarray([r["distance"] for r in cases]);choices=np.asarray([r["choices"] for r in cases]);final=dist[:,-1];wins=(final<=20).astype(float)
            arrays[task+"_"+mode+"_distance"]=dist;arrays[task+"_"+mode+"_choice"]=choices;vals[mode]=(wins,final)
            group[mode]={"final_success":float(wins.mean()),"mean_final_px":float(final.mean()),"ever_success":float((dist.min(1)<=20).mean()),
                "hold_fraction":float((choices==3).mean()),"maximum_executed_branch_replay_error":max(r["max_replay_difference"] for r in cases)}
        effects={}
        for index,key in enumerate(("final_success","mean_final_px")):
            diff=vals["oracle_plus_hold"][index]-vals["oracle_three"][index]
            effects[key]={"hold_minus_three":float(diff.mean()),"paired_ci95":np.quantile(diff[draw].mean(1),[.025,.975]).tolist()}
        results[task]={"policies":group,"hold_effect":effects}
    for disk in (a,b):
        folder=disk/"live-oracle";folder.mkdir(exist_ok=True)
        with (folder/"rows.part").open("wb") as f:np.savez_compressed(f,**arrays)
        (folder/"rows.part").replace(folder/"rows.npz")
    pin=sha(out/"rows.npz")
    if sha(b/"live-oracle/rows.npz")!=pin:raise ValueError("Physical oracle trajectory copies differ")
    doc={"design":"Post hoc mechanism study on 64 new paired starts per task. Reconstructed counterfactual dynamics; no learned controller was trained or selected.",
         "ranges":RANGES,"decisions":24,"ticks":384,"rows_sha256":pin,"results":results}
    for disk in (a,b):
        put(disk/"live-oracle/complete.json",doc);put(disk/"public/live-oracle.json",doc)
    print("Independent live oracle controls complete",flush=True)


if __name__=="__main__":main()
