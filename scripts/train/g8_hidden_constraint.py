"""Quantify the price of a genuinely unobserved physical constraint on separately reserved paired worlds."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from training.g8_scenes import physics,initial,execute_prefix
from training.g7_units import put
from scripts.train.g6_artifacts import roots
from scripts.train.g7_lock_heads import sha


def example(seed):
    rng,state,_,heading,_=initial("puck_barrier",seed)
    direction=np.array([np.cos(heading),np.sin(heading)])
    block=state[2:4];approach=block-50*direction;prefix=np.linspace(state[:2],approach,16)
    angles=heading+np.array([0.,1.35,-1.35])+rng.uniform(-.22,.22,3)
    plans=[np.linspace(approach,np.clip(approach+rng.uniform(105,155)*np.array([np.cos(phi),np.sin(phi)]),32,480),17)[1:] for phi in angles]
    all_y=[];past=None;image=None;contact_any=False
    for wall_present in (True,False):
        env=physics("puck_barrier");future=[]
        for branch,plan in enumerate([*plans,plans[0]]):
            env.reset(seed=int(seed),options={"reset_to_state":state.copy()})
            barrier=next(s for s in env.space.shapes if s.collision_type==22)
            if not wall_present:env.space.remove(barrier)
            history=[];camera=[]
            from training.g8_scenes import observation
            for command in prefix:
                history.append(env.step(command)[0])
                if branch==0:camera.append(observation(env))
            history=np.stack(history)
            if past is None:past=history.copy()
            else:np.testing.assert_allclose(history,past,atol=1e-7,rtol=0)
            if branch==0:
                if image is None:image=np.stack(camera)
                else:np.testing.assert_array_equal(image,np.stack(camera))
            before=env.puck_contacts;xy=np.stack([env.step(a)[0][2:4] for a in plan])-history[-1,2:4]
            if wall_present and env.puck_contacts>before:contact_any=True
            if branch==3:np.testing.assert_allclose(xy,future[0],atol=1e-6,rtol=0)
            else:future.append(xy)
        all_y.append(np.stack(future)[:,-1]);env.close()
    y=np.stack(all_y);g=np.random.default_rng([81867,seed]);angles=g.uniform(-np.pi,np.pi,2);length=g.uniform(40,100,2)
    d=length[:,None]*np.stack([np.cos(angles),np.sin(angles)],1);goals=np.stack([d[0],-d[0],d[1],-d[1]])
    cost=np.linalg.norm(y[:,:,None]-goals[None,None],axis=-1)
    minimum=cost.min(1);best=cost.argmin(1)
    common=(cost.mean(0).min(0)-minimum.mean(0))
    if (common<-1e-8).any():raise ValueError("A blind oracle exceeded the informed oracle")
    optimum=np.isclose(cost,minimum[:,None,:],rtol=0,atol=1e-8)
    disjoint=~(optimum[0]&optimum[1]).any(0)
    return {"seed":seed,"excess":float(common.mean()),"different_optimum":float(disjoint.mean()),"contacts":contact_any,
            "endpoint_change_px":float(np.linalg.norm(y[0]-y[1],axis=-1).mean()),"all_conditional_costs":cost.tolist()}


def main():
    with ProcessPoolExecutor(max_workers=5) as pool:rows=list(pool.map(example,range(1157000,1157512),chunksize=4))
    n=len(rows);draw=np.random.default_rng(81867).integers(n,size=(4000,n));summary={}
    for k in ("excess","different_optimum","contacts","endpoint_change_px"):
        x=np.array([r[k] for r in rows],float);summary[k]={"mean":float(x.mean()),"paired_scene_ci95":np.quantile(x[draw].mean(1),[.025,.975]).tolist()}
    a,b=roots("g8")
    for disk in (a,b):
        out=disk/"hidden-constraint";out.mkdir(exist_ok=True)
        with (out/"rows.part").open("wb") as f:np.savez_compressed(f,seed=[r["seed"] for r in rows],excess=[r["excess"] for r in rows],
            optimal_sets_disjoint=[r["different_optimum"] for r in rows],cost=[r["all_conditional_costs"] for r in rows])
        (out/"rows.part").replace(out/"rows.npz")
    if sha(a/"hidden-constraint/rows.npz")!=sha(b/"hidden-constraint/rows.npz"):raise ValueError("Oracle rows differ across disks")
    for disk in (a,b):
        put(disk/"public/hidden-constraint.json",{"seeds":[1157000,1157512],"pairs":n,"goals_per_start":4,
            "identical_native_camera_and_proprioception":True,"world_prior":[.5,.5],"rows_sha256":sha(disk/"hidden-constraint/rows.npz"),
            "summaries":summary,"units":"Excess realized goal distance in pixels; disjoint optimal choices reported as a fraction of goals."})
    print(json.dumps(summary),flush=True)


if __name__=="__main__":main()
