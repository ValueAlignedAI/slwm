"""Batched, closed-loop goal reaching with actual simulator transitions and common candidate-generation rules."""
from multiprocessing import get_context
from pathlib import Path
import numpy as np
from training.g8_scenes import physics,initial,execute_prefix,normalized,observation


def commands(n,goal,seeds,tick,puck):
    """A goal-aware proposal bank shared by learned, raw-geometry and heuristic policies."""
    agent=(n[:,-1,:2]+1)*256;block=(n[:,-1,2:4]+1)*256
    angle=np.arctan2(goal[:,1]-block[:,1],goal[:,0]-block[:,0]);all_plans=[]
    for i,seed in enumerate(seeds):
        rng=np.random.default_rng([851731,int(seed),tick])
        headings=angle[i]+np.array([0.,1.4,-1.4])+rng.uniform(-.15,.15,3)
        direction=np.stack((np.cos(headings),np.sin(headings)),axis=1);size=45 if puck else 64
        targets=block[i]-size*direction
        travel=np.clip(np.linalg.norm(goal[i]-block[i]),30,98)
        finish=block[i]+travel*direction
        plans=[np.concatenate((np.linspace(agent[i],approach,7)[1:],np.linspace(approach,end,11)[1:])) for approach,end in zip(targets,finish)]
        all_plans.append(np.clip(plans,26,486))
    return np.asarray(all_plans,np.float32)


def worker(conn,task,seeds):
    envs=[physics(task) for _ in seeds]
    try:
        while True:
            name,payload=conn.recv()
            if name=="close":break
            nums=[];vids=[];goals=[]
            for i,(env,seed) in enumerate(zip(envs,seeds)):
                if name=="reset":
                    _,state,prefix,_,_=initial(task,int(seed))
                    poses,video=execute_prefix(env,seed,state,prefix,frames=payload)
                    target_rng=np.random.default_rng([851733,int(seed)])
                    phi=target_rng.uniform(-np.pi,np.pi);radius=target_rng.uniform(75,120)
                    if task=="puck_barrier":goal=np.array([512-state[2],state[3]+target_rng.uniform(-65,65)])
                    else:goal=state[2:4]+radius*np.array([np.cos(phi),np.sin(phi)])
                    goals.append(np.clip(goal,42,470))
                elif name=="step":
                    plans,render=payload;states=[];frames=[]
                    for u in plans[i]:
                        states.append(env.step(u)[0])
                        if render:frames.append(observation(env))
                    poses=np.stack(states);video=np.stack(frames) if render else None
                else:raise ValueError("Unrecognized simulation operation")
                nums.append(normalized(poses));vids.append(video if video is not None else np.zeros((16,256),np.uint8))
            conn.send((np.stack(nums),np.stack(vids),np.asarray(goals,np.float32),envs[0].control_hz))
    except BaseException as e:
        try:conn.send(("failure",repr(e)))
        finally:raise
    finally:
        for env in envs:env.close()
        conn.close()


class ScenePool:
    def __init__(self,task,lo,hi,workers=4):
        self.task=task;self.seeds=np.arange(lo,hi);self.ends=[];self.children=[];self.splits=np.array_split(np.arange(len(self.seeds)),workers)
        ctx=get_context("spawn")
        for ids in self.splits:
            a,b=ctx.Pipe();child=ctx.Process(target=worker,args=(b,task,self.seeds[ids]));child.start();b.close()
            self.ends.append(a);self.children.append(child)
    def exchange(self,cmd,payloads):
        for conn,payload in zip(self.ends,payloads):conn.send((cmd,payload))
        values=[]
        for conn in self.ends:
            if not conn.poll(180):raise TimeoutError("Simulator workers stopped responding")
            data=conn.recv()
            if isinstance(data[0],str):raise RuntimeError(str(data))
            values.append(data)
        return tuple(np.concatenate([v[k] for v in values]) for k in range(3))+(float(values[0][3]),)
    def reset(self,render):
        return self.exchange("reset",[render]*len(self.ends))
    def step(self,plans,render):
        return self.exchange("step",[(plans[ix],render) for ix in self.splits])
    def close(self):
        for c in self.ends:
            try:c.send(("close",None))
            except (BrokenPipeError,EOFError):pass
        for child in self.children:
            child.join(5)
            if child.is_alive():child.terminate();child.join()
        for c in self.ends:c.close()


def rollout(pool,tag,spec,p,cal,prior,dest,weight):
    from training.g4_train import write
    from training.g8_models import read
    from training.g8_pushing import predictions
    from scripts.train.g7_lock_heads import sha
    dest=dest/"live"/pool.task/tag;dest.mkdir(parents=True,exist_ok=True)
    if (dest/"metrics.json").exists():
        rec=read(dest/"metrics.json")
        if rec["weights"]!=weight or rec["rows_sha256"]!=sha(dest/"trajectory.npz"):raise ValueError("Changed live artifact")
        return
    render=spec is not None;n,v,goal,hz=pool.reset(render);steps=24;nworlds=len(n)
    positions=[(n[:,-1,2:4]+1)*256];agents=[(n[:,-1,:2]+1)*256];chosen=[];issued=[]
    for t in range(steps):
        bank=commands(n,goal,pool.seeds,t,pool.task=="puck_barrier");u=bank/256-1
        if tag=="passive":
            pick=np.full(nworlds,-1);actual=np.repeat(agents[-1][:,None],16,1)
        elif tag=="uniform":
            pick=np.array([np.random.default_rng([851737,int(s),t]).integers(3) for s in pool.seeds]);actual=bank[np.arange(nworlds),pick]
        elif tag=="first_proposal":
            pick=np.zeros(nworlds,int);actual=bank[:,0]
        else:
            future=prior.predict(n,u) if tag=="geometry" else predictions(spec,p,cal,n,v,u)
            proposed=positions[-1][:,None]+future
            pick=np.linalg.norm(proposed-goal[:,None],axis=-1).argmin(1);actual=bank[np.arange(nworlds),pick]
        chosen.append(pick);issued.append(actual);n,v,_,_=pool.step(actual,render)
        positions.append((n[:,-1,2:4]+1)*256);agents.append((n[:,-1,:2]+1)*256)
    xy=np.stack(positions,1);distance=np.linalg.norm(xy-goal[:,None],axis=-1);final=distance[:,-1];success=final<=20
    with (dest/"trajectory.part").open("wb") as f:
        np.savez_compressed(f,seed=pool.seeds,goals=goal,object_xy=xy,agent_xy=np.stack(agents,1),executed=np.stack(issued,1),
            choice=np.stack(chosen,1),distance=distance,final_px=final,success=success,ever_success=(distance[:,1:]<=20).any(1))
    (dest/"trajectory.part").replace(dest/"trajectory.npz")
    write(dest/"metrics.json",{"controller":tag,"task":pool.task,"weights":weight,"starts":nworlds,"decisions":steps,"ticks_per_decision":16,
          "control_hz":hz,"simulated_seconds":steps*16/hz,"mean_initial_px":float(distance[:,0].mean()),"mean_final_px":float(final.mean()),
          "success":float(success.mean()),"ever_success":float((distance[:,1:].min(1)<=20).mean()),"rows_sha256":sha(dest/"trajectory.npz")})
