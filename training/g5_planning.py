"""Sustained 24-decision closed-loop Push-T probe using shared streaming state and identical proposals."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from models.g4_shared import Spec
from models.g5_shared import encode
from training.g4_train import sha_file, write
from training.g5_priors import geometry


def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-6)


def proposals(history,goal):
    agent=(history[:,-1,:2]+1)*256
    block=(history[:,-1,2:4]+1)*256
    heading=unit(goal-block)
    phi=np.arange(8)*np.pi/4
    dirs=np.column_stack((np.cos(phi),np.sin(phi)))
    fixed=agent[:,None]+96*dirs
    endpoints=np.concatenate((fixed,(block+96*heading)[:,None],(block-55*heading)[:,None]),axis=1)
    endpoints=np.clip(endpoints,38,474)
    grid=np.arange(1,17,dtype=float)[None,None,:,None]/16
    return (agent[:,None,None]+grid*(endpoints-agent[:,None])[:,:,None]).astype(np.float32)


def normalized(states):
    return np.concatenate((states[:,:,:4]/256-1,np.sin(states[:,:,4:5]),np.cos(states[:,:,4:5])),axis=-1).astype(np.float32)


def features(v,n,u,sense):
    b,t=u.shape[:2]
    return {"v":v,"a":np.zeros((b,t,100),np.uint8),"n":n,"u":u,"input_x":np.zeros((b,t),np.int32),
            "ok":np.ones((b,t),np.uint8),"sense":np.full((b,t),sense,np.uint8)}


def planner(run):
    card=json.loads((run/"config.json").read_bytes())
    if sha_file(run/"inference.npz")!=json.loads((run/"complete.json").read_bytes())["sha256"]["inference.npz"]:
        raise ValueError("Planning weights changed")
    spec=Spec(**card["spec"])
    with np.load(run/"inference.npz") as f:
        p={k:jax.device_put(f[k]) for k in f if k!="stats" and not k.startswith("prior_")}
    @jax.jit
    def field(b,mem,offset):
        src=jnp.broadcast_to(jnp.array([[0,1,1,0]],jnp.uint8),(len(b["v"]),4))
        z,mem=encode(p,spec,b,src,memory=mem,offset=offset)
        return (z[:,-1]@p["readout"][:,633:635].astype(z.dtype)).astype(jnp.float32),mem
    def candidates(history,video,plans,memory,clock,prev_frame,prev_n):
        b,c=plans.shape[:2]
        v=video.astype(np.float32)
        dv=v-np.concatenate((prev_frame[:,None] if prev_frame is not None else v[:,:1],v[:,:-1]),1)
        n=np.zeros((b,16,17),np.float32)
        n[:,:,:6]=history
        n[:,:,6:8]=history[:,:,:2]-history[:,:,2:4]
        n[:,:,8:10]=history[:,:,2:4]-np.concatenate((prev_n[:,None,2:4] if prev_n is not None else history[:,:1,2:4],
                                                       history[:,:-1,2:4]),1)
        obs=features(np.clip(v+.65*dv,0,255),n,np.zeros((b,16,2),np.float32),1)
        _,memory=field(obs,memory,np.int32(clock))
        expanded=jax.tree.map(lambda x:jnp.repeat(x,c,axis=0),memory)
        control=(plans/256-1-history[:,-1,None,None,2:4]).reshape(b*c,16,2)
        admitted=features(np.zeros((b*c,16,256),np.uint8),np.zeros((b*c,16,17),np.float32),control,0)
        residual,branches=field(admitted,expanded,np.int32(clock+16))
        return np.asarray(residual).reshape(b,c,2)*16,branches
    return candidates


def experiment(root,data,out,sim_site,memory_study=False,runs=None,first_seed=None):
    sys.path.append(str(sim_site))
    from gym_pusht.envs.pusht import PushTEnv
    with np.load(data/"prior.npz") as f:
        w=f["coef"][-1]
    first_seed=(401000 if memory_study else 400000) if first_seed is None else first_seed
    seeds=np.arange(first_seed,first_seed+32)
    rules=("passive","prior","hybrid","hybrid-reset","hybrid-local","gpt2","gpt2-reset","gpt2-local") if memory_study else ("random","geometry","prior","hybrid","gpt2")
    if runs is not None:
        rules=("passive","prior","hybrid","hybrid-reset","gpt2","gpt2-reset")
    output={}
    for rule in rules:
        core=rule.split("-")[0]
        learned=core in ("hybrid","gpt2")
        run=(runs[core] if runs is not None else root/f"g5-medium-{core}-511-residual-gpu") if learned else None
        model=planner(run) if learned else None
        envs=[PushTEnv(obs_type="state",render_mode="rgb_array",observation_width=16,observation_height=16) for _ in seeds]
        poses=[]; frames=[]; goals=[]; rng_sample=np.random.default_rng(30303)
        gray=np.array([.299,.587,.114])
        def observe(env,plan):
            hist=[]; image=[]
            for command in plan:
                hist.append(env.step(command)[0])
                if learned:
                    image.append(np.rint(env._render(visualize=False)@gray).astype(np.uint8).reshape(-1))
            return np.stack(hist),np.stack(image) if image else None
        for env,seed in zip(envs,seeds):
            rng=np.random.default_rng(seed)
            block=rng.uniform(125,385,2); theta=rng.uniform(-np.pi,np.pi); phi=rng.uniform(-np.pi,np.pi)
            direction=np.array([np.cos(phi),np.sin(phi)])
            agent=block-rng.uniform(65,116)*direction
            entry=block-rng.uniform(12,52)*direction
            env.reset(seed=int(seed),options={"reset_to_state":np.r_[agent,block,theta]})
            h,v=observe(env,np.linspace(agent,entry,16)); poses.append(h); frames.append(v)
            angle=rng.uniform(-np.pi,np.pi)
            goals.append(np.clip(h[-1,2:4]+rng.uniform(60,140)*np.array([np.cos(angle),np.sin(angle)]),70,442))
        goals=np.stack(goals); n=len(seeds); memory=None; prev_v=None; prev_n=None
        poses=np.stack(poses)
        frames=np.stack(frames) if learned else None
        distances=[np.linalg.norm(poses[:,-1,2:4]-goals,axis=-1)]
        for cycle in range(24):
            history=normalized(poses)
            plans=proposals(history,goals)
            n,c=plans.shape[:2]
            vel=geometry(np.repeat(history,c,0),plans.reshape(n*c,16,2)/256-1,[3,7,11,15])@w[1:]+w[0]
            future=poses[:,-1,None,2:4]+64*vel.reshape(n,c,2)
            if learned:
                reset=rule.endswith("-reset")
                clock=0 if reset or rule.endswith("-local") else cycle*32
                correction,branches=model(history,frames,plans,None if reset else memory,clock,
                                          None if reset else prev_v,None if reset else prev_n)
                future+=correction
            behind=future-55*unit(goals[:,None]-future)
            cost=np.linalg.norm(future-goals[:,None],axis=-1)+.15*np.linalg.norm(plans[:,:,-1]-behind,axis=-1)
            if rule=="random":
                pick=rng_sample.integers(c,size=n)
            elif rule=="geometry":
                block=poses[:,-1,2:4]; agent=poses[:,-1,:2]
                goal_dir=unit(goals-block)
                aligned=np.sum(unit(block-agent)*goal_dir,axis=1)>.75
                close=np.linalg.norm(agent-block,axis=1)<100
                pick=np.where(aligned&close,8,9)
            else:
                pick=cost.argmin(1)
            if learned:
                memory=jax.tree.map(lambda a:a[np.arange(n)*c+pick],branches)
                prev_n,prev_v=history[:,-1],frames[:,-1]
            actions=plans[np.arange(n),pick]
            if rule=="passive":
                actions=np.repeat(poses[:,-1,None,:2],16,axis=1)
            arr=[observe(env,plan) for env,plan in zip(envs,actions)]
            poses=np.stack([q[0] for q in arr])
            frames=np.stack([q[1] for q in arr]) if learned else None
            distances.append(np.linalg.norm(poses[:,-1,2:4]-goals,axis=-1))
        for env in envs:
            env.close()
        dist=np.stack(distances)
        output[rule]={"mean_initial_px":float(dist[0].mean()),"mean_final_px":float(dist[-1].mean()),
                      "final_within_30_px":float((dist[-1]<=30).mean()),"ever_within_30_px":float((dist.min(0)<=30).mean()),
                      "final_distances_px":dist[-1].tolist()}
        print(rule,output[rule]["mean_final_px"],output[rule]["ever_within_30_px"],flush=True)
        write(out,{"complete":len(output)==len(rules),"seed_range":[int(seeds[0]),int(seeds[-1])],"cycles":24,"seconds":38.4,"episodes":len(seeds),"controls":output})
        jax.clear_caches()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True); p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True); p.add_argument("--sim-site",type=Path,required=True)
    p.add_argument("--memory-study",action="store_true")
    a=p.parse_args(); experiment(a.root,a.data,a.out,a.sim_site,a.memory_study)
