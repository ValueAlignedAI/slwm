"""Build replay-checked, paired physical outcomes before opening any trained model's predictions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from training.g4_data import digest
from training.g5_physical_data import simulate,save_npz
from training.g7_units import put

CFG=Path("configs/t8/factorial.json")
PANEL_SEED=810319
LEVELS=("pusht","puck_barrier")


def physics(task):
    from gym_pusht.envs.pusht import PushTEnv
    import pygame
    import pymunk
    if task=="pusht":cls=PushTEnv
    elif task=="puck_barrier":
        class Puck(PushTEnv):
            @staticmethod
            def add_tee(space,position,angle,scale=30,color="LightSlateGray",mask=None):
                angles=2*np.pi*np.arange(24)/24
                vertices=list(zip(28*np.cos(angles),28*np.sin(angles)))
                mass=1.;body=pymunk.Body(mass,pymunk.moment_for_poly(mass,vertices))
                body.position=position;body.angle=angle
                shape=pymunk.Poly(body,vertices);shape.friction=.55;shape.color=pygame.Color(color);shape.collision_type=21
                space.add(body,shape);return body,[shape]
            def _setup(self):
                super()._setup();wall=self.add_segment(self.space,(256,178),(256,334),5)
                wall.color=pygame.Color("DimGray");wall.collision_type=22;self.space.add(wall);self.puck_contacts=0
                handler=self.space.add_collision_handler(21,22)
                def touched(arbiter,space,data):self.puck_contacts+=1
                handler.post_solve=touched
        cls=Puck
    else:raise ValueError(task)
    return cls(obs_type="state",render_mode="rgb_array",observation_width=16,observation_height=16)


def initial(task,seed):
    rng=np.random.default_rng(seed)
    if task=="pusht":
        block=rng.uniform(125,385,2);angle=rng.uniform(-np.pi,np.pi);heading=rng.uniform(0,2*np.pi)
    else:
        side=rng.choice((-1,1))
        block=np.array([256+side*rng.uniform(49,88),rng.uniform(135,377)])
        angle=rng.uniform(-np.pi,np.pi);heading=(np.pi if side>0 else 0)+rng.uniform(-.42,.42)
    direction=np.array([np.cos(heading),np.sin(heading)]);agent=block-rng.uniform(65,116)*direction
    entry=block-rng.uniform(12,52)*direction
    return rng,np.r_[agent,block,angle],np.linspace(agent,entry,16),heading,entry


def observation(env):
    image=env._render(visualize=False)
    return np.rint(image@np.array([.299,.587,.114])).astype(np.uint8).reshape(256)


def normalized(poses):
    return np.column_stack((poses[:,:4]/256-1,np.sin(poses[:,4]),np.cos(poses[:,4]))).astype(np.float32)


def execute_prefix(env,seed,state,prefix,frames=True):
    env.reset(seed=int(seed),options={"reset_to_state":state.copy()});poses=[];pixels=[]
    for u in prefix:
        poses.append(env.step(u)[0])
        if frames:pixels.append(observation(env))
    return np.stack(poses),np.stack(pixels) if frames else None


def puck(job):
    seed,dest=job;file=dest/f"{seed}.npz"
    if file.exists():return
    env=physics("puck_barrier");rng,state,prefix,heading,entry=initial("puck_barrier",seed)
    angles=heading+rng.uniform(-.32,.32,3)+np.array([0,1.35,-1.35])
    plans=[];outcomes=[];contacts=[]
    for i,h in enumerate(angles):
        hist,video=execute_prefix(env,seed,state,prefix,frames=i==0)
        if i==0:reference=hist.copy();saved=video
        else:np.testing.assert_allclose(hist,reference,atol=1e-7,rtol=0)
        end=np.clip(entry+rng.uniform(85,150)*np.array([np.cos(h),np.sin(h)]),35,477)
        commands=np.linspace(entry,end,17)[1:];count=env.puck_contacts
        future=np.stack([env.step(u)[0] for u in commands])
        outcomes.append((future[:,2:4]-hist[-1,2:4])/64);plans.append(commands);contacts.append(env.puck_contacts-count)
    execute_prefix(env,seed,state,prefix,frames=False)
    replay=(np.stack([env.step(u)[0] for u in plans[0]])[:,2:4]-reference[-1,2:4])/64
    np.testing.assert_allclose(replay,outcomes[0],rtol=0,atol=1e-7);env.close()
    save_npz(file,{"n":normalized(reference)[None],"v":saved[None],"plan":(np.stack(plans)[None]/256-1).astype(np.float32),
        "outcome":np.asarray(outcomes,np.float32)[None],"group":np.array([seed],np.int64),"barrier_contacts":np.array([contacts],np.int32)})


def confirm_push(root,task,lo,hi,workers):
    dest=root/task;card=dest/"card.json";pack=dest/"scenes.npz"
    if card.exists():
        if digest(pack)!=json.loads(card.read_bytes())["sha256"]:raise ValueError("Changed held pushing scenes")
        return
    records=dest/"records";records.mkdir(exist_ok=True,parents=True)
    with ProcessPoolExecutor(max_workers=workers) as p:list(p.map(simulate if task=="pusht" else puck,((s,records) for s in range(lo,hi)),chunksize=4))
    arrays={}
    for s in range(lo,hi):
        with np.load(records/f"{s}.npz") as f:
            if f["group"][0]!=s:raise ValueError("Wrong held reset")
            for k,v in f.items():arrays.setdefault(k,[]).append(v)
    arrays={k:np.concatenate(v) for k,v in arrays.items()};n=hi-lo;rng=np.random.default_rng([PANEL_SEED,lo])
    angles=rng.uniform(-np.pi,np.pi,(n,2));radii=rng.uniform(40,100,(n,2));d=np.stack((radii*np.cos(angles),radii*np.sin(angles)),axis=-1)
    arrays["goals"]=np.stack((d[:,0],-d[:,0],d[:,1],-d[:,1]),axis=1).astype(np.float32)
    order=rng.permutation(n);wrong=np.empty(n,np.int32);wrong[order]=np.roll(order,1);arrays["wrong"]=wrong
    truth=arrays["outcome"][:,:,-1]*64;cost=np.linalg.norm(truth[:,:,None]-arrays["goals"][:,None],axis=-1)
    with pack.with_suffix(".part").open("wb") as f:np.savez_compressed(f,**arrays)
    pack.with_suffix(".part").replace(pack)
    put(card,{"task":task,"range":[lo,hi],"scenes":n,"goals_per_scene":4,"candidate_plans":3,"sha256":digest(pack),
        "generator_sha256":digest(Path("training/g8_scenes.py")),"original_simulator_sha256":digest(Path("training/g5_physical_data.py")),
        "barrier_contact_scene_fraction":float((arrays["barrier_contacts"].sum(1)>0).mean()) if "barrier_contacts" in arrays else None,
        "uniform_action_regret_px":float((cost.mean(1)-cost.min(1)).mean()),"nonidentical_branch_fraction":float((np.linalg.norm(truth[:,0]-truth[:,1],axis=-1)>.01).mean())})
    print(task,n,digest(pack),flush=True)


def reacher(root,lo,hi,workers):
    from training.g7_reacher import example
    dest=root/"reacher_test";dest.mkdir(exist_ok=True,parents=True);card=dest/"card.json"
    if card.is_file():return
    records=dest/"records";records.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as p:files=list(p.map(example,((s,records) for s in range(lo,hi)),chunksize=4))
    with np.load(files[0]) as first:
        arr={k:np.lib.format.open_memmap(dest/(k+".npy"),mode="w+",dtype=v.dtype,shape=(hi-lo,*v.shape)) for k,v in first.items()}
    for i,f in enumerate(files):
        with np.load(f) as x:
            if int(x["seed"])!=lo+i:raise ValueError("Wrong fresh Reacher start")
            for k,v in x.items():arr[k][i]=v
    for a in arr.values():a.flush()
    put(card,{"range":[lo,hi],"scenes":hi-lo,"source_generator_sha256":digest(Path("training/g7_reacher.py")),"sha256":{k:digest(dest/(k+".npy")) for k in arr}})
    print("Reacher",hi-lo,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("task",choices=(*LEVELS,"reacher_test"));p.add_argument("--root",required=True,type=Path);p.add_argument("--workers",type=int,default=4)
    a=p.parse_args();lo,hi=json.loads(CFG.read_bytes())["panels"][a.task]
    if a.task=="reacher_test":reacher(a.root,lo,hi,a.workers)
    else:confirm_push(a.root,a.task,lo,hi,a.workers)
