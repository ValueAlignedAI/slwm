"""A second manipulation environment with identical present observations and counterbalanced past goal cues."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import tempfile

import numpy as np

H=8
C=4
K=9
ENV=None


def environment():
    import gymnasium as gym
    e=gym.make("Reacher-v5",render_mode="rgb_array",width=32,height=32).unwrapped
    # A visible static target in the vanilla environment would disclose an unused goal.
    import mujoco
    idx=mujoco.mj_name2id(e.model,mujoco.mjtObj.mjOBJ_GEOM,"target")
    if idx<0:raise ValueError("Cannot identify simulator target geometry")
    e.model.geom_rgba[idx,3]=0.
    return e


def setup(e,seed):
    rng=np.random.default_rng(seed); e.reset(seed=int(seed))
    q=e.data.qpos.copy(); vel=e.data.qvel.copy(); q[:2]=rng.uniform(-np.pi,np.pi,2); vel[:2]=rng.uniform(-.35,.35,2)
    q[2:4]=0;vel[2:4]=0;e.set_state(q,vel)
    angle=rng.uniform(0,2*np.pi); radius=rng.uniform(.115,.185)
    g=radius*np.array([np.cos(angle),np.sin(angle)])
    return rng,np.stack([g,-g]).astype(np.float32)


def actions(rng):
    d=np.arange(8)*np.pi/4+rng.uniform(-np.pi/8,np.pi/8);amp=rng.uniform(.60,.98)
    return np.concatenate([np.stack([np.cos(d),np.sin(d)],-1)*amp,np.zeros((1,2))]).astype(np.float32)


def observe(e,control):
    q=e.data.qpos[:2];vel=e.data.qvel[:2];tip=e.get_body_com("fingertip")[:2]
    result=np.zeros(17,np.float32);result[:2]=tip/.21
    result[2:4]=[np.cos(q[0])*.1/.21,np.sin(q[0])*.1/.21]
    result[4:8]=[np.sin(q[0]),np.cos(q[0]),np.sin(q[1]),np.cos(q[1])]
    result[8:10]=np.clip(vel/8,-4,4);result[10:12]=control
    img=e.render()
    gray=np.tensordot(img[...,:3],np.array([.299,.587,.114]),axes=1)
    return result,np.rint(gray.reshape(16,2,16,2).mean((1,3))).astype(np.uint8).reshape(256)


def advance(e,control,images=True):
    ns=[];vs=[]
    for _ in range(H):
        e.step(control)
        if images:
            n,v=observe(e,control);ns.append(n);vs.append(v)
    return (np.stack(ns),np.stack(vs)) if images else np.asarray(e.get_body_com("fingertip")[:2],np.float32).copy()


def branches(e,bank):
    import mujoco as mj
    spec=mj.mjtState.mjSTATE_INTEGRATION
    snapshot=np.empty(mj.mj_stateSize(e.model,spec));mj.mj_getState(e.model,e.data,snapshot,spec)
    def reset():
        mj.mj_setState(e.model,e.data,snapshot,spec);mj.mj_forward(e.model,e.data)
    output=[]
    for a in bank:
        reset();output.append(advance(e,a,False))
    reset();check=advance(e,bank[0],False);np.testing.assert_allclose(check,output[0],rtol=0,atol=1e-6)
    reset();return np.stack(output)


def example(job):
    global ENV
    seed,dest=job;path=dest/f"{seed}.npz"
    if path.is_file():return str(path)
    if ENV is None:ENV=environment()
    rng,goals=setup(ENV,seed); control=rng.uniform(-.8,.8,2).astype(np.float32)
    n=[];v=[];banks=[];out=[];executed=[]
    for _ in range(C):
        nums,view=advance(ENV,control); bank=actions(rng); end=branches(ENV,bank)
        n.append(nums);v.append(view);banks.append(bank);out.append(end)
        pick=int(rng.integers(K));executed.append(pick);control=bank[pick]
    with path.with_suffix(".part").open("wb") as f:np.savez(f,n=np.stack(n),v=np.stack(v),actions=np.stack(banks),end=np.stack(out),goals=goals,executed=np.array(executed,np.uint8),seed=np.int32(seed))
    path.with_suffix(".part").replace(path);return str(path)


def prepare(root,workers):
    from training.g4_data import digest
    roles=json.loads(Path("configs/t7/study.json").read_bytes())["policy"]
    for role in ("train","dev","test"):
        lo,hi=roles[role]; dest=root/"reacher"/role
        if (dest/"card.json").is_file():continue
        pieces=root/"reacher-records"/role;pieces.mkdir(exist_ok=True,parents=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            files=[]
            for i,p in enumerate(pool.map(example,((s,pieces) for s in range(lo,hi)),chunksize=4),1):
                files.append(p)
                if i%512==0:print(role,i,hi-lo,flush=True)
        dest.mkdir(exist_ok=True,parents=True); arrays={}
        with np.load(files[0]) as f:
            for k,v in f.items():arrays[k]=np.lib.format.open_memmap(dest/(k+".npy"),mode="w+",dtype=v.dtype,shape=(len(files),*v.shape))
        for j,path in enumerate(files):
            with np.load(path) as f:
                for k in arrays:arrays[k][j]=f[k]
        for arr in arrays.values():arr.flush()
        d=np.linalg.norm(arrays["end"][:,:,None]-arrays["goals"][:,None,:,None],axis=-1)
        ambiguous=float((d.argmin(-1)[:,:,0]!=d.argmin(-1)[:,:,1]).mean())
        card={"role":role,"seeds":[lo,hi],"scenes":hi-lo,"cycles":C,"plans":K,"future_control_steps":H,"counterbalanced_goals":2,
              "different_best_action_fraction":ambiguous,"simulator":"Gymnasium 1.2.2 Reacher-v5 / MuJoCo 3.3.7",
              "sha256":{k:digest(dest/(k+".npy")) for k in arrays}}
        (dest/"card.tmp").write_text(json.dumps(card,sort_keys=True,indent=2)+"\n");(dest/"card.tmp").replace(dest/"card.json")
        print(json.dumps(card),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--workers",type=int,default=3)
    a=p.parse_args();prepare(a.root,a.workers)
