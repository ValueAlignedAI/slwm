"""Learn goal-independent physical transitions and compare latent one- and two-decision planning on new starts."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import jax
import jax.numpy as jnp
import optax
from training.g4_train import sha_file,write
from training.g7_policy import base_weights,adjusted,batch,data,policy_fields
from training.g7_reacher import H,actions,advance,environment,setup
from training.g7_rl import stream,zeros
from scripts.train.g7_lock_heads import lock


def mlp_init(sizes,seed):
    rng=np.random.default_rng(seed);p={}
    for i,(m,n) in enumerate(zip(sizes,sizes[1:])):
        p[f"w{i}"]=jnp.asarray(rng.normal(0,m**-.5,(m,n)),jnp.float32);p[f"b{i}"]=jnp.zeros(n)
    return p


def mlp(p,x):
    n=len(p)//2
    for i in range(n):
        x=x@p[f"w{i}"]+p[f"b{i}"]
        if i+1<n:x=jax.nn.silu(x)
    return x


def cached(path):
    with np.load(path) as f:return {k:jax.device_put(v) for k,v in f.items()}


def save(path,weights):
    with path.with_suffix(".part").open("wb") as f:np.savez(f,**jax.device_get(weights))
    path.with_suffix(".part").replace(path)


def train_model(folder,sizes,inputs,targets,mask,steps,seed):
    net=mlp_init(sizes,seed);opt=optax.adamw(.001,weight_decay=.0005);st=opt.init(net)
    @jax.jit
    def learn(p,o,x,y,m):
        def loss(p):
            sq=(mlp(p,x)-y)**2
            return jnp.mean((sq*m).sum(0)/jnp.maximum(m.sum(0),1))
        v,g=jax.value_and_grad(loss)(p);ch,o=opt.update(g,o,p)
        return optax.apply_updates(p,ch),o,v
    inputs,targets,mask=map(jax.device_put,(inputs,targets,mask))
    rng=np.random.default_rng(seed)
    for it in range(steps):
        ix=rng.integers(len(inputs),size=768);net,st,val=learn(net,st,inputs[ix],targets[ix],mask[ix])
    if not np.isfinite(float(val)):raise FloatingPointError("Transition/readout loss diverged")
    folder.mkdir(parents=True,exist_ok=True)
    save(folder/"weights.npz",net);save(folder/"optimizer.npz",{f"v{i}":v for i,v in enumerate(jax.tree.leaves(st))})
    return net


def features(x,u,xp):
    v=xp.concatenate((x[...,2:6],u),axis=-1)
    return xp.concatenate((v,xp.sin(v),xp.cos(v),xp.sin(3*v),xp.cos(3*v)),axis=-1)


def unit(pair,xp):
    return pair/xp.maximum(.05,xp.sqrt((pair**2).sum(-1,keepdims=True)))


def transition(p,state,action):
    """Apply one eight-step action and return predicted state and physical endpoint."""
    y=mlp(p,features(state,action,jnp));s,c=state[...,0],state[...,1]
    tip=jnp.stack((c*y[...,0]-s*y[...,1],s*y[...,0]+c*y[...,1]),-1)*.21
    sc=unit(y[...,2:4],jnp);sd,cd=sc[...,0],sc[...,1]
    q0=jnp.stack((s*cd+c*sd,c*cd-s*sd),axis=-1)
    return jnp.concatenate((q0,unit(y[...,4:6],jnp),y[...,6:8].clip(-4,4)),axis=-1),tip


def transition_data(root,role):
    path=root/"reacher"/role;rows=data(path);a=np.asarray(rows["actions"]);n=np.asarray(rows["n"]);targets=np.asarray(rows["end"])/.21
    state=n[:,:,-1,4:10];bank=np.broadcast_to(state[:,:,None],(*a.shape[:-1],6))
    features_in=features(bank,a,np); y=np.zeros((*a.shape[:-1],8),np.float32);mask=np.zeros_like(y)
    s,c=bank[...,0],bank[...,1]
    y[...,0]=c*targets[...,0]+s*targets[...,1];y[...,1]=-s*targets[...,0]+c*targets[...,1];mask[...,:2]=1
    executed=np.load(path/"executed.npy").astype(int)
    row,cycle=np.indices((len(a),3));idx=(row,cycle,executed[:,:3])
    np.testing.assert_allclose(a[idx],n[:,1:,-1,10:12],rtol=0,atol=1e-6)
    now=state[:,:-1];future=state[:,1:]
    ds=future[...,0]*now[...,1]-future[...,1]*now[...,0]
    dc=future[...,1]*now[...,1]+future[...,0]*now[...,0]
    labels=y[idx].copy();labels[...,2:]=np.concatenate((ds[...,None],dc[...,None],future[...,2:]),-1)
    y[idx]=labels;mask[idx]=np.ones_like(labels)
    return features_in.reshape(-1,30),y.reshape(-1,8),mask.reshape(-1,8),state


def train_transition(a):
    folder=a.out/"transition"
    if (folder/"complete.json").exists():return
    x,y,m,_=transition_data(a.root,"train")
    p=train_model(folder,(30,192,192,192,8),x,y,m,3600,71821)
    x,y,m,_=transition_data(a.root,"dev")
    err=(np.asarray(mlp(p,x))-y)**2
    pred=np.sqrt((err*m).sum(0)/np.maximum(m.sum(0),1))
    write(folder/"complete.json",{"weights":sha_file(folder/"weights.npz"),"optimizer":sha_file(folder/"optimizer.npz"),
        "updates":3600,"branch_endpoints":147456,"full_next_states":12288,"dev_component_rmse":pred.tolist()})


def load_head(a):
    root=a.parents.parent/"g7";lockfile=json.loads((root/"policy-prior/sft-lock.json").read_bytes())
    chosen=lockfile["models"][f"{a.core}-{a.seed}"];file=root/"policy-prior"/f"{a.core}-{chosen['mode']}-{a.seed}"/"weights.npz"
    if sha_file(file)!=chosen["sha256"]:raise ValueError("SFT model changed")
    spec,p,_,_,parent=base_weights(a.parents,a.core,a.seed)
    with np.load(file) as f:h={k:jax.device_put(v) for k,v in f.items() if k!="parent"}
    return p,h,spec,chosen


def readout(a):
    folder=a.out/f"{a.core}-{a.seed}"
    if (folder/"complete.json").exists():return
    p,h,spec,chosen=load_head(a)
    @jax.jit
    def encoder(b):return policy_fields(p,h,spec,b,False,False)[:,:,-1]
    def xy(role):
        rows=data(a.root/"reacher"/role);zs=[];states=[]
        for j in range(0,len(rows["seed"]),32):
            ix=np.arange(j,min(j+32,len(rows["seed"])));b=batch(rows,ix)
            zs.append(np.asarray(encoder(b)));states.append(b["n"][:,:,-1,4:10])
        return np.concatenate(zs).reshape(-1,128),np.concatenate(states).reshape(-1,6)
    x,y=xy("train")
    model=train_model(folder,(128,192,128,6),x,y,np.ones_like(y),1200,a.seed)
    x,y=xy("dev");v=np.asarray(mlp(model,x));rmse=((v-y)**2).mean(0)**.5
    write(folder/"complete.json",{"weights":sha_file(folder/"weights.npz"),"optimizer":sha_file(folder/"optimizer.npz"),
         "sft":chosen["sha256"],"updates":1200,"dev_state_rmse":rmse.tolist()})


def sealed(a):
    tags=["transition"]+[f"{c}-{s}" for s in (711,712) for c in ("hybrid","gpt2")]
    hashes={}
    for tag in tags:
        root=a.out/tag;d=json.loads((root/"complete.json").read_bytes());hashes[tag]=sha_file(root/"weights.npz")
        if d["weights"]!=hashes[tag]:raise ValueError("Planning model changed")
    lock(a.out/"lock.json",{"weights":hashes,"first_start":890000,"paired_starts":64,"decisions":24})
    return hashes


@jax.jit
def score(model,state,goal,bank,future,horizon):
    state=jnp.concatenate((unit(state[:,:2],jnp),unit(state[:,2:4],jnp),state[:,4:].clip(-4,4)),-1)
    offered=jnp.broadcast_to(state[:,None],(*bank.shape[:2],6))
    next_x,tip=transition(model,offered,bank)
    now=jnp.linalg.norm(tip-goal[:,None],axis=-1)
    next_input=jnp.broadcast_to(next_x[:,:,None],(*bank.shape[:2],9,6))
    future_bank=jnp.broadcast_to(future[:,None],(*bank.shape[:2],9,2))
    _,tip2=transition(model,next_input,future_bank)
    later=jnp.linalg.norm(tip2-goal[:,None,None],axis=-1).min(-1)
    return jnp.where(horizon==1,now,.2*now+later).argmin(-1)


def closed(a):
    sys.path.append(str(a.sim_site))
    tag=f"{a.core}-{a.seed}";out=a.out/tag
    if (out/"test.json").exists():return
    hashes=sealed(a);p,h,spec,_=load_head(a);p=adjusted(p,h,spec)
    phys=cached(a.out/"transition/weights.npz");probe=cached(out/"weights.npz")
    prior=cached(a.parents.parent/"g7/reacher-prior/prior.npz")
    inference=stream(spec);probe_fn=jax.jit(mlp)
    modes=("latent-one","latent-two","latent-two-cue-held","raw-one-cue-held","raw-two-cue-held")
    results={}
    for mode in modes:
        final=[];ever=[]
        for start in range(890000,890064,16):
            seeds=np.repeat(np.arange(start,start+16),2);envs=[environment() for _ in seeds];goals=[];controls=[]
            for i,(e,s) in enumerate(zip(envs,seeds)):
                rng,g=setup(e,int(s));goals.append(g[i%2]);controls.append(rng.uniform(-.8,.8,2).astype(np.float32))
            goals=np.array(goals);memory,decision=zeros(spec,32);prev=None;history=[]
            for t in range(24):
                obs=[advance(e,c,True) for e,c in zip(envs,controls)]
                n,v=[np.stack(x) for x in zip(*obs)]
                if t==0:n[:,:4,12:14]=goals[:,None]/.21;n[:,:4,14]=1;prev=v[:,0].astype(np.float32)
                else:history.append(np.linalg.norm(n[:,-1,:2]*.21-goals,axis=-1))
                bank=np.stack([actions(np.random.default_rng([int(s),t,8119])) for s in seeds])
                later=np.stack([actions(np.random.default_rng([int(s),t+1,8119])) for s in seeds])
                if mode.startswith("raw"):
                    state=n[:,-1,4:10];goal=goals
                else:
                    _,context,_,memory,decision=inference(p,h,prior,n,v,bank,memory,decision,prev,np.int32(t*H))
                    state=probe_fn(probe,context[:,96:224]);goal=goals if "cue-held" in mode else context[:,-2:]*.21
                prev=v[:,-1].astype(np.float32)
                pick=np.asarray(score(phys,state,goal,bank,later,1 if "one" in mode or t==23 else 2));controls=bank[np.arange(len(bank)),pick]
            tip=np.stack([advance(e,u,False) for e,u in zip(envs,controls)]);dist=np.linalg.norm(tip-goals,axis=-1)
            history.append(dist);final.extend(dist.reshape(-1,2).tolist());ever.extend(np.min(history,axis=0).reshape(-1,2).tolist())
            for e in envs:e.close()
        final=np.array(final);ever=np.array(ever);success=(final<.03).mean(1)
        results[mode]={"success":float(success.mean()),"by_start":success.tolist(),"final_cm":float(final.mean()*100),"ever_success":float((ever<.03).mean())}
    sample=np.random.default_rng(77133).integers(64,size=(4000,64))
    def difference(first,other):
        diff=np.array(results[first]["by_start"])-results[other]["by_start"]
        return {"change":float(diff.mean()),"ci95":np.quantile(diff[sample].mean(1),[.025,.975]).tolist()}
    write(out/"test.json",{"results":results,"latent_horizon_gain":difference("latent-two","latent-one"),
          "raw_horizon_gain":difference("raw-two-cue-held","raw-one-cue-held"),
          "explicit_cue_gain":difference("latent-two-cue-held","latent-two"),"locked_weights":hashes,"distinct_scenes":64,
          "goals_per_scene":2,"decisions":24,"future_proposals":"known exogenous controls; no future observation supplied"})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=("transition","readout","lock","test"))
    for k in ("root","out","parents","sim-site"):p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--core",choices=("hybrid","gpt2"),required=True);p.add_argument("--seed",type=int,required=True)
    a=p.parse_args();{"transition":train_transition,"readout":readout,"lock":sealed,"test":closed}[a.phase](a)
