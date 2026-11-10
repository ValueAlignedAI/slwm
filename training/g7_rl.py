"""On-policy PPO residual over a qualified recurrent supervised choice head, with true simulator rewards."""
import argparse
import json
import os
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
from models.g5_shared import encode
from training.g4_train import write,sha_file
from training.g7_policy import base_weights,adjusted,obs_batch,update_memory,choose
from training.g7_reacher import H,K,environment,setup,advance,actions
from training.g7_reacher_prior import infer


def zeros(spec,b):
    dtype=getattr(jnp,spec.dtype);mem=[]
    for i in range(spec.depth):
        if spec.core=="gpt2" or (i+1)%4==0:
            shape=(b,64,spec.width//64,64);mem.append((jnp.zeros(shape,dtype),jnp.zeros(shape,dtype),jnp.zeros((b,64),bool)))
        else:mem.append(jnp.zeros((b,8,spec.width//8),jnp.float32))
    return tuple(mem),jnp.zeros((b,96),jnp.float32)


def stream(spec):
    @jax.jit
    def step(p,h,prior,n,v,bank,mem,decision,prev_v,offset):
        b=obs_batch(n,v);b["v"]=b["v"].at[:,0].set(jnp.clip(v[:,0].astype(jnp.float32)+.65*(v[:,0].astype(jnp.float32)-prev_v),0,255))
        src=jnp.broadcast_to(jnp.array([0,1,1,0],jnp.uint8),(len(n),4))
        z,new=encode(p,spec,b,src,memory=mem,offset=offset)
        z=z.astype(jnp.float32); m=update_memory(h,z,decision)
        prior_y=infer(prior,n,bank)
        score,forecast,goal,_=choose(h,z[:,-1],m,bank,prior_y)
        context=jnp.concatenate([m,z[:,-1],goal],axis=-1)
        phi=jnp.concatenate((bank,forecast,forecast-goal[:,None],score[...,None]),axis=-1)
        return score/.105,context,phi,new,m
    return step


def actor_init(seed):
    rng=np.random.default_rng(seed)
    w=lambda m,n:jnp.asarray(rng.standard_normal((m,n),dtype=np.float32)/np.sqrt(m),dtype=jnp.float32)
    return {"state":w(226,64),"action":w(7,64),"out":jnp.zeros((64,1)),"value":w(64,1)*.01,"bias":jnp.zeros(64)}


def actor(q,base,context,phi):
    state=jnp.tanh(context@q["state"]+q["bias"])
    candidate=jnp.tanh(phi@q["action"]+state[:,None])*jax.nn.sigmoid(phi@q["action"])
    logits=base+4*jnp.tanh((candidate@q["out"])[...,0])
    return logits,(state@q["value"])[...,0]


def roll(spec,p,h,prior,q,seeds,cycles,mode,stochastic,noise_seed,step_fn,actor_fn):
    unique=len(seeds); count=2*unique
    envs=[environment() for _ in range(count)]
    arm=[];goals=[]; initial=[]; last=[]
    for i,env in enumerate(envs):
        seed=int(seeds[i//2]);rng,gs=setup(env,seed)
        arm.append(rng.uniform(-.8,.8,2).astype(np.float32));goals.append(gs[i%2])
    goals=np.asarray(goals); arm=np.stack(arm); memory,decision=zeros(spec,count)
    storage={k:[] for k in ("context","phi","base","action","oldlog","oldvalue","reward")}
    dist=[]; sample=np.random.default_rng(noise_seed); prev=None; goalest=[]
    for t in range(cycles):
        obs=[advance(e,u,True) for e,u in zip(envs,arm)]
        n,v=[np.stack(x) for x in zip(*obs)]
        pos=n[:,-1,:2]*.21; current=np.linalg.norm(pos-goals,axis=-1)
        if t==0:
            initial=current.copy();prev=v[:,0].astype(np.float32)
        else:dist.append(current)
        cue=goals.reshape(unique,2,2)[:,::-1].reshape(-1,2) if mode=="shuffled" else goals
        if t==0:
            n[:,:4,12:14]=cue[:,None]/.21;n[:,:4,14]=1
        if mode=="oracle":
            n[:,:,12:14]=goals[:,None]/.21;n[:,:,14]=1
        bank=np.stack([actions(np.random.default_rng([int(s),t,8119])) for s in np.repeat(seeds,2)])
        if mode in ("reset","rl-reset"):memory,decision=zeros(spec,count);prev=v[:,0].astype(np.float32)
        score,ctx,phi,memory,decision=step_fn(p,h,prior,n,v,bank,memory,decision,prev,np.int32(0 if mode in ("reset","rl-reset") else t*H))
        prev=v[:,-1].astype(np.float32); lg,val=actor_fn(q,score,ctx,phi)
        effective=lg if mode in ("rl","rl-reset") else score
        probs=np.asarray(jax.nn.softmax(effective,-1)); probs=probs/probs.sum(-1,keepdims=True)
        if mode=="passive":choice=np.full(count,8,int)
        elif stochastic:choice=np.array([sample.choice(K,p=x.astype(float)/x.sum(dtype=float)) for x in probs])
        else:choice=probs.argmax(1)
        arm=bank[np.arange(count),choice]
        for k,value in (("context",ctx),("phi",phi),("base",score),("action",choice),("oldlog",np.log(np.maximum(probs[np.arange(count),choice],1e-9))),("oldvalue",val)):
            storage[k].append(np.asarray(value))
        if t:
            before=initial if t==1 else dist[-2]
            reward=(before-current)/.21+.35*(current<.03)-.008*(last**2).sum(1)
            storage["reward"].append(reward)
        last=arm.copy()
    final=np.stack([advance(e,u,False) for e,u in zip(envs,arm)]);remaining=np.linalg.norm(final-goals,axis=-1)
    before=dist[-1] if dist else initial
    storage["reward"].append((before-remaining)/.21+.35*(remaining<.03)-.008*(arm**2).sum(1));dist.append(remaining)
    dt=envs[0].dt
    for e in envs:e.close()
    data={k:np.asarray(v) for k,v in storage.items()}
    return data,{"final_cm":(remaining*100).reshape(unique,2).mean(1).tolist(),
                 "final_success":(remaining<.03).reshape(unique,2).mean(1).tolist(),
                 "ever_success":(np.stack(dist).min(0)<.03).reshape(unique,2).mean(1).tolist(),
                 "mean_initial_cm":float(initial.mean()*100),"simulated_seconds":float(cycles*H*dt)}


def ppo(q,optstate,opt):
    @jax.jit
    def update(q,o,b):
        def loss(params):
            logits,value=actor(params,b["base"],b["context"],b["phi"]);lp=jax.nn.log_softmax(logits,-1);prob=jax.nn.softmax(logits,-1)
            now=jnp.take_along_axis(lp,b["action"][:,None],axis=1)[:,0];ratio=jnp.exp(now-b["oldlog"])
            gain=jnp.minimum(ratio*b["adv"],jnp.clip(ratio,.8,1.2)*b["adv"])
            kl=jnp.sum(prob*(lp-jax.nn.log_softmax(b["base"],-1)),-1)
            return -gain.mean()+.45*jnp.mean((value-b["return"])**2)+.018*kl.mean()+.008*jnp.mean(jnp.sum(prob*lp,-1))
        v,g=jax.value_and_grad(loss)(q);upd,o=opt.update(g,o,q);return optax.apply_updates(q,upd),o,v
    return update


def save(path,q):
    with path.with_suffix(".part").open("wb") as f:np.savez(f,**jax.device_get(q))
    path.with_suffix(".part").replace(path)


def aggregate(rows):return {k:float(np.mean(v)) for k,v in rows.items()}


def fit(a):
    sys.path.append(str(a.sim_site))
    dest=a.out/f"{a.core}-{a.seed}";dest.mkdir(exist_ok=True,parents=True)
    lock=json.loads(a.lock.read_bytes());chosen=lock["models"][f"{a.core}-{a.seed}"]
    parent=a.sft/f"{a.core}-{chosen['mode']}-{a.seed}"; hfile=parent/"weights.npz"
    if sha_file(hfile)!=chosen["sha256"] or not chosen["gate"]:raise ValueError("Supervised gate is not satisfied")
    if (dest/"test.json").is_file():return
    spec,p,_,_,base=base_weights(a.parents,a.core,a.seed)
    with np.load(hfile) as f:h={k:jax.device_put(f[k]) for k in f.files if k!="parent"}
    p=adjusted(p,h,spec)
    with np.load(a.prior) as f:prior={k:jax.device_put(f[k]) for k in f.files}
    step=stream(spec);action=jax.jit(actor); q=actor_init(a.seed);opt=optax.chain(optax.clip_by_global_norm(.6),optax.adam(.0007));o=opt.init(q);upd=ppo(q,o,opt)
    seen=0;ck=dest/"resume.npz";best=(-1.,-1e6);choices=[]
    if ck.exists():
        with np.load(ck) as f:
            if str(f["sft"])!=chosen["sha256"]:raise ValueError("Different SFT parent")
            q={k:jax.device_put(f["p"+k]) for k in q};tree=jax.tree.structure(o);o=jax.tree.unflatten(tree,[jax.device_put(f[f"o{i}"]) for i in range(tree.num_leaves)])
            seen=int(f["batch"]);best=(float(f["best_success"]),float(f["best_distance"]))
        if (dest/"selection.json").is_file():choices=json.loads((dest/"selection.json").read_bytes())
    for it in range(seen,a.iterations):
        seeds=np.arange(850000+it*12,850000+(it+1)*12)
        b,result=roll(spec,p,h,prior,q,seeds,12,"rl",True,9170+it+a.seed,step,action)
        value=b["oldvalue"];reward=b["reward"];adv=np.zeros_like(value);carry=np.zeros_like(value[0])
        for t in range(len(value)-1,-1,-1):
            nv=value[t+1] if t+1<len(value) else 0.;carry=reward[t]+.97*nv-value[t]+.97*.95*carry;adv[t]=carry
        b["return"]=adv+value;b["adv"]=(adv-adv.mean())/max(adv.std(),1e-4)
        flat={k:v.reshape(-1,*v.shape[2:]) for k,v in b.items() if k!="reward"};rng=np.random.default_rng([7711,it])
        for epoch in range(4):
            for ids in np.array_split(rng.permutation(len(flat["action"])),4):
                q,o,val=upd(q,o,{k:jax.device_put(v[ids]) for k,v in flat.items()})
        if not np.isfinite(float(val)):raise FloatingPointError(it)
        if (it+1)%8==0:
            _,r=roll(spec,p,h,prior,q,np.arange(855000,855016),24,"rl",False,41,step,action)
            stats=aggregate(r);candidate=(stats["final_success"],-stats["final_cm"])
            if candidate>best:best=candidate;save(dest/"best.npz",q)
            choices.append({"iteration":it+1,"metrics":stats});write(dest/"selection.json",choices)
        values={"p"+k:v for k,v in jax.device_get(q).items()};values.update({f"o{i}":v for i,v in enumerate(jax.tree.leaves(jax.device_get(o)))})
        with ck.with_suffix(".part").open("wb") as f:np.savez(f,**values,batch=it+1,sft=chosen["sha256"],best_success=best[0],best_distance=best[1])
        ck.with_suffix(".part").replace(ck)
        if (it+1)%4==0:print("PPO",a.core,a.seed,it+1,aggregate(result),flush=True)
    with np.load(dest/"best.npz") as f:q={k:jax.device_put(f[k]) for k in f.files}
    write(dest/"test-lock.json",{"policy":sha_file(dest/"best.npz"),"sft":chosen["sha256"],"start":860000,"count":64,"cycles":24})
    results={}
    for mode in ("rl","sft","reset","rl-reset","shuffled","passive","oracle"):
        chunks=[]
        for s in range(860000,860064,16):
            _,v=roll(spec,p,h,prior,q,np.arange(s,s+16),24,mode,False,31,step,action);chunks.append(v)
        vals={k:sum((r[k] for r in chunks),[]) for k in ("final_cm","final_success","ever_success")}
        results[mode]={"mean":aggregate(vals),"by_start":vals,"seconds":chunks[0]["simulated_seconds"]}
    rng=np.random.default_rng(71191);bi=rng.integers(64,size=(4000,64));effects={}
    for name in ("sft","passive","reset","rl-reset"):
        delta=np.array(results["rl"]["by_start"]["final_success"])-results[name]["by_start"]["final_success"]
        effects[name]={"success_gain":float(delta.mean()),"ci95":np.quantile(delta[bi].mean(1),[.025,.975]).tolist()}
    long={}
    for mode in ("rl","sft","reset","passive"):
        _,r=roll(spec,p,h,prior,q,np.arange(870000,870008),192,mode,False,19,step,action);long[mode]=aggregate(r)
    write(dest/"test.json",{"core":a.core,"seed":a.seed,"selected_sft":chosen,"real_training_decisions":a.iterations*24*12,"ppo_updates":a.iterations*16,
         "checkpoint_sha256":sha_file(dest/"best.npz"),"optimizer_sha256":sha_file(ck),"results":results,"success_contrasts":effects,
         "replication_gate_this_seed":all(v["ci95"][0]>0 for v in effects.values()),"long_192_cycle_diagnostic":long,
         "forgetting_control":"SFT world and shared adapters frozen during policy-gradient updates."})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--core",choices=("hybrid","gpt2"),required=True);p.add_argument("--seed",type=int,required=True);p.add_argument("--iterations",type=int,default=40)
    for k in ("out","sft","parents","prior","lock","sim-site"):p.add_argument("--"+k,type=Path,required=True)
    fit(p.parse_args())
