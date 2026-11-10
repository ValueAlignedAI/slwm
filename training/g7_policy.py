"""Matched goal-conditioned action ranking on four consecutive decisions with a genuinely hidden earlier cue."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from models.g4_shared import Spec
from models.g5_shared import encode
from models.g5_residual import objective as foundation_loss
from training.g4_train import write,sha_file
from training.g5_batches import JointData
from training.g7_reacher import H,C,K

FIELD=("n","v","actions","end","goals","seed")


def data(folder):
    return {k:np.load(folder/(k+".npy"),mmap_mode="r") for k in FIELD}


def batch(rows,ix,condition="correct"):
    n=np.repeat(rows["n"][ix],2,axis=0).copy();v=np.repeat(rows["v"][ix],2,axis=0).copy()
    goal=rows["goals"][ix].reshape(-1,2).astype(np.float32)/.21
    bank=np.repeat(rows["actions"][ix],2,axis=0)
    y=np.repeat(rows["end"][ix],2,axis=0)/.21
    cue=goal.reshape(-1,2,2)[:,::-1].reshape(-1,2) if condition=="shuffled" else goal
    if condition!="erased":
        n[:,0,:4,12:14]=cue[:,None];n[:,0,:4,14]=1
    if condition=="oracle":
        n[:,:,:,12:14]=goal[:,None,None];n[:,:,:,14]=1
    if condition=="wrong_video":
        # Only the camera is substituted. Both goals from a single scene share its donor.
        vv=np.roll(rows["v"][ix],1,axis=0);v=np.repeat(vv,2,axis=0)
    if condition=="erased_video":v.fill(128)
    return {"n":n,"v":v,"goal":goal,"bank":bank,"end":y}


def base_weights(root,core,seed):
    name=f"g6-v5e-anchor-{core}-{seed-200}"; stage=root/name/"step-008192"
    cfg=json.loads((stage/"config.json").read_bytes());rec=json.loads((stage/"verified.json").read_bytes())
    if sha_file(stage/"inference.npz")!=rec["sha256"]["inference.npz"]:raise ValueError("Unverified policy parent")
    with np.load(stage/"inference.npz") as f:
        p={k:jax.device_put(f[k]) for k in f if k!="stats" and not k.startswith("prior_")}
        cal={k[6:]:jax.device_put(f[k]) for k in f if k.startswith("prior_")}; stats=np.array(f["stats"])
    return Spec(**cfg["spec"]),p,cal,stats,rec["sha256"]["inference.npz"]


def init_head(spec,seed,adapt):
    rng=np.random.default_rng(seed)
    def w(m,n):return jnp.asarray(rng.normal(0,m**-.5,(m,n)).astype(np.float32))
    h={"input":w(256,96),"gate":w(192,96),"proposal":w(192,96),"goal":w(96,2),"common":w(224,2),
       "response":w(224,10)*.1,"value":w(96,1)*.01,"bias":jnp.zeros(96),"choice":w(96,5)*.02}
    if adapt:
        for i in (spec.depth-2,spec.depth-1):
            h[f"lora{i}a"]=w(spec.width,8)*.15;h[f"lora{i}b"]=jnp.zeros((8,spec.width))
    return h


def adjusted(p,h,spec):
    if f"lora{spec.depth-1}a" not in h:return p
    w=dict(p)
    for i in (spec.depth-2,spec.depth-1):
        w[f"b{i}_out"]=p[f"b{i}_out"]+h[f"lora{i}a"]@h[f"lora{i}b"]
    return w


def obs_batch(n,v):
    b,t=n.shape[:2]
    f=v.astype(jnp.float32); f=jnp.clip(f+.65*(f-jnp.concatenate((f[:,:1],f[:,:-1]),1)),0,255)
    return {"n":n,"v":f,"a":jnp.zeros((b,t,100),jnp.uint8),"u":jnp.zeros((b,t,2),jnp.float32),
            "input_x":jnp.zeros((b,t),jnp.int32),"ok":jnp.ones((b,t),jnp.uint8),"sense":jnp.ones((b,t),jnp.uint8)}


def policy_fields(p,h,spec,b,raw,reset):
    n=b["n"];v=b["v"];m,c,t=n.shape[:3]
    if raw:
        coarse=v.reshape(m,c,t,4,4,4,4).mean((4,6)).reshape(m,c,t,16)/127.5-1
        return jnp.pad(jnp.concatenate([n,coarse],-1),((0,0),(0,0),(0,0),(0,95)))
    if reset: n,v=n.reshape(m*c,t,17),v.reshape(m*c,t,256)
    else: n,v=n.reshape(m,c*t,17),v.reshape(m,c*t,256)
    src=jnp.broadcast_to(jnp.array([0,1,1,0],jnp.uint8),(len(n),4))
    z,_=encode(adjusted(p,h,spec),spec,obs_batch(n,v),src)
    return z.reshape(m,c,t,128).astype(jnp.float32)


def update_memory(h,z,previous):
    x=jnp.tanh(jnp.concatenate([z.mean(-2),z[...,-1,:]],-1)@h["input"]+h["bias"])
    both=jnp.concatenate([x,previous],-1);keep=jax.nn.sigmoid(both@h["gate"])
    return keep*previous+(1-keep)*jnp.tanh(both@h["proposal"])


def choose(h,final,mem,bank,physical=None):
    context=jnp.concatenate((final,mem),-1)
    phi=jnp.stack((bank[...,0],bank[...,1],bank[...,0]**2,bank[...,1]**2,bank[...,0]*bank[...,1]),-1)
    end=(context@h["common"])[...,None,:]+jnp.einsum("...kf,...fd->...kd",phi,(context@h["response"]).reshape(*context.shape[:-1],5,2))
    if physical is not None:end=physical+.1*end
    goal=mem@h["goal"]
    rank=-jnp.sqrt(jnp.sum((end-goal[...,None,:])**2,-1)+1e-5)+.04*jnp.einsum("...kf,...f->...k",phi,mem@h["choice"])
    return rank,end,goal,(mem@h["value"])[...,0]


def outputs(p,h,spec,b,raw,reset=False,prior=None):
    z=policy_fields(p,h,spec,b,raw,reset); prev=jnp.zeros((len(z),96),jnp.float32)
    if prior is not None:
        from training.g7_reacher_prior import infer
        physical=infer(prior,b["n"],b["bank"])
    else:physical=None
    outputs=[]
    for i in range(z.shape[1]):
        prev=update_memory(h,z[:,i],jnp.zeros_like(prev) if reset else prev)
        outputs.append(choose(h,z[:,i,-1],prev,b["bank"][:,i],None if physical is None else physical[:,i]))
    return tuple(jnp.stack(parts,axis=1) for parts in zip(*outputs))


def objective(p,h,spec,b,raw,prior=None):
    score,pos,goal,_=outputs(p,h,spec,b,raw,prior=prior); true=b["end"]; target=b["goal"]
    costs=jnp.linalg.norm(true-target[:,None,None],axis=-1)
    rank=-(jax.nn.softmax(-costs/.14,-1)*jax.nn.log_softmax(score/.14,-1)).sum(-1).mean()
    delta=(pos-pos.mean(-2,keepdims=True))-(true-true.mean(-2,keepdims=True))
    loss=rank+1.5*jnp.mean((pos-true)**2)+5*jnp.mean(delta**2)+2*jnp.mean((goal-target[:,None])**2)
    return loss,jnp.stack((loss,rank,jnp.mean((goal-target[:,None])**2)))


def save_weights(out,h,**meta):
    with out.with_suffix(".part").open("wb") as f:np.savez(f,**jax.device_get(h),**meta)
    out.with_suffix(".part").replace(out)


def evaluate(p,h,spec,rows,raw,prior=None):
    @jax.jit
    def field(b):return outputs(p,h,spec,b,raw,prior=prior)
    @jax.jit
    def reset(b):return outputs(p,h,spec,b,raw,True,prior=prior)
    results={}
    costs_all=np.linalg.norm(rows["end"][:,:,None]-rows["goals"][:,None,:,None],axis=-1)
    # The same observation/action histories are paired with opposite true goals.
    assert np.allclose(rows["goals"][:,0],-rows["goals"][:,1])
    for cond in ("correct","reset","shuffled","erased","oracle","wrong_video","erased_video"):
        regrets=[];correct=[];preds=[]; errors=[];truegoals=[]
        for j in range(0,len(rows["seed"]),32):
            ix=np.arange(j,min(j+32,len(rows["seed"]))); b=batch(rows,ix,cond)
            scores,_,gh,_=map(np.asarray,(reset if cond=="reset" else field)(b))
            cost=np.linalg.norm(b["end"]-b["goal"][:,None,None],axis=-1)*21
            # Report cycles after the cue vanished.
            choice=scores[:,1:].argmax(-1); targets=cost[:,1:]
            best=targets.argmin(-1); r=np.take_along_axis(targets,choice[...,None],axis=-1)[...,0]-targets.min(-1)
            regrets.extend(r.reshape(len(ix),-1).mean(1));correct.extend((choice==best).reshape(len(ix),-1).mean(1))
            errors.extend(np.linalg.norm(gh[:,1:]-b["goal"][:,None],axis=-1).reshape(len(ix),-1).mean(1)*21)
        results[cond]={"regret_cm":float(np.mean(regrets)),"top1":float(np.mean(correct)),"goal_error_cm":float(np.mean(errors)),
                       "scene_regret":list(map(float,regrets))}
    measured=costs_all[:,1:]*100
    results["uniform"]={"regret_cm":float((measured.mean(-1)-measured.min(-1)).mean())}
    results["passive"]={"regret_cm":float((measured[...,-1]-measured.min(-1)).mean())}
    delta=np.array(results["reset"]["scene_regret"])-results["correct"]["scene_regret"]
    shuffled=np.array(results["shuffled"]["scene_regret"])-results["correct"]["scene_regret"]
    ix=np.random.default_rng(771).integers(0,len(delta),(2000,len(delta)))
    ci=lambda x:np.quantile(x[ix].mean(1),[.025,.975]).tolist()
    results["gate"]={"passed":bool(results["correct"]["regret_cm"]<results["uniform"]["regret_cm"] and ci(delta)[0]>0 and ci(shuffled)[0]>0),
                     "reset_minus_carried_cm_ci95":ci(delta),"shuffled_minus_correct_cm_ci95":ci(shuffled)}
    return results


def fit(a):
    folder=a.out/f"{a.core}-{a.mode}-{a.seed}";folder.mkdir(exist_ok=True,parents=True)
    if (folder/"dev.json").exists():return
    raw=a.mode=="raw"; adapted=a.mode=="adapter"
    spec,p,cal,stats,parent=base_weights(a.parents,a.core,a.seed)
    train=data(a.data/"reacher/train"); dev=data(a.data/"reacher/dev")
    prior=None
    if a.prior is not None:
        with np.load(a.prior) as f:prior={k:jax.device_put(f[k]) for k in f.files}
    h=init_head(spec,a.seed,adapted); opt=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(.001,weight_decay=.001));ost=opt.init(h)
    replay=JointData(a.replay,"train",verify=False,stats=stats);held=JointData(a.replay,"dev",verify=False,stats=stats)
    @jax.jit
    def learn(p,h,ost,b,ref):
        def loss(h):
            v,m=objective(p,h,spec,b,raw,prior)
            if adapted:
                w=adjusted(p,h,spec); src=ref["source"]
                base,_=encode(p,spec,ref,src);new,_=encode(w,spec,ref,src)
                dist=jnp.mean((base.astype(jnp.float32)-new.astype(jnp.float32))**2)
                supervised,_=foundation_loss(w,spec,ref,cal)
                v=v+.20*dist+.025*supervised
            return v,m
        (_,m),gr=jax.value_and_grad(loss,has_aux=True)(h);ud,st=opt.update(gr,ost,h)
        return optax.apply_updates(h,ud),st,m
    @jax.jit
    def ref_loss(p,h,b):return foundation_loss(adjusted(p,h,spec),spec,b,cal)[1]
    before=np.mean([np.asarray(ref_loss(p,h,held.requests(j,32))) for j in range(12)],axis=0)
    first=0; ck=folder/"optimizer.npz"
    if ck.exists():
        with np.load(ck) as f:
            if str(f["parent"])!=parent:raise ValueError("Policy parent changed")
            h={k:jax.device_put(f["w-"+k]) for k in h}; tree=jax.tree.structure(ost);ost=jax.tree.unflatten(tree,[jax.device_put(f["o-"+str(i)]) for i in range(tree.num_leaves)]);first=int(f["step"])
    for step in range(first,a.steps):
        rng=np.random.default_rng([721,step]);ix=rng.integers(len(train["seed"]),size=24)
        b=batch(train,ix); ref=replay.requests(step,32)
        # Keep replay physically observed and stratified; both supervised alternatives see the same policy rows.
        h,ost,m=learn(p,h,ost,b,ref)
        if (step+1)%128==0 or step+1==a.steps:
            metric=np.asarray(m)
            if not np.isfinite(metric).all():raise FloatingPointError(metric)
            values={"w-"+k:v for k,v in jax.device_get(h).items()}
            values.update({"o-"+str(j):v for j,v in enumerate(jax.tree.leaves(jax.device_get(ost)))})
            with ck.with_suffix(".part").open("wb") as f:np.savez(f,**values,step=step+1,parent=parent)
            ck.with_suffix(".part").replace(ck);print(a.core,a.mode,a.seed,step+1,metric,flush=True)
    save_weights(folder/"weights.npz",h,parent=np.array(parent))
    after=np.mean([np.asarray(ref_loss(p,h,held.requests(j,32))) for j in range(12)],axis=0)
    result=evaluate(p,h,spec,dev,raw,prior)
    write(folder/"dev.json",{"core":a.core,"mode":a.mode,"seed":a.seed,"steps":a.steps,"parent":parent,
        "prior_sha256":sha_file(a.prior) if a.prior else None,"params":sum(v.size for v in h.values()),"foundation_held_before":before.tolist(),"foundation_held_after":after.tolist(),
        "weights_sha256":sha_file(folder/"weights.npz"),"optimizer_sha256":sha_file(ck),"results":result})
    print("Policy gate",a.core,a.mode,a.seed,result["gate"],flush=True)


if __name__=="__main__":
    q=argparse.ArgumentParser();q.add_argument("--core",required=True,choices=("hybrid","gpt2"));q.add_argument("--mode",required=True,choices=("raw","frozen","adapter"))
    q.add_argument("--seed",required=True,type=int);q.add_argument("--steps",type=int,default=1024)
    for k in ("out","data","replay","parents"):q.add_argument("--"+k,type=Path,required=True)
    q.add_argument("--prior",type=Path);fit(q.parse_args())
