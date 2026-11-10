"""Prospective video-only camera substitution and candidate-action ranking on a second environment."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax
from models.g5_shared import encode
from training.g4_train import write,sha_file
from training.g7_policy import base_weights,data
from scripts.train.g7_lock_heads import lock


def donor(n):
    walk=np.random.default_rng(771103).permutation(n);d=np.empty(n,int)
    d[walk]=np.roll(walk,-1)
    return d


def encoder(p,spec,raw):
    proj=np.random.default_rng(7752).normal(0,.065,(256,128)).astype(np.float32)
    @jax.jit
    def apply(video,actions,sense):
        m=len(video);v=video.reshape(m,32,256).astype(jnp.float32)
        delta=v-jnp.concatenate((v[:,:1],v[:,:-1]),1)
        v=jnp.clip(v+.65*delta,0,255)
        if raw:
            z=(v/127.5-1)@proj*sense[...,None]
        else:
            b={"v":v,"a":jnp.zeros((m,32,100),jnp.uint8),"n":jnp.zeros((m,32,17)),
               "input_x":jnp.zeros((m,32),jnp.int32),"ok":jnp.ones((m,32),jnp.uint8),"sense":sense,
               "u":actions.reshape(m,32,2)}
            src=jnp.tile(jnp.array([[0,1,0,0]],jnp.uint8),(m,1))
            z,_=encode(p,spec,b,src);z=z.astype(jnp.float32)
        z=z.reshape(m,4,8,128)
        return jnp.concatenate((z.mean(2),z[:,:,-1],actions.reshape(m,4,16)),axis=-1)
    return apply


def features(encode_fn,rows,mode):
    n=len(rows["seed"]);d=donor(n);out=[]
    for j in range(0,n,64):
        ids=np.arange(j,min(j+64,n));video=np.array(rows["v"][d[ids] if mode=="wrong" else ids])
        if mode=="delayed":
            flat=video.reshape(len(ids),32,256)
            video=np.concatenate((np.repeat(flat[:,:1],8,axis=1),flat[:,:-8]),1).reshape(video.shape)
        seen=np.ones((len(ids),32),np.float32)
        if mode=="erased":seen.fill(0)
        out.append(np.asarray(encode_fn(video,rows["n"][ids,:,:,10:12],seen)))
    return np.concatenate(out)


def initial(seed):
    rng=np.random.default_rng(seed)
    def w(m,n):return jnp.array(rng.normal(0,m**-.5,(m,n)),dtype=jnp.float32)
    return {"w":w(272,256),"b":jnp.zeros(256),"w2":w(256,192),"b2":jnp.zeros(192),"a":w(5,192),"y":w(192,2)*.1,"main":w(192,2)*.15}


def forecast(h,x,action):
    z=jax.nn.gelu(x@h["w"]+h["b"])
    context=jax.nn.gelu(z@h["w2"]+h["b2"])
    phi=jnp.concatenate((action,action**2,(action[...,0]*action[...,1])[...,None]),-1)
    joint=jnp.tanh(context[...,None,:]+phi@h["a"])*jax.nn.sigmoid(phi@h["a"])
    return (context@h["main"])[...,None,:]+joint@h["y"]


def fit(h,x,rows):
    opt=optax.adamw(.001,weight_decay=.001);state=opt.init(h)
    @jax.jit
    def update(h,state,x,action,y):
        def loss(p):
            error=forecast(p,x,action)-y
            return (error**2).mean()+4*((error-error.mean(-2,keepdims=True))**2).mean()
        value,g=jax.value_and_grad(loss)(h);change,state=opt.update(g,state,h)
        return optax.apply_updates(h,change),state,value
    for step in range(1024):
        ids=np.random.default_rng([71137,step]).integers(len(x),size=128)
        h,state,l=update(h,state,x[ids],rows["actions"][ids],rows["end"][ids]/.21)
    if not np.isfinite(float(l)):raise FloatingPointError("Camera training diverged")
    return h,jax.tree.leaves(state)


def analyze(rows,y):
    true=rows["end"][:,1:];goals=rows["goals"][:,None,:,None,:]
    cost=np.linalg.norm(true[:,:,None]-goals,axis=-1)*100
    score=np.linalg.norm(y[:,1:,None]*.21-goals,axis=-1)
    pick=score.argmin(-1);best=cost.min(-1)
    regret=(np.take_along_axis(cost,pick[...,None],axis=-1)[...,0]-best).mean((1,2))
    return regret,{"regret_cm":float(regret.mean()),"top1":float((pick==cost.argmin(-1)).mean()),
                   "endpoint_cm":float((np.linalg.norm(y[:,1:]*.21-true,axis=-1)*100).mean()),
                   "uniform_regret_cm":float((cost.mean(-1)-best).mean()),"passive_regret_cm":float((cost[...,-1]-best).mean())}


def main(a):
    out=a.out/f"{a.core}-{a.seed}";out.mkdir(exist_ok=True,parents=True)
    testlock=a.out/"lock.json"
    if a.phase=="lock":
        docs={}
        for s in (711,712):
            for c in ("hybrid","gpt2","raw"):
                f=a.out/f"{c}-{s}";d=json.loads((f/"dev.json").read_bytes())
                if d["weights"]!=sha_file(f/"weights.npz"):raise ValueError("Camera identity changed")
                docs[f"{c}-{s}"]=d["weights"]
        lock(testlock,docs);return
    if (out/("dev.json" if a.phase=="train" else "test.json")).is_file():return
    raw=a.core=="raw"
    if raw:p={};spec=None;parent="raw-identical-projection"
    else:spec,p,_,_,parent=base_weights(a.parents,a.core,a.seed)
    enc=encoder(p,spec,raw)
    if a.phase=="train":
        train=data(a.root/"reacher/train");x=features(enc,train,"correct")
        h,opt=fit(initial(a.seed),x,train)
        np.savez(out/"weights.npz",**jax.device_get(h))
        np.savez(out/"optimizer.npz",**{str(i):np.asarray(v) for i,v in enumerate(opt)})
        rows=data(a.root/"reacher/dev");z=features(enc,rows,"correct")
        _,scores=analyze(rows,np.asarray(jax.jit(forecast)(h,z,rows["actions"])))
        write(out/"dev.json",{"weights":sha_file(out/"weights.npz"),"optimizer":sha_file(out/"optimizer.npz"),
             "parent":parent,"updates":1024,"dev":scores});return
    if json.loads(testlock.read_bytes()).get(out.name)!=sha_file(out/"weights.npz"):raise ValueError("Camera head outside test lock")
    with np.load(out/"weights.npz") as f:h={k:jnp.asarray(v) for k,v in f.items()}
    rows=data(a.root/"reacher/test");per_scene={};metrics={}
    for mode in ("correct","delayed","wrong","erased"):
        z=features(enc,rows,mode)
        outcomes=np.asarray(jax.jit(forecast)(h,z,rows["actions"]))
        per_scene[mode],metrics[mode]=analyze(rows,outcomes)
    np.savez_compressed(out/"scenes.npz",**per_scene,scene=rows["seed"],donor=donor(len(rows["seed"])))
    write(out/"test.json",{"metrics":metrics,"weights":sha_file(out/"weights.npz"),"rows_sha256":sha_file(out/"scenes.npz"),
         "units":"centimetres per independent starting scene, two opposite goals, decisions 2-4","parent":parent})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=("train","lock","test"))
    p.add_argument("--core",choices=("hybrid","gpt2","raw"),required=True);p.add_argument("--seed",type=int,required=True)
    for k in ("out","root","parents"):p.add_argument("--"+k,type=Path,required=True)
    main(p.parse_args())
