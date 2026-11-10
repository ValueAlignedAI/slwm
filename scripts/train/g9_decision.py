"""Label learner-visited or random trajectories and tune a small decision head on frozen fields."""
import argparse
import json
from functools import lru_cache
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from evals.g9_live import initial_memory,live
from models.g9_shared import Spec,encode,decision_scores as action_scores
from models.g9_tabletop import ACTIONS,advance,camera,expert,initial,observation,target
from scripts.train.g9_fit import numpy_weights,save_json,weights

TRAIN_SEED,TEST_SEED=322121,631120
N=128


def initial_head(field,seed):
    r=np.random.default_rng(seed)
    return dict(w=jnp.asarray(r.normal(0,field**-.5,(field,128)).astype("f4")),b=jnp.zeros(128),
                a=jnp.zeros((128,ACTIONS)),c=jnp.zeros(ACTIONS))


@lru_cache(maxsize=16)
def trajectory(spec,behavior,evaluation):
    @jax.jit
    def run(p,head):
        steps=512 if evaluation else 128
        seed=TEST_SEED if evaluation else TRAIN_SEED
        state=initial(jnp.arange(N),seed)
        board=camera(state["stations"],state["tint"]);first=state["goal"];keep=jnp.ones((N,4),bool)
        exploratory=jax.random.randint(jax.random.PRNGKey(TRAIN_SEED+79),(steps,N),0,ACTIONS)
        def tick(c,t):
            st,mem=c
            st={**st,"goal":(first+t//(64 if evaluation else 32))%3}
            obs=observation(st,board,t)
            z,mem=encode(p,spec,{k:v[:,None] for k,v in obs.items()},keep,memory=mem,offset=t)
            z=z[:,0];label=expert(st)
            if behavior=="random":act=exploratory[t]
            elif behavior=="parent":act=(z@p["read"][:,-ACTIONS:].astype(z.dtype)).argmax(-1)
            else:act=action_scores(z,p["read"][:,-ACTIONS:],head).argmax(-1)
            nxt=advance(st,act,t,stall=evaluation)
            reached=(nxt["pos"]==target(st)).all(1)
            record=jnp.stack([reached,act==0,(act==0)&~reached,act>=5,act],-1) if evaluation else (z,label)
            return (nxt,mem),record
        return jax.lax.scan(tick,(state,initial_memory(spec,N)),jnp.arange(steps))[1]
    return run


@jax.jit
def fit(z,y,read,init,seed):
    z=z.reshape(-1,z.shape[-1]);y=y.reshape(-1)
    # Identical 384 updates and class weighting in both acquisition treatments.
    count=jnp.bincount(y,length=ACTIONS)
    importance=jnp.clip(jnp.sqrt(len(y)/(ACTIONS*jnp.maximum(count,1))),.4,4)
    def step(c,t):
        head,m,v=c
        ix=jax.random.randint(jax.random.fold_in(jax.random.PRNGKey(seed),t),(256,),0,len(y))
        def loss(h):
            logits=action_scores(z[ix],read,h)
            ce=-jnp.take_along_axis(jax.nn.log_softmax(logits),y[ix,None],-1)[:,0]
            return (ce*importance[y[ix]]).mean()
        val,grad=jax.value_and_grad(loss)(head)
        m=jax.tree.map(lambda b,g:.9*b+.1*g,m,grad);v=jax.tree.map(lambda a,g:.99*a+.01*g*g,v,grad)
        n=t+1
        head=jax.tree.map(lambda w,a,b:w-.002*a/(1-.9**n)/(jnp.sqrt(b/(1-.99**n))+1e-6),head,m,v)
        return (head,m,v),val
    zero=jax.tree.map(jnp.zeros_like,init)
    (head,_,_),loss=jax.lax.scan(step,(init,zero,zero),jnp.arange(384))
    return head,count,loss[-1]


def summary(values):
    r=np.asarray(values);cols=("occupancy","hold","false_hold","probe")
    doc={k:r[:,:,i].mean(0).tolist() for i,k in enumerate(cols)}
    doc["success"]=r[:,:,0].reshape(8,64,-1).any(1).mean(0).tolist()
    doc["example_actions"]=r[:,0,4].tolist();return doc


def arm(stage,out):
    meta=json.loads((stage/"receipt.json").read_text());spec=Spec(**meta["spec"])
    dest=out/meta["tag"];dest.mkdir(parents=True,exist_ok=True);file=dest/"posttraining.json"
    if file.exists():return
    if spec.field!=512 or spec.act!="gelu" or meta["step"]!=4096:raise ValueError("Wrong frozen parent")
    p=jax.tree.map(jnp.asarray,numpy_weights(stage/"weights.npz"));h=initial_head(spec.field,meta["seed"]+54)
    result=dict(parent_sha256=meta["inference_sha256"],tag=meta["tag"],head_parameters=sum(v.size for v in h.values()),training_missions=N,
                labels_per_arm=N*128,held_missions=N,held_seconds=51.2,train_seed=TRAIN_SEED,test_seed=TEST_SEED,arms={})
    result["parent"]=summary(trajectory(spec,"parent",True)(p,h))
    for mode in ("random","parent"):
        z,y=trajectory(spec,mode,False)(p,h)
        head,count,loss=fit(z,y,p["read"][:,-ACTIONS:],h,jnp.int32(meta["seed"]+73))
        score=summary(trajectory(spec,"adapted",True)(p,head))
        path=dest/(mode+"-head.npz")
        score.update(sha256=weights(path,head),label_counts=np.asarray(count).tolist(),last_training_loss=float(loss))
        result["arms"][mode]=score
    for mode in ("raw","idle"):
        vals=live(None,spec,mode,seed=TEST_SEED)
        result[mode]=dict(occupancy=vals[:,:,0].mean(0).tolist(),success=vals[:,:,0].reshape(8,64,-1).any(1).mean(0).tolist())
    save_json(file,result)
    print(meta["tag"],{k:round(np.mean(v["occupancy"]),5) for k,v in result["arms"].items()},flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("runs",type=Path);ap.add_argument("out",type=Path)
    a=ap.parse_args()
    for seed in (912,911):
        for core in ("hybrid","gpt2"):
            arm(a.runs/f"confirm-{core}-z512-gelu-s{seed}-fresh"/"step-04096",a.out)
