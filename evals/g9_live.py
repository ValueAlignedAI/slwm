"""Paired, 51.2 s closed loops and sensor-identical action-history counterfactuals."""
from functools import lru_cache
import jax
import jax.numpy as jnp
import numpy as np
from models.g9_shared import encode
from models.g9_tabletop import ACTIONS, advance, camera, expert, initial, observation, target


def initial_memory(spec,batch,window=64):
    dtype=getattr(jnp,spec.dtype)
    def layer(i):
        if spec.core=="gated":return jnp.zeros((batch,spec.d),jnp.float32)
        if spec.core=="gpt2" or (i+1)%4==0:
            shape=(batch,window,spec.d//64,64)
            return jnp.zeros(shape,dtype),jnp.zeros(shape,dtype),jnp.zeros((batch,window),bool)
        return jnp.zeros((batch,8,spec.d//8),jnp.float32)
    return tuple(layer(i) for i in range(spec.layers))


def raw(obs,sign,seen):
    """Oracle receiver recovers its calibration solely from observed spectra, force and issued controls."""
    u=obs["u"];tap=u>=5;dir=jnp.where(u==5,1,-1)
    hz=jnp.argmax(obs["a"][:,:16],axis=1)
    reading=jnp.stack([jnp.where(hz>7,1,-1),jnp.where(obs["n"][:,2]>0,1,-1)],1)*dir[:,None]
    sign=jnp.where(tap[:,None],reading,sign);seen=seen|tap
    rgb=obs["v"].reshape(-1,16,16,3).astype(jnp.float32)
    colour=obs["w"][:,2].astype(jnp.int32)-1
    plane=jnp.take_along_axis(rgb,colour[:,None,None,None],axis=3)[...,0]
    index=plane.reshape(len(u),-1).argmax(1)
    dest=jnp.stack([(index%16)//4,(index//16)//4],axis=1)
    pos=jnp.rint((obs["n"][:,:2]+1)*1.5).astype(jnp.int32)
    diff=dest-pos;axis=(jnp.abs(diff[:,1])>jnp.abs(diff[:,0])).astype(jnp.int32)
    positive=(jnp.take_along_axis(diff*sign,axis[:,None],1)[:,0]>0)
    action=jnp.where((diff==0).all(1),0,jnp.where(seen,1+2*axis+(~positive),5))
    return action.astype(jnp.int32),sign,seen


@lru_cache(maxsize=100)
def _live_kernel(spec,mode,b,stall,seed,steps):
    def run(params):
        ids=jnp.arange(b)
        state=initial(ids,seed);image=camera(state["stations"],state["tint"])
        init_goal=state["goal"]
        window=512 if mode=="full" else 64
        persistent=mode in ("carry","wrong","full","block32")
        memory=initial_memory(spec,b,window) if persistent else ()
        keep=jnp.ones((b,4),bool)
        def step(c,tick):
            st,mem,estimate,seen=c
            st={**st,"goal":(init_goal+tick//64)%3}
            obs=observation(st,image,tick)
            raw_action,estimate,seen=raw(obs,estimate,seen)
            if mode in ("carry","reset","wrong","full","block32"):
                if mode=="wrong":obs={**obs,"u":jnp.where(obs["u"]>=5,11-obs["u"],obs["u"])}
                if mode=="block32":
                    mem=jax.tree.map(lambda a:jnp.where(tick%32==0,jnp.zeros_like(a),a),mem)
                z,new=encode(params,spec,{k:v[:,None] for k,v in obs.items()},keep,
                             memory=mem if persistent else None,offset=tick,window=window)
                action=jnp.argmax((z[:,0]@params["read"][:,-ACTIONS:].astype(z.dtype)).astype(jnp.float32),-1)
                mem=new if persistent else ()
            else:action=raw_action if mode=="raw" else jnp.zeros(b,jnp.int32)
            nxt=advance(st,action,tick,stall)
            reached=(nxt["pos"]==target(st)).all(-1)
            record=jnp.stack([reached,action==0,action>=5,(action==0)&~reached,raw_action==expert(st),action],-1)
            return (nxt,mem,estimate,seen),record
        init=(state,memory,jnp.zeros((b,2),jnp.int32),jnp.zeros(b,bool))
        return jax.lax.scan(step,init,jnp.arange(steps))[1]
    return jax.jit(run)


def live(p,spec,mode,b=128,stall=True,seed=180013,steps=512):
    return np.asarray(_live_kernel(spec,mode,b,stall,seed,steps)(p))


def paired_histories(p,spec,b=256,seed=180044):
    def work(params):
        ids=jnp.arange(b)
        first=initial(ids,seed);dest=target(first)
        # Move the starting position away from every requested target without revealing calibration.
        pos=first["pos"].at[:,0].set((dest[:,0]+1)%4)
        first={**first,"pos":pos}
        def trace(flip):
            st={**first,"sign":first["sign"]*flip}
            image=camera(st["stations"],st["tint"])
            def move(s,t):
                obs=observation(s,image,t)
                act=jnp.where(t==1,jnp.full(b,5 if flip==1 else 6),jnp.zeros(b,jnp.int32))
                return advance(s,act,t),(obs,expert(s))
            _,(rows,labels)=jax.lax.scan(move,st,jnp.arange(32))
            return jax.tree.map(lambda x:x.swapaxes(0,1),rows),labels[-1]
        l,lt=trace(1);r,rt=trace(-1)
        same=jnp.all(jnp.stack([jnp.array_equal(l[k],r[k]) for k in ("a","v","n","w")]))
        keep=jnp.ones((b,4),bool)
        def guess(obs):
            z,_=encode(params,spec,obs,keep)
            return (z[:,-1]@params["read"][:,-ACTIONS:].astype(z.dtype)).argmax(-1)
        gl,gr=guess(l),guess(r)
        reset=guess({k:v[:,-1:] for k,v in l.items()})
        return dict(sensor_identical=same,opposite=jnp.mean(lt!=rt),
                    matched=.5*((gl==lt).astype(float)+(gr==rt)),wrong=.5*((gl==rt).astype(float)+(gr==lt)),
                    reset=.5*((reset==lt).astype(float)+(reset==rt)),both=(gl==lt)&(gr==rt))
    out=jax.jit(work)(p)
    return {k:np.asarray(v) for k,v in out.items()}


def raw_receipt():
    # No model enters the positive or constant-action control.
    from models.g9_shared import Spec
    s=Spec("gpt2",128,1,128,"gelu")
    result={}
    for mode in ("raw","idle"):
        arr=live(None,s,mode)
        result[mode]={"goal_occupancy":float(arr[:,:,0].mean()),"subgoals_reached":float(arr[:,:,0].reshape(8,64,-1).any(1).mean()),
                      "hold_fraction":float(arr[:,:,1].mean()),"oracle_agreement":float(arr[:,:,4].mean())}
    if result["raw"]["oracle_agreement"]!=1 or result["raw"]["subgoals_reached"]!=1:raise ValueError(result)
    return result


if __name__=="__main__":
    import json
    print(json.dumps(raw_receipt(),indent=2))
