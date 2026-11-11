"""Matched continuous trajectories and context segmentation for G10."""
from functools import partial
import jax
import jax.numpy as jnp
from models.g9_tabletop import ACTIONS, MOVES, camera, expert, initial, keys_for, observation, randint
from models.g9_shared import encode, decode, errors

TICKS = 512
PATTERNS = jnp.asarray([[1,1,1,1]]*8+[[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0],
                         [1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],bool)


def schedule(ids,seed,changes):
    state=initial(ids,seed);keys=keys_for(ids,seed);ticks=jnp.arange(TICKS)
    intervals=32+randint(keys,32,(17,),65)
    steps=jnp.sum(ticks[None,:,None]>=jnp.cumsum(intervals,axis=1)[:,None,:],axis=-1)
    first=104+randint(keys,33,(),145)
    second=320+randint(keys,34,(),96)
    if changes=="train":
        enabled=randint(keys,35,(),4)!=0
    else:
        enabled=jnp.full(len(ids),changes>0)
    notices=(ticks[None,:]==first[:,None])&enabled[:,None]
    if changes==2:
        notices=notices|(ticks[None,:]==second[:,None])
    # A change in one or both polarities is guaranteed. The notice never names the new signs.
    axis=randint(keys,36,(2,),3)
    factor=jnp.stack([jnp.where(axis!=1,-1,1),jnp.where(axis!=0,-1,1)],-1)
    signs=state["sign"][:,None,:]*jnp.where(((ticks[None,:]>=first[:,None])&enabled[:,None])[...,None],factor[:,0,None,:],1)
    if changes==2:signs=signs*jnp.where((ticks[None,:]>=second[:,None])[...,None],factor[:,1,None,:],1)
    goals=(state["goal"][:,None]+steps)%3
    return state,goals,signs,notices,first


def brief_visibility(goals):
    ticks=jnp.arange(goals.shape[1])
    different=jnp.concatenate([jnp.ones((len(goals),1),bool),goals[:,1:]!=goals[:,:-1]],axis=1)
    last=jax.lax.associative_scan(jnp.maximum,jnp.where(different,ticks[None,:],0),axis=1)
    return ticks[None,:]-last<4


def sensed(state,image,notice):
    r=observation(state,image,0)
    r["n"]=r["n"].at[:,5].set(notice.astype(jnp.float32))
    return r


def simulation(ids,seed):
    b=len(ids);keys=keys_for(ids,seed);st,goals,signs,notices,change=schedule(ids,seed,"train")
    t=jnp.arange(TICKS)
    random=randint(keys,45,(TICKS,),64)
    frequent=randint(keys,46,(),2).astype(bool)
    act=jnp.where(frequent[:,None]&(random>=60),5+random%2,random%5)
    delay=randint(keys,47,(),9);second=change+1+randint(keys,48,(),11)
    explicit=(t[None,:]==delay[:,None])|((t[None,:]==second[:,None])&notices.any(-1)[:,None])
    act=jnp.where(explicit,5+randint(keys,49,(TICKS,),2),act)
    previous=jnp.concatenate([jnp.zeros((b,1),jnp.int32),act[:,:-1]],axis=1)
    def dynamics(carry,item):
        pos,known=carry;u,old,sign,notice=item
        known=(known&~notice)|(old>=5)
        nxt=jnp.clip(pos+MOVES[u]*sign,0,3)
        return (nxt,known),(pos,known,nxt)
    _,vals=jax.lax.scan(dynamics,(st["pos"],st["known"]),(act.T,previous.T,signs.swapaxes(0,1),notices.T))
    pos,known,nxt=(v.swapaxes(0,1) for v in vals)
    def frame(p,g,known,u,sign,notice):
        cur={**st,"pos":p,"goal":g,"known":known,"previous":u,"sign":sign}
        return sensed(cur,image,notice),expert(cur)
    image=camera(st["stations"],st["tint"])
    obs,pol=jax.vmap(frame,in_axes=(1,1,1,1,1,1),out_axes=1)(pos,goals,known,previous,signs,notices)
    next_sign=jnp.concatenate([signs[:,1:],signs[:,-1:]],1);next_notice=jnp.concatenate([notices[:,1:],jnp.zeros((b,1),bool)],1)
    # Outcome of the specified control; future position and response never enter the current input.
    following,_=jax.vmap(frame,in_axes=(1,1,1,1,1,1),out_axes=1)(nxt,goals,known,act,next_sign,next_notice)
    obs.update(policy=pol,plan=act,next_n=following["n"],next_a=following["a"])
    return obs


def windows(bank,starts,length):
    ix=starts[:,None]+jnp.arange(length)[None,:]
    rows={k:bank[k][ix] for k in ("a","v","w","n")}
    zero=jnp.zeros(ix.shape,jnp.int32)
    rows.update(u=zero,policy=zero,plan=zero,next_n=bank["n"][ix+1],next_a=bank["a"][ix+1])
    return rows


def continuous(bank,starts,update,seed,batch=16):
    ns=batch*3//4
    ids=update*ns+jnp.arange(ns,dtype=jnp.int32)
    sim=simulation(ids,seed)
    random=jax.random.fold_in(jax.random.PRNGKey(seed+99),update);sel,mask=jax.random.split(random)
    real=windows(bank,starts[jax.random.randint(sel,(batch-ns,),0,len(starts))],TICKS)
    rows=jax.tree.map(lambda a,b:jnp.concatenate([a,b],0),sim,real)
    rows.update(keep=PATTERNS[jax.random.randint(mask,(batch,),0,16)],is_sim=jnp.arange(batch)<ns)
    return rows


def segment(r,length):
    chunks=TICKS//length
    return {k:jnp.repeat(x,chunks,0) if k in ("keep","is_sim") else x.reshape(-1,length,*x.shape[2:]) for k,x in r.items()}


def loss_from(z,p,r):
    pred=decode(p,z,r["plan"]);err=errors(pred,r)
    recon=jnp.stack([err[k].mean(-1) for k in ("a","v","n","w")],-1)
    weight=jnp.where(r["keep"],.10,1.)
    recon=((recon*weight).sum(-1)/weight.sum(-1)).mean()
    policy=(err["policy"].mean(1)*r["is_sim"]).sum()/jnp.maximum(r["is_sim"].sum(),1)
    future=.5*(err["next_n"].mean()+err["next_a"].mean())
    return policy+recon+future,jnp.stack([policy,recon,future])


def objective(p,spec,rows):
    z,_=encode(p,spec,rows,rows["keep"],window=512)
    return loss_from(z,p,rows)
