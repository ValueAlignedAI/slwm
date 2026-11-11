"""Learn correspondence in the emitted world field with within-recording hard negatives."""
import jax
import jax.numpy as jnp
import numpy as np
from models.g9_shared import encode,decode,errors
from models.g10_context import PATTERNS,simulation,segment,loss_from

GROUPS=8
CANDIDATES=8
LENGTH=32


def add_edges(base,spec,seed):
    rng=np.random.default_rng(seed+8441)
    p=dict(base)
    p["utterance_edge"]=jnp.asarray(rng.normal(0,.04,(128,spec.d)).astype("f4"))
    p["motion_edge"]=jnp.zeros((256,spec.d),jnp.float32)
    p["utterance_read"]=jnp.asarray(rng.normal(0,.02,(spec.field,128)).astype("f4"))
    return p


def encode_record(p,spec,rows,keep):
    dtype=getattr(jnp,spec.dtype)
    language=rows["x"].astype(dtype)/127
    motion=rows["m"].astype(dtype)
    extra=(language@p["utterance_edge"].astype(dtype))*keep[:,3,None,None]
    extra=extra+(motion@p["motion_edge"].astype(dtype))*keep[:,1,None,None]
    return encode(p,spec,rows,keep,window=512,edge_residual=extra)[0]


def from_indices(bank,ix,previous):
    row={k:bank[k][ix] for k in ("a","v","n","w","x")}
    diff=jnp.abs(bank["v"][ix].astype(jnp.int16)-bank["v"][previous].astype(jnp.int16))
    row["m"]=jnp.clip(diff.reshape(*ix.shape,256,3).astype(jnp.float32).mean(-1)/64,0,2)
    empty=jnp.zeros(ix.shape,jnp.int32)
    row.update(u=empty,plan=empty,policy=empty,next_n=bank["n"][ix+1],next_a=bank["a"][ix+1])
    return row


def groups(bank,i,seed):
    key=jax.random.fold_in(jax.random.PRNGKey(seed+431),i)
    choice,temporal,mask,shift=jax.random.split(key,4)
    # Each source contributes four independent recording groups. Windows within each group cannot overlap.
    ids=jnp.concatenate([jax.random.randint(jax.random.fold_in(choice,j),(4,),0,bank[k].shape[0]) for j,k in enumerate(("star_len","holo_len"))])
    lengths=jnp.concatenate([bank["star_len"][ids[:4]],bank["holo_len"][ids[4:]]])
    origins=jnp.concatenate([bank["star_off"][ids[:4]],bank["holo_off"][ids[4:]]])
    span=lengths[:,None]//CANDIDATES
    jitter=(jax.random.uniform(temporal,(GROUPS,CANDIDATES))*jnp.maximum(span-LENGTH-1,1)).astype(jnp.int32)
    start=origins[:,None]+jnp.arange(CANDIDATES)[None,:]*span+jitter
    ix=(start[:,:,None]+jnp.arange(LENGTH)).reshape(-1,LENGTH)
    r=from_indices(bank,ix,jnp.maximum(ix-1,jnp.repeat(origins,CANDIDATES)[:,None]))
    r["keep"]=PATTERNS[jax.random.randint(mask,(GROUPS*CANDIDATES,),0,len(PATTERNS))]
    r["holo"]=jnp.repeat(jnp.arange(GROUPS)>=4,CANDIDATES)
    return r,jax.random.randint(shift,(GROUPS,1),1,CANDIDATES)


def normalized(x):
    x=x-x.mean(axis=1,keepdims=True)
    return x*jax.lax.rsqrt(jnp.maximum(jnp.sum(x*x,axis=-1,keepdims=True),1e-6))


def contrast(p,s,r,offset,mode):
    n=GROUPS*CANDIDATES
    def field(source):
        keep=jnp.broadcast_to(jnp.asarray(source,bool),(n,4))
        z=encode_record(p,s,r,keep)[:,-8:].astype(jnp.float32).mean(1)
        return normalized(z.reshape(GROUPS,CANDIDATES,s.field))
    audio=field((1,0,0,0));vision=field((0,1,0,0))
    scores=jnp.einsum("gid,gjd->gij",audio,vision)/.12
    idx=jnp.broadcast_to(jnp.arange(CANDIDATES),(GROUPS,CANDIDATES))
    delta=offset if mode=="shifted" else jnp.zeros_like(offset)
    def ce(logits,target):
        return -jnp.take_along_axis(jax.nn.log_softmax(logits),target[...,None],axis=-1).mean()
    value=(ce(scores,(idx+delta)%CANDIDATES)+ce(scores.swapaxes(1,2),(idx-delta)%CANDIDATES))/2
    return value,(scores.argmax(-1)==idx).mean()


def objective(p,s,bank,i,seed,mode,length):
    rr,offset=groups(bank,i,seed)
    z=encode_record(p,s,rr,rr["keep"]);pred=decode(p,z,rr["plan"]);err=errors(pred,rr)
    natural=(z@p["utterance_read"].astype(z.dtype)).astype(jnp.float32)
    truth=rr["x"].astype(jnp.float32)/127
    have=(rr["x"]!=0).any(-1)
    text=((natural-truth)**2).mean(-1)
    textrow=(text*have).sum(1)/jnp.maximum(1,have.sum(1))
    loss=jnp.stack([err[k].mean(1) for k in ("a","v","n")]+[jnp.where(rr["holo"],textrow,err["w"].mean(1))],-1)
    weights=jnp.where(rr["keep"],.10,1.)
    signal=((loss*weights).sum(-1)/weights.sum(-1)).mean()+.5*(err["next_a"].mean()+err["next_n"].mean())
    r=simulation(i*8+jnp.arange(8),seed+5533);mk=jax.random.fold_in(jax.random.PRNGKey(seed+441),i)
    r.update(keep=PATTERNS[jax.random.randint(mk,(8,),0,len(PATTERNS))],is_sim=jnp.ones(8,bool))
    r=segment(r,length);zs,_=encode(p,s,r,r["keep"],window=512)
    rehearsal,_=loss_from(zs,p,r)
    alignment,hits=contrast(p,s,rr,offset,mode)
    total=.5*rehearsal+signal+(.0 if mode=="reconstruct" else .6)*alignment
    return total,jnp.stack([rehearsal,signal,alignment,hits])
