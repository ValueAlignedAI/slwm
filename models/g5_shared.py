"""A 128-dimensional causal field shared by all G5 targets and sensor combinations."""
import jax
import jax.numpy as jnp

from models.g4_shared import initialize, norm


def parameters(spec, seed):
    p=initialize(spec, seed)
    p["readout"]=p["readout"][:128,:635]
    p["plan"]=jax.random.normal(jax.random.key(seed+93),(2,spec.width))*.1
    return p


def attention(q, k, v, valid, previous):
    b,t,d=q.shape
    q,k,v=(a.reshape(b,t,d//64,64) for a in (q,k,v))
    if previous is not None:
        k=jnp.concatenate((previous[0],k),1)
        v=jnp.concatenate((previous[1],v),1)
        valid=jnp.concatenate((previous[2],valid),1)
    distance=jnp.arange(t)[:,None]+k.shape[1]-t-jnp.arange(k.shape[1])[None,:]
    mask=(distance>=0)&(distance<64)
    scores=jnp.einsum("bthd,bshd->bhts",q,k).astype(jnp.float32)/8
    prob=jax.nn.softmax(jnp.where(mask[None,None]&valid[:,None,None],scores,-1e9),axis=-1).astype(q.dtype)
    out=jnp.einsum("bhts,bshd->bthd",prob,v).reshape(b,t,d)
    return out,(k[:,-64:],v[:,-64:],valid[:,-64:])


def retention(q, k, value, valid, previous):
    b,t,d=q.shape
    groups=8
    prior=jnp.exp(-1/jnp.geomspace(1.,128.,groups))
    base=jnp.log(prior/(1-prior))
    key=k.reshape(b,t,groups,-1)
    gate=key.astype(jnp.float32).mean(-1)+base
    log_decay=jnp.where(valid[...,None],jax.nn.log_sigmoid(gate),0.)
    log_prefix=jnp.cumsum(log_decay,axis=1).transpose(0,2,1)
    # Product of admitted decays after an observation, including the current update.
    distance=log_prefix[:,:,:,None]-log_prefix[:,:,None,:]
    causal=jnp.arange(t)[:,None]>=jnp.arange(t)[None,:]
    kernel=jnp.where(causal,jnp.exp(jnp.minimum(distance,0)),0.)
    kernel=kernel*(-jnp.expm1(log_decay)).transpose(0,2,1)[:,:,None,:]
    obs=(q*jax.nn.sigmoid(k)).reshape(b,t,groups,-1)
    smooth=jnp.einsum("bgts,bsgc->btgc",kernel.astype(q.dtype),obs,preferred_element_type=jnp.float32)
    if previous is not None:
        smooth=smooth+jnp.exp(log_prefix.transpose(0,2,1))[...,None]*previous[:,None]
    mixing=jax.nn.sigmoid(value)
    return mixing*smooth.reshape(b,t,d).astype(q.dtype)+(1-mixing)*obs.reshape(b,t,d),smooth[:,-1]


def encode(p,spec,batch,source,*,memory=None,offset=0,erase=False):
    dtype=getattr(jnp,spec.dtype)
    valid=batch["ok"].astype(bool)
    b,t=valid.shape
    pos=jnp.arange(t)+offset
    a=batch["a"].astype(jnp.float32)/48-7/3
    v=batch["v"].astype(jnp.float32)/127.5-1
    # Numeric input is standardized with training statistics by the sampler.
    n=batch["n"].astype(jnp.float32)
    keep=source.astype(dtype)
    inputs=jnp.concatenate([x*keep[:,i,None,None] for i,x in enumerate((a,v,n))],-1)
    x=inputs.astype(dtype)@p["edge"][:inputs.shape[-1]].astype(dtype)
    ids=batch["input_x"].astype(jnp.int32)
    x=x+p["embed"][ids].astype(dtype)*keep[:,3,None,None]
    x=x*batch["sense"][...,None]
    x=x/jnp.sqrt(jnp.maximum(keep.sum(-1),1))[:,None,None]
    x=jnp.where(jnp.asarray(erase),jnp.zeros_like(x),x)
    phases=pos[:,None]*jnp.array([1,2,4,8])[None,:]*(2*jnp.pi/64)
    clock=jnp.concatenate([jnp.sin(phases),jnp.cos(phases)],-1).astype(dtype)@p["clock"].astype(dtype)
    tags=keep@p["type"].astype(dtype)
    # Plans are issued controls and are independent of any withheld future observation.
    x=(x+clock+tags[:,None]+batch["u"].astype(dtype)@p["plan"].astype(dtype))*valid[...,None]
    memories=[]
    for i in range(spec.depth):
        key=f"b{i}_"
        z=norm(x,p[key+"n1"],p[key+"n1b"],dtype)
        q,k,vv=jnp.split(z@p[key+"qkv"].astype(dtype),3,axis=-1)
        previous=None if memory is None else memory[i]
        op=attention if spec.core=="gpt2" or (i+1)%4==0 else retention
        mixed,carried=op(q,k,vv,valid,previous)
        memories.append(carried)
        x=x+mixed@p[key+"out"].astype(dtype)
        z=norm(x,p[key+"n2"],p[key+"n2b"],dtype)
        ff=jax.nn.gelu(z@p[key+"up"].astype(dtype))@p[key+"down"].astype(dtype)
        x=(x+ff)*valid[...,None]
    field=norm(x,p["norm"],p["bias"],dtype)@p["align"].astype(dtype)
    code=field.astype(jnp.float32)
    return (code*jax.lax.rsqrt((code**2).mean(-1,keepdims=True)+1e-5)).astype(dtype),tuple(memories)


def decoded(p,spec,batch,source):
    z,_=encode(p,spec,batch,source)
    return (z@p["readout"].astype(z.dtype)).astype(jnp.float32)


def objective(p,spec,batch):
    src=batch["source"].astype(bool)
    pred=decoded(p,spec,batch,src)
    valid=batch["ok"].astype(jnp.float32)
    fam=batch["family"]
    def masked(a,mask):
        return jnp.sum(a*mask,axis=1)/jnp.maximum(mask.sum(1),1)
    audio=batch["a"].astype(jnp.float32)/48-7/3
    video=batch["v"].astype(jnp.float32)/127.5-1
    targets=batch["has"].astype(bool)&~src
    losses=[]
    for output,target in ((pred[:,:,:100],audio),(pred[:,:,100:356],video),(pred[:,:,356:373],batch["target_n"])):
        diff=output-target
        spatial=masked((diff**2).mean(-1),valid)
        motion=masked((jnp.diff(diff,axis=1)**2).mean(-1),valid[:,1:]*valid[:,:-1])
        losses.append(spatial+.25*motion)
    x=batch["x"].astype(jnp.int32)
    lm=(fam==5)[:,None]
    target=jnp.where(lm,jnp.pad(x[:,1:],((0,0),(0,1))),x)
    prev=jnp.where(lm,x,jnp.pad(x[:,:-1],((0,0),(1,0)),constant_values=1))
    lp=jax.nn.log_softmax(pred[:,:,373:632]+p["bigram"][prev])
    losses.append(masked(-jnp.take_along_axis(lp,target[...,None],-1)[...,0],(target>0)*valid))
    targets=targets.at[:,3].set(targets[:,3]|(fam==5))
    cross=(jnp.stack(losses,-1)*targets).sum(-1)/jnp.maximum(targets.sum(-1),1)
    # No future pixels, state or targets are available to the action-conditioned predictor.
    # World-control outputs share the same 128-D state with the signal decoders.
    # Use a separate row block of the same readout matrix for these family-specific outputs.
    event_logits=pred[:,:,632]
    event=batch["event"].astype(jnp.float32)
    contact=masked(jax.nn.softplus(event_logits)*(1-event)+8*jax.nn.softplus(-event_logits)*event,valid)
    delta=pred[:,16:32,633:635]-batch["outcome"]
    dynamics=((delta**2).mean(-1)*jnp.linspace(.25,1.,16)).mean(1)+(delta[:,-1]**2).mean(-1)
    event_mask=(fam==7)&src[:,1]&~src[:,0]
    per=cross*(fam!=8)+contact*event_mask+dynamics*(fam==8)
    metrics=jnp.stack([per.mean(),(cross*(fam!=8)).mean(),(contact*event_mask).mean(),(dynamics*(fam==8)).mean()])
    return metrics[0],metrics
