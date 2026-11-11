"""Shared physical object fields with strictly separate passive and proposed-control forecasts."""
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
from models.g12_world import norm

CORES=("gpt2","gated","oscillator","associative","transport","dense_spatial")
RECIPES=("flat","slots","bound","residual")
@dataclass(frozen=True)
class Spec:
    width:int=128
    layers:int=3
    heads:int=4
    field:int=128

def initial(s,seed):
    rng=np.random.default_rng(seed);d=s.width;z=s.field
    def mat(a,b,gain=1.):
        return jnp.asarray((rng.standard_normal((a,b))*gain/np.sqrt(a)).astype("f4"))
    p=dict(scene=mat(480+48+6,d),obj=mat(12,d),role=mat(4,d),text=mat(128,d),slot=mat(5,d,.15),time=mat(4,d,.12),
           emit=mat(d,z),moved=mat(z,1,.12),stationary=mat(z,1,.12),flow=mat(z,2,.08),
           passive=mat(z,8,.06),command=mat(z,8,.06),plan=mat(8,z),forecast=mat(z,2,.1),
           step=mat(4,z,.12),gate=mat(z,2,.08),dyn_bias=jnp.zeros((4,2)),bias=jnp.zeros(5))
    for i in range(s.layers):
        for k,a,b,g in (("qkv",d,3*d,1),("o",d,d,.65),("ff",d,4*d,1),("down",4*d,d,.65)):
            p[f"{i}_{k}"]=mat(a,b,g)
        for n in ("n1","n2"):
            p[f"{i}_{n}s"]=jnp.ones(d);p[f"{i}_{n}b"]=jnp.zeros(d)
    return p


def inputs(row,mode):
    px=row["px"].astype(jnp.float32)/255-.45
    geom=row["geo"].astype(jnp.float32);kin=row["kin"]
    text=row["goal"];role=row["bind"]
    if mode=="wrong":text=row["wrong"];role=row["wb"]
    if mode=="none":text=jnp.zeros_like(text);role=jnp.zeros_like(role)
    if mode in ("no_vision","text_only"):
        px=jnp.zeros_like(px);geom=jnp.zeros_like(geom)
    if mode=="text_only":kin=jnp.zeros_like(kin)
    if mode=="reset":
        px=jnp.broadcast_to(px[:,-1:],px.shape);geom=jnp.broadcast_to(geom[:,-1:],geom.shape)
        kin=jnp.broadcast_to(kin[:,-1:],kin.shape).at[:,:,2:].set(0)
    if mode not in ("correct","wrong","none","no_vision","text_only","reset","zero_plan","wrong_plan"):raise ValueError(mode)
    return px,geom,kin,text,role


def signal_mix(q,k,v,core,pos,valid,heads):
    b,t,o,d=q.shape;depth=d//heads;dt=q.dtype
    if core in ("gated","oscillator"):
        # Shared temporal state per slot. The common scene and pooled field couple the objects.
        from models.g12_world import mix
        def flatten(x):return x.transpose(0,2,1,3).reshape(b*o,t,d)
        h=mix(core,flatten(q),flatten(k),flatten(v),heads).reshape(b,o,t,d).transpose(0,2,1,3)
        return h
    def split(x):return x.reshape(b,t*o,heads,depth).transpose(0,2,1,3)
    q,k,v=map(split,(q,k,v));age=jnp.repeat(jnp.arange(t),o);age=age[:,None]-age[None,:]
    mask=age>=0;spatial=core in ("transport","dense_spatial")
    bias=jnp.zeros((1,1,1,1),jnp.float32)
    if spatial:
        loc=pos.reshape(b,t*o,2);vv=valid.reshape(b,t*o)
        sq=((loc[:,:,None]-loc[:,None,:])**2).sum(-1)
        scale=jnp.asarray([.09,.20,.45,2.],jnp.float32)
        bias=-.5*sq[:,None]/(scale[None,:,None,None]**2)
        bias=jnp.where((vv[:,:,None]&vv[:,None,:])[:,None],bias,0.)
    if core in ("gpt2","dense_spatial"):
        logits=(q@k.swapaxes(-2,-1)).astype(jnp.float32)/np.sqrt(depth)+bias
        weight=jax.nn.softmax(jnp.where(mask,logits,-jnp.inf),axis=-1)
    elif core in ("associative","transport"):
        a=jax.nn.elu(q.astype(jnp.float32))+1.;bkey=jax.nn.elu(k.astype(jnp.float32))+1.
        similarity=a@bkey.swapaxes(-2,-1)
        rate=jnp.asarray([.50,.77,.92,.995],jnp.float32)
        w=jnp.where(mask,similarity*jnp.exp(bias)*rate[None,:,None,None]**jnp.maximum(age,0),0.)
        weight=w/jnp.maximum(w.sum(-1,keepdims=True),1e-8)
    else:raise ValueError(core)
    return (weight.astype(dt)@v).transpose(0,2,1,3).reshape(b,t,o,d)


def propose(p,s,core,recipe,row,mode="correct"):
    dt=jnp.bfloat16
    def linear(x,k):return x.astype(dt)@p[k].astype(dt)
    px,geom,kin,text,bind=inputs(row,mode);b,t=px.shape[:2]
    mask=geom[:,:,:,:,5:6]
    local=(geom-jnp.array([.5,.5,0,0,0,0]))*jnp.array([2.,2.,5.,5.,75.,1.])*mask
    obj=local.reshape(b,t,4,12);coords=(geom[:,:,:,:,:2]*mask).sum(3)/jnp.maximum(mask.sum(3),1.)
    loc=jnp.concatenate((jnp.zeros((b,t,1,2)),coords),2)
    valid=jnp.concatenate((jnp.zeros((b,t,1),bool),mask[...,0].sum(3)>0),2)
    fixed=(kin-jnp.array([.4,0,0,0,0,0]))*jnp.array([4.,4.,16.,16.,16.,16.])
    scene=jnp.concatenate((px,obj.reshape(b,t,48),fixed),-1)
    # Flat models receive exactly the same color geometry and words through the global projection.
    h=linear(scene,"scene")[:,:,None,:]+p["time"].astype(dt)[None,:,None,:]+p["slot"].astype(dt)[None,None]
    h=h+linear(text.astype(jnp.float32)/127,"text")[:,None,None]
    # Both projections are active in all recipes. The flat control pools these codes across color slots.
    oc=linear(obj,"obj");rc=linear(bind,"role")[:,None]
    if recipe=="flat":oc=jnp.broadcast_to(oc.mean(2,keepdims=True),oc.shape)
    if recipe in ("flat","slots"):rc=jnp.broadcast_to(rc.mean(2,keepdims=True),rc.shape)
    if recipe not in RECIPES:raise ValueError(recipe)
    edge=oc+rc;h=h+jnp.concatenate((edge.mean(2,keepdims=True),edge),2)
    for i in range(s.layers):
        z=norm(h)*p[f"{i}_n1s"].astype(dt)+p[f"{i}_n1b"].astype(dt)
        q,k,v=jnp.split(linear(z,f"{i}_qkv"),3,-1)
        m=signal_mix(q,k,v,core,loc,valid,s.heads)
        h=h+linear(m,f"{i}_o")
        z=norm(h)*p[f"{i}_n2s"].astype(dt)+p[f"{i}_n2b"].astype(dt)
        h=h+linear(jax.nn.gelu(linear(z,f"{i}_ff")),f"{i}_down")
    field=norm(linear(norm(h[:,-1]),"emit"));slots=field[:,1:];global_state=norm(.5*(field[:,0]+slots.mean(1)))
    moved=jnp.concatenate((linear(global_state,"stationary"),linear(slots,"moved")[:,:,0]),axis=1).astype(jnp.float32)+p["bias"]
    passive=linear(global_state,"passive").astype(jnp.float32).reshape(b,4,2)
    act=linear(global_state,"command").astype(jnp.float32).reshape(b,4,2)
    plan=row["plan"] if mode!="wrong_plan" else row["alt_plan"]
    if mode=="zero_plan":plan=jnp.zeros_like(plan)
    causal=jnp.tril(jnp.ones((4,4),jnp.float32))
    proposed=(plan[:,None]*causal[None,:,:,None]).reshape(b,4,8)/.05
    latent=jnp.tanh(global_state[:,None]+linear(proposed,"plan")+p["step"].astype(dt)[None])
    dyn=linear(latent,"forecast").astype(jnp.float32)+p["dyn_bias"]
    if recipe=="residual":
        # Physical references use only the allowed prefix and each horizon's control prefix.
        pass_ref=row["ref_pass_reset"] if mode=="reset" else row["ref_passive"]
        ref={"wrong_plan":"ref_wrong","zero_plan":"ref_zero","reset":"ref_reset"}.get(mode,"ref_active")
        passive=passive*.4+(0. if mode=="text_only" else pass_ref/.08)
        dyn=dyn*.35+(0. if mode=="text_only" else row[ref]/.08)
    return dict(mover=moved,flow=linear(slots,"flow").astype(jnp.float32)*30,passive=passive*.08,
                active=dyn*.08,action=act*.05,certainty=linear(global_state,"gate").astype(jnp.float32),field=field.astype(jnp.float32))


def loss(p,s,core,recipe,r):
    y=propose(p,s,core,recipe,r);valid=r["mover"]>=0;target=jnp.maximum(r["mover"],0)
    ce=-jnp.take_along_axis(jax.nn.log_softmax(y["mover"]),target[:,None],axis=1)[:,0]
    obj=(ce*valid).sum()/jnp.maximum(valid.sum(),1)
    def huber(error,scale):
        a=jnp.abs(error/scale);return jnp.where(a<1.,.5*a*a,a-.5)
    dyn=huber(y["active"]-r["future"],.035).mean()
    passive=huber(y["passive"]-r["future"],.07).mean()
    actor=huber(y["action"]-r["plan"],.035).mean()
    mask=r["visible"]&r["first"][:,None]
    flow=(huber(y["flow"]-r["flow"],18).mean(-1)*mask).sum()/jnp.maximum(mask.sum(),1)
    ok=(jnp.argmax(y["mover"],-1)==target).astype("float32")
    phys=(jnp.linalg.norm(y["active"][:,-1]-r["future"][:,-1],axis=-1)<.012).astype("float32")
    def bce(v,t):return jax.nn.softplus(v)-v*jax.lax.stop_gradient(t)
    uc=(bce(y["certainty"][:,0],ok)*valid).sum()/jnp.maximum(valid.sum(),1)+bce(y["certainty"][:,1],phys).mean()
    return obj+2*dyn+.7*passive+.4*actor+.35*flow+.10*uc,jnp.stack([obj,dyn,passive,actor,flow,uc])
