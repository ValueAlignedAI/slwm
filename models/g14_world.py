"""Matched dense and spatial contact processors with a portable NumPy/JAX forward path."""
import numpy as np

CORES=("gpt2","transport","gated","dense_spatial")
ARMS=("direct","protected")

def initialize(dim,seed,width=80,layers=2):
    rng=np.random.default_rng(seed)
    def mat(a,b,g=1):return (rng.standard_normal((a,b))*g/np.sqrt(a)).astype("f4")
    s,im,f=dim;p=dict(in_slot=mat(s,width),in_img=mat(im,width,.35),time=mat(4,width,.1),slot=mat(5,width,.1),
               emit=mat(width,128),field=mat(128,width),candidate=mat(f,width,.7),response=mat(11,width,.4),
               next=mat(width,11,.08),risk=mat(width,1,.08),bias=np.zeros(11,"f4"))
    for i in range(layers):
        for k,a,b in (("qkv",width,3*width),("out",width,width),("ff",width,3*width),("down",3*width,width)):
            p[f"{i}_{k}"]=mat(a,b,.65 if k in ("out","down") else 1)
    return p

def norm(x,xp):
    c=x-xp.mean(x,axis=-1,keepdims=True);return c/xp.sqrt(xp.mean(c*c,axis=-1,keepdims=True)+1e-5)
def sigmoid(x,xp):return 1./(1.+xp.exp(-xp.clip(x,-25.,25.)))
def softmax(x,xp):
    e=xp.exp(x-xp.max(x,axis=-1,keepdims=True));return e/xp.sum(e,axis=-1,keepdims=True)
def core_mix(q,k,v,pos,core,xp):
    b,t,o,d=q.shape
    if core=="gated":
        h=xp.zeros_like(q[:,0]);out=[]
        for j in range(t):
            keep=sigmoid(k[:,j],xp);h=keep*h+(1-keep)*v[:,j];out.append(sigmoid(q[:,j],xp)*h)
        return xp.stack(out,axis=1)
    heads=4
    def split(x):return x.reshape(b,t*o,heads,d//heads).transpose(0,2,1,3)
    q,k,v=map(split,(q,k,v));clock=xp.repeat(xp.arange(t),o);age=clock[:,None]-clock[None,:]
    bias=xp.zeros((1,1,1,1),dtype=q.dtype)
    if core in ("transport","dense_spatial"):
        locations=pos.reshape(b,t*o,2);dist=xp.sum((locations[:,:,None]-locations[:,None,:])**2,axis=-1)
        scale=xp.asarray([.20,.45,.85,3.0],dtype=q.dtype)
        bias=-.5*dist[:,None,:,:]/(scale[None,:,None,None]**2)
    if core in ("gpt2","dense_spatial"):
        scores=(q@k.swapaxes(-1,-2))/np.sqrt(d//heads)+bias
        prob=softmax(xp.where(age>=0,scores,-1e5),xp)
    elif core=="transport":
        qp=xp.where(q>=0,q+1,xp.exp(xp.minimum(q,0)))
        kp=xp.where(k>=0,k+1,xp.exp(xp.minimum(k,0)))
        rate=xp.asarray([.55,.78,.93,.99],dtype=q.dtype)
        weight=xp.where(age>=0,(qp@kp.swapaxes(-1,-2))*xp.exp(bias)*rate[None,:,None,None]**xp.maximum(age,0),0)
        prob=weight/xp.maximum(weight.sum(-1,keepdims=True),1e-7)
    else:raise ValueError(core)
    return (prob@v).transpose(0,2,1,3).reshape(b,t,o,d)

def predict(p,r,core,arm,xp=np,include_long=False):
    h=r["slots"]@p["in_slot"]+(r["image"]@p["in_img"])[:,:,None,:]+p["time"][None,:,None,:]+p["slot"][None,None,:,:]
    for i in range(2):
        q,k,v=xp.split(norm(h,xp)@p[f"{i}_qkv"],3,axis=-1)
        h=h+core_mix(q,k,v,r["location"],core,xp)@p[f"{i}_out"]
        u=norm(h,xp)@p[f"{i}_ff"];h=h+(u*sigmoid(u,xp))@p[f"{i}_down"]
    slots=norm(norm(h[:,-1],xp)@p["emit"],xp)
    focus=xp.sum(slots*r["focus"][:,:,None],axis=1)
    z=norm(.4*slots[:,-1]+.4*focus+.2*slots.mean(1),xp)
    if arm=="protected":z=xp.concatenate((r["raw_action"],z[:,:126]),axis=1)
    elif arm!="direct":raise ValueError(arm)
    proposal=xp.tanh((z@p["field"])[:,None,:]+r["phi"]@p["candidate"]+r["prior"]@p["response"])
    y=proposal@p["next"]+p["bias"]
    if arm=="protected":y=r["prior"]+.45*y
    risk=sigmoid((proposal@p["risk"])[:,:,0],xp)
    if include_long:
        value=r["long_prior"]+.65*(proposal@p["continuation"])[:,:,0]
        return y,risk,z,value
    return y,risk,z

def objective(p,r,core,arm):
    import jax
    import jax.numpy as j
    long="continuation" in p
    if long:y,risk,_,value=predict(p,r,core,arm,j,True)
    else:y,risk,_=predict(p,r,core,arm,j)
    delta=y-r["y"];ae=j.abs(delta)
    smooth=j.where(ae<.35,.5*delta*delta,.35*(ae-.175))
    actual=r["y"][:,:,-1];score=y[:,:,-1];soft=jax.nn.softmax(actual/.16,axis=1)
    ranking=-j.sum(soft*jax.nn.log_softmax(score/.16,axis=1),1).mean()
    target=j.argmax(r["focus"],1)+1;effects=smooth[:,:,:10].reshape(len(y),y.shape[1],5,2)
    obj=j.take_along_axis(effects,target[:,None,None,None],axis=2).mean()
    unreliable=jax.lax.stop_gradient((ae[:,:,-1]>.15).astype("f4"))
    brier=j.mean((risk-unreliable)**2)
    loss=2.3*smooth[:,:,-1].mean()+effects.mean()+.8*obj+.15*ranking+.06*brier
    if long:
        valid=r["has_long"];v=value-value[:,1:2];target=r["long"]
        d=j.abs(v-target);term=j.where(d<.4,.5*d*d,.4*(d-.2)).mean(1)
        pair=-(jax.nn.softmax(target/.24,axis=1)*jax.nn.log_softmax(v/.24,axis=1)).sum(1)
        loss+=2.2*((term+.12*pair)*valid).sum()/j.maximum(valid.sum(),1)
    return loss,j.stack([smooth[:,:,-1].mean(),effects.mean(),obj,ranking,brier])
