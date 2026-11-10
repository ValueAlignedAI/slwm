"""G4 JAX cores. Each conditional decoder receives one latent field [B,T,D]."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Spec:
    core: str
    width: int
    depth: int
    dtype: str = "bfloat16"
    expansion: int = 4
    tied: bool = False

    @property
    def groups(self):
        return 4 if self.core=="grouped" else 1


def initialize(spec, seed):
    rng=np.random.default_rng(seed)
    d=spec.width
    def weight(shape, scale=None):
        return rng.standard_normal(shape,dtype=np.float32)*(scale if scale is not None else shape[-2]**-.5)
    p={"edge":weight((384,d)), "embed":weight((259,d),.06), "type":weight((4,d),.04),
       "clock":weight((8,d),.015), "readout":weight((d,768),.02), "align":weight((d,128)),
       "bigram":np.zeros((259,259),np.float32),"norm":np.ones(d,np.float32),"bias":np.zeros(d,np.float32)}
    for i in range(spec.depth):
        k=f"b{i}_"
        p[k+"qkv"]=weight((d,3*d))
        p[k+"out"]=weight((d,d),1/np.sqrt(d*2*spec.depth))
        g=spec.groups
        h=spec.expansion*d
        if g==1:
            p[k+"up"]=weight((d,h))
            p[k+"down"]=weight((h,d),1/np.sqrt(2*h*spec.depth))
        else:
            p[k+"up"]=weight((g,d//g,h//g))
            p[k+"down"]=weight((g,h//g,d//g),1/np.sqrt(2*h/g*spec.depth))
        for key in ("n1","n2"):
            p[k+key]=np.ones(d,np.float32)
            p[k+key+"b"]=np.zeros(d,np.float32)
    return {k:jnp.asarray(v,dtype=jnp.float32) for k,v in p.items()}


def norm(x, scale, bias, dtype):
    f=x.astype(jnp.float32)
    mean=f.mean(-1,keepdims=True)
    f=(f-mean)*jax.lax.rsqrt(((f-mean)**2).mean(-1,keepdims=True)+1e-5)
    return (f*scale+bias).astype(dtype)


def unpack(batch, stats):
    a=(batch["a"].astype(jnp.float32)/16-7)/3
    v=batch["v"].astype(jnp.float32)/127.5-1
    n=jnp.clip((batch["n"].astype(jnp.float32)-stats[0])/stats[1],-8,8)
    return a,v,n


def encode(p, spec, batch, source, stats, *, erase=None, memory=None, offset=0, prior_action=None):
    """One observed source per example. Optional layer memory carries causal state between chunks."""
    dtype=getattr(jnp,spec.dtype)
    a,v,n=unpack(batch,stats)
    b,t,_=a.shape
    valid=batch["ok"].astype(bool)
    previous=jnp.zeros_like(n[:,:1,10:]) if prior_action is None else prior_action
    history=jnp.concatenate([n[:,:,:10],jnp.concatenate([previous,n[:,:-1,10:]],axis=1)],axis=-1)
    signals=jnp.concatenate([x*(source==m)[:,None,None] for m,x in enumerate((a,v,history))],axis=-1)
    signals=jnp.pad(signals,((0,0),(0,0),(0,384-signals.shape[-1])))
    x=signals.astype(dtype)@p["edge"].astype(dtype)
    ids=batch["x"].astype(jnp.int32)
    length=jnp.maximum((ids>0).sum(1),1)
    pos=jnp.arange(t)
    retimed=(pos[None,:]*(length[:,None]-1))//max(t-1,1)
    text=jnp.where((batch["family"]==5)[:,None],ids,jnp.take_along_axis(ids,retimed,axis=1))
    x=x+p["embed"][text].astype(dtype)*(source==3)[:,None,None]
    if erase is not None:
        x=x*(~erase)[:,None,None]
    # A fixed public clock supplies the same timing-only baseline to both cores.
    phases=(pos[:,None]+offset)*jnp.array([1,2,4,8])[None,:]*(2*jnp.pi/64)
    clock=jnp.concatenate([jnp.sin(phases),jnp.cos(phases)],axis=-1).astype(dtype)@p["clock"].astype(dtype)
    x=(x+p["type"][source,None,:].astype(dtype)+clock)*valid[...,None]
    carried=[]
    for i in range(spec.depth):
        name=f"b{i}_"
        z=norm(x,p[name+"n1"],p[name+"n1b"],dtype)
        q,k,vv=jnp.split(z@p[name+"qkv"].astype(dtype),3,axis=-1)
        prev=memory[i] if memory is not None else None
        if spec.core=="gpt2":
            heads=spec.width//64
            q,k,vv=(u.reshape(b,t,heads,64) for u in (q,k,vv))
            if prev is None:
                key,value,km=k,vv,valid
            else:
                key=jnp.concatenate([prev[0],k],1)
                value=jnp.concatenate([prev[1],vv],1)
                km=jnp.concatenate([prev[2],valid],1)
            past=key.shape[1]-t
            distance=(pos[:,None]+past)-jnp.arange(key.shape[1])[None,:]
            mask=(distance>=0)&(distance<64)
            logits=jnp.einsum("bthd,bshd->bhts",q,key).astype(jnp.float32)/8
            prob=jax.nn.softmax(jnp.where(mask[None,None]&km[:,None,None],logits,-1e9),axis=-1).astype(dtype)
            mix=jnp.einsum("bhts,bshd->bthd",prob,value).reshape(b,t,spec.width)
            carried.append((key[:,-64:],value[:,-64:],km[:,-64:]))
        else:
            g=8
            decay=jnp.exp(-1/jnp.geomspace(1.,128.,g))
            distance=pos[:,None]-pos[None,:]
            kernel=jnp.where(distance[None]>=0,(1-decay[:,None,None])*decay[:,None,None]**jnp.maximum(distance,0),0)
            obs=(q*jax.nn.sigmoid(k)*valid[...,None]).reshape(b,t,g,-1)
            smooth=jnp.einsum("gts,bsgc->btgc",kernel.astype(dtype),obs,preferred_element_type=jnp.float32)
            if prev is not None:
                smooth=smooth+prev[:,None]*(decay[None,None,:,None]**(pos[None,:,None,None]+1))
            carried.append(smooth[:,-1])
            gate=jax.nn.sigmoid(vv)
            mix=gate*smooth.reshape(b,t,-1).astype(dtype)+(1-gate)*obs.reshape(b,t,-1)
        x=x+mix@p[name+"out"].astype(dtype)
        z=norm(x,p[name+"n2"],p[name+"n2b"],dtype)
        if spec.groups==1:
            u=jax.nn.gelu(z@p[name+"up"].astype(dtype))@p[name+"down"].astype(dtype)
        else:
            z=z.reshape(b,t,spec.groups,-1)
            u=jax.nn.gelu(jnp.einsum("btgi,gih->btgh",z,p[name+"up"].astype(dtype)))
            u=jnp.einsum("btgh,ghi->btgi",u,p[name+"down"].astype(dtype)).reshape(b,t,-1)
        x=(x+u)*valid[...,None]
    field=norm(x,p["norm"],p["bias"],dtype)
    if spec.tied:
        basis=p["align"].astype(dtype)
        code=(field@basis).astype(jnp.float32)
        code=code*jax.lax.rsqrt((code*code).mean(-1,keepdims=True)+1e-5)
        field=(code.astype(dtype)@basis.T)*(spec.width/128)**.5
    return field,tuple(carried)


def errors(p,spec,z,batch,stats):
    a,v,n=unpack(batch,stats)
    pred=(z@p["readout"].astype(z.dtype)).astype(jnp.float32)
    mask=batch["ok"].astype(jnp.float32)
    def masked(error, valid):
        return (error*valid).sum(1)/jnp.maximum(valid.sum(1),1)
    results=[]
    temporal=[]
    for y,target in ((pred[:,:,:100],a),(pred[:,:,100:356],v),(pred[:,:,356:373],n)):
        diff=y-target
        results.append(masked(jnp.mean(diff**2,axis=-1),mask))
        temporal.append(masked(jnp.mean(jnp.diff(diff,axis=1)**2,axis=-1),mask[:,1:]*mask[:,:-1]))
    ids=batch["x"].astype(jnp.int32)
    is_lm=batch["family"]==5
    nxt=jnp.pad(ids[:,1:],((0,0),(0,1)))
    target=jnp.where(is_lm[:,None],nxt,ids)
    prev=jnp.where(is_lm[:,None],ids,jnp.pad(ids[:,:-1],((0,0),(1,0)),constant_values=1))
    logits=pred[:,:,448:707]+p["bigram"][prev]
    lp=jax.nn.log_softmax(logits)
    nll=-jnp.take_along_axis(lp,target[...,None],-1)[...,0]
    results.append(masked(nll,(target>0)*mask))
    delta=jnp.mean((pred[:,:-1,384:401]-n[:,1:])**2,axis=-1)
    upcoming=masked(delta,mask[:,1:]*mask[:,:-1])
    return jnp.stack(results,axis=-1),jnp.stack(temporal+[jnp.zeros_like(upcoming)],axis=-1),upcoming,pred


def centroid(z, batch):
    m=batch["ok"][...,None].astype(jnp.float32)
    return (z.astype(jnp.float32)*m).sum(1)/jnp.maximum(m.sum(1),1)


def alignment(p,z,paired,text_pair,batch,block,direct=False,denom=None):
    if direct:
        mask=batch["ok"].reshape(len(z),4,-1,1).astype(jnp.float32)
        slots=z.astype(jnp.float32).reshape(len(z),4,-1,z.shape[-1])
        vec=((slots*mask).sum(2)/jnp.maximum(mask.sum(2),1)).reshape(len(z),-1)
    else:
        projection=p["align"].astype(z.dtype)
        vec=(centroid(z,batch).astype(z.dtype)@projection).astype(jnp.float32)
    vec=vec*jax.lax.rsqrt((vec*vec).sum(-1,keepdims=True)+1e-8)
    a,b=jnp.split(vec,2,axis=0)
    a,b=(x.reshape(-1,block,x.shape[-1]) for x in (a,b))
    score=jnp.einsum("gnd,gmd->gnm",a,b)/.1
    n=paired.shape[0]
    fam=batch["family"][:n].reshape(-1,block)
    cap=batch["cap"][:n].reshape(-1,block)
    text_pair=text_pair.reshape(-1,block)
    ok=paired.reshape(-1,block)
    allowed=(fam[:,:,None]==fam[:,None,:])&ok[:,:,None]&ok[:,None,:]
    equivalent_text=(cap[:,:,None]==cap[:,None,:])&text_pair[:,:,None]&text_pair[:,None,:]
    positive=allowed & (equivalent_text|jnp.eye(block,dtype=bool))
    def direction(s,allow,pos):
        a=jax.nn.logsumexp(jnp.where(allow,s,-1e5),axis=-1)
        b=jax.nn.logsumexp(jnp.where(pos,s,-1e5),axis=-1)
        return ((a-b)*ok).sum()/(jnp.maximum(ok.sum(),1) if denom is None else denom)
    return .5*(direction(score,allowed,positive)+direction(score.swapaxes(-1,-2),allowed.swapaxes(-1,-2),positive.swapaxes(-1,-2)))


def motion_energy(x,valid):
    mask=valid[...,None].astype(jnp.float32)
    y=(x-(x*mask).sum(1,keepdims=True)/jnp.maximum(mask.sum(1,keepdims=True),1))*mask
    return jnp.sum(y*y,axis=(1,2))/jnp.maximum(mask.sum((1,2))*x.shape[-1],1)


def effective_denominators(data,ids,views,accumulations,policy):
    """Divide by effective-batch counts when gradients are averaged across microbatches."""
    aligned=(views[:,0]!=views[:,1])&~(views[:,2].astype(bool)|views[:,3].astype(bool))
    family=data["family"][ids]
    size=lambda n:jnp.maximum(n,1)/accumulations
    den={"align":size(aligned.sum()),"future":size(2*((family==4)&data["has"][ids,2].astype(bool)).sum())}
    if policy=="motion":
        ok=data["ok"][ids]
        a=((data["a"][ids].astype(jnp.float32)/16-7)/3).reshape(len(ids),64,10,10).mean(-1)
        v=(data["v"][ids].astype(jnp.float32)/127.5-1).reshape(len(ids),64,4,4,4,4).mean((3,5)).reshape(len(ids),64,16)
        for key,source,x in (("motion_v",0,v),("motion_a",1,a)):
            eligible=(family==0)&(motion_energy(x,ok)>.001)
            used=(views[:,:2]==source)&~views[:,2:].astype(bool)
            den[key]=size((used&eligible[:,None]).sum())
    return den


def motion_term(prediction,target,valid,eligible,denom=None):
    """Centered motion plus same-recording temporal foils. The decoder sees only the observed source."""
    mask=valid.astype(jnp.float32)[...,None]
    def center(x):
        return (x-(x*mask).sum(1,keepdims=True)/jnp.maximum(mask.sum(1,keepdims=True),1))*mask
    y,truth=center(prediction),center(target)
    energy=jnp.sum(truth*truth,axis=(1,2))/jnp.maximum(mask.sum((1,2))*truth.shape[-1],1)
    mse=jnp.sum((y-truth)**2,axis=(1,2))/jnp.maximum(mask.sum((1,2))*truth.shape[-1],1)
    valid3=mask[:,16:]*mask[:,8:-8]*mask[:,:-16]
    query=y[:,8:-8]*valid3
    def cosine(x):
        x=x*valid3
        return jnp.sum(query*x,axis=(1,2))*jax.lax.rsqrt(jnp.sum(query**2,axis=(1,2))*jnp.sum(x*x,axis=(1,2))+.01)
    logits=jnp.stack([cosine(truth[:,8:-8]),cosine(truth[:,:-16]),cosine(truth[:,16:])],-1)/.15
    keep=eligible & (energy>.001)
    per=jnp.minimum(mse/jnp.maximum(energy,.0025),12.)-.5*jax.nn.log_softmax(logits)[:,0]
    return jnp.sum(per*keep)/(jnp.maximum(keep.sum(),1) if denom is None else denom)


def objective(p,spec,batch,views,stats,policy,block=32,normalizers=None):
    den={} if normalizers is None else normalizers
    inputs=jax.tree.map(lambda x:jnp.concatenate([x,x],axis=0),batch)
    sources=jnp.concatenate([views[:,0],views[:,1]],axis=0)
    erase=jnp.concatenate([views[:,2],views[:,3]]).astype(bool)
    z,_=encode(p,spec,inputs,sources,stats,erase=erase)
    err,change,forecast,pred=errors(p,spec,z,inputs,stats)
    has=inputs["has"].astype(bool)
    self_mask=jax.nn.one_hot(sources,4,dtype=jnp.bool_)
    multiple=has.sum(1)>1
    score=has & (jnp.where(multiple[:,None],~self_mask,self_mask) if policy in ("cross","motion") else self_mask)
    per=(err+.25*change)*score
    supervised=(per.sum(1)/jnp.maximum(score.sum(1),1)).mean()
    if policy=="motion":
        a,v,_=unpack(inputs,stats)
        b,t=pred.shape[:2]
        face=lambda x:x.reshape(b,t,4,4,4,4).mean((3,5)).reshape(b,t,16)
        bands=lambda x:x.reshape(b,t,10,10).mean(-1)
        av=(inputs["family"]==0)&~erase
        supervised=supervised+.12*motion_term(face(pred[:,:,100:356]),face(v),inputs["ok"],av&(sources==0),den.get("motion_v"))
        supervised=supervised+.12*motion_term(bands(pred[:,:,:100]),bands(a),inputs["ok"],av&(sources==1),den.get("motion_a"))
    # An explicit one-step dynamics term is available wherever the robot stream exists.
    robot=has[:,2] & (inputs["family"]==4)
    future=(forecast*robot).sum()/den.get("future",jnp.maximum(robot.sum(),1))
    pair=(views[:,0]!=views[:,1]) & ~(views[:,2].astype(bool)|views[:,3].astype(bool))
    align=alignment(p,z,pair,(views[:,0]==3)|(views[:,1]==3),inputs,block,spec.tied,den.get("align"))
    loss=supervised+.12*align+.12*future
    return loss,jnp.stack([loss,supervised,align,future])
