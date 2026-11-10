"""Shared-state corrections to explicit training-only action and contact priors."""
import jax
import jax.numpy as jnp

from models.g5_shared import decoded, parameters
from training.g5_priors import geometry


def initialize(spec,seed):
    p=parameters(spec,seed)
    p["readout"]=p["readout"].at[:,632:635].set(0)
    return p


def world(p,spec,b,source,cal):
    video=b["v"].astype(jnp.float32)
    dv=video-jnp.concatenate([video[:,:1],video[:,:-1]],axis=1)
    audio=b["a"].astype(jnp.float32)
    da=audio-jnp.concatenate([audio[:,:1],audio[:,:-1]],axis=1)
    n=b["n"]
    history=n[:,:16,:6]
    plan=b["u"][:,16:32]
    has_state=source[:,2].astype(bool)&(b["family"]==8)
    has_plan=jnp.any(plan!=0,axis=(1,2))
    # Each time step sees only control positions at or before that step.
    def prior(t,w):
        idx=jnp.minimum(jnp.array([3,7,11,15]),t)
        f=geometry(history,plan,idx,jnp)
        return f@w[1:]+w[0]
    action=jax.vmap(prior)(jnp.arange(16),cal["coef"]).transpose(1,0,2)
    action=action*(has_state&has_plan)[:,None,None]
    motion=jnp.log1p(jnp.abs(dv).mean(-1))
    base=cal["contact"][0]+cal["contact"][1]*motion
    relative=n.copy()
    relative=relative.at[:,:,6:8].set(n[:,:,:2]-n[:,:,2:4])
    relative=relative.at[:,1:,8:10].set(n[:,1:,2:4]-n[:,:-1,2:4])
    changed=dict(b)
    changed["v"]=jnp.clip(video+.65*dv,0,255)
    changed["a"]=jnp.clip(audio+.65*da,0,255)
    changed["n"]=jnp.where((b["family"]==8)[:,None,None],relative,n)
    centered=plan-history[:,-1,None,2:4]
    changed["u"]=b["u"].at[:,16:32].set(jnp.where((has_state&has_plan)[:,None,None],centered,plan))
    shared=decoded(p,spec,changed,source)
    # Do not disclose hidden motion when evaluating erased or audio-only inputs.
    seen=source[:,1,None]*b["sense"]
    onset=shared[:,:,632]+cal["contact"][0]+seen*(base-cal["contact"][0])
    dyn=.25*shared[:,16:32,633:635]+action
    return shared,onset,dyn


def objective(p,spec,b,cal):
    pred,event_logits,dyn=world(p,spec,b,b["source"],cal)
    valid=b["ok"].astype(jnp.float32)
    fam=b["family"]
    src=b["source"].astype(bool)
    targets=b["has"].astype(bool)&~src
    def mean(v,m):
        return (v*m).sum(1)/jnp.maximum(m.sum(1),1)
    losses=[]
    for out,y in ((pred[:,:,:100],b["a"].astype(jnp.float32)/48-7/3),
                  (pred[:,:,100:356],b["v"].astype(jnp.float32)/127.5-1),(pred[:,:,356:373],b["target_n"])):
        diff=out-y
        losses.append(mean((diff**2).mean(-1),valid)+.25*mean((jnp.diff(diff,axis=1)**2).mean(-1),valid[:,1:]*valid[:,:-1]))
    x=b["x"].astype(jnp.int32)
    lm=(fam==5)[:,None]
    target=jnp.where(lm,jnp.pad(x[:,1:],((0,0),(0,1))),x)
    prev=jnp.where(lm,x,jnp.pad(x[:,:-1],((0,0),(1,0)),constant_values=1))
    logp=jax.nn.log_softmax(pred[:,:,373:632]+p["bigram"][prev])
    losses.append(mean(-jnp.take_along_axis(logp,target[...,None],-1)[...,0],valid*(target>0)))
    targets=targets.at[:,3].set(targets[:,3]|(fam==5))
    cross=(jnp.stack(losses,-1)*targets).sum(-1)/jnp.maximum(1,targets.sum(-1))
    event=b["event"].astype(jnp.float32)
    # Offset the fitted population log odds for the fixed 8x positive training weighting.
    logits=event_logits+jnp.log(8.)
    contact=mean(jax.nn.softplus(logits)*(1-event)+8*jax.nn.softplus(-logits)*event,valid)
    err=dyn-b["outcome"]
    action=((err**2).mean(-1)*jnp.linspace(.25,1.,16)).mean(1)+(err[:,-1]**2).mean(-1)
    scores=jnp.stack([(cross*(fam!=8)).mean(),(contact*((fam==7)&src[:,1]&~src[:,0])).mean(),
                      (action*(fam==8)).mean()])
    return scores.sum(),jnp.concatenate((scores.sum()[None],scores))
