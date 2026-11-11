"""Causal object fields, raw candidate predictors and a metric-aligned command interface."""
import numpy as np

def mean_group(values,g):
    values=np.asarray(values);ok=np.isfinite(values)
    if not ok.any():return None
    _,ix,n=np.unique(g[ok],return_inverse=True,return_counts=True)
    return float((np.bincount(ix,weights=values[ok])/n).mean())

def pack(data):
    # Exactly the same visible positions, robot telemetry, lower-resolution RGB and map reach every policy.
    q=np.asarray(data["q"],"f4");p=np.asarray(data["p"],"f4");motor=np.asarray(data["motor"],"f4")
    seen=np.asarray(data["seen"],"f4");u=np.asarray(data["u"],"f4");cmd=np.asarray(data["choices"],"f4")
    c=np.asarray(data["color"],int);target=np.asarray(data["target"],"f4")
    b,t=p.shape[:2];idx=np.arange(b);sel=q[idx,-1,c];goal=target-sel;dist=np.linalg.norm(goal,axis=1)
    axis=goal/np.maximum(dist[:,None],.0001);side=np.stack((-axis[:,1],axis[:,0]),1)
    basis=np.stack((axis,side),-1)
    def coord(x):
        return np.einsum("b...i,bij->b...j",x,basis)
    # Append the observed effector as a fifth slot. Coordinates are in the commanded object's goal frame.
    all_pos=np.concatenate((q,p[:,:,None]),axis=2)
    rel=coord(all_pos-sel[:,None,None])/.12
    delta=coord(np.diff(all_pos,axis=1,prepend=all_pos[:,:1]))/.05
    gp=coord(target[:,None,None]-all_pos)/.2
    gripper=coord(p[:,:,None]-all_pos)/.12
    flags=np.concatenate((seen,np.ones((b,t,1))),2)
    onehot=np.broadcast_to(np.eye(5,dtype="f4")[None,None],(b,t,5,5))
    focus=np.broadcast_to((np.arange(5)[None,:]==c[:,None])[:,None,:,None],(b,t,5,1))
    ap=coord(u)/.05;lag=coord(motor-p)/.05
    act=np.broadcast_to(np.concatenate((ap,lag),-1)[:,:,None],(b,t,5,4))
    slots=np.concatenate((rel,delta,gp,gripper,flags[:,:,:,None],onehot,focus,act),axis=-1).astype("f4")
    pixels=data["px"].reshape(b,t,10,16,3).astype("f4")/255
    image=pixels.reshape(b,t,2,5,4,4,3).mean((3,5)).reshape(b,t,24)-.5
    options=coord(cmd)/.05;k=cmd.shape[1]
    # Analytic contact features are available to raw and learned candidates before execution.
    r=coord(sel-p[:,-1])/.05;goalvec=coord(goal)/.15
    mov=coord(motor[:,-1]-p[:,-1])[:,None]/.05+options
    dot=(r[:,None]*mov).sum(-1);den=(mov**2).sum(-1)
    alpha=np.clip(dot/np.maximum(den,1e-5),0,1)
    gap=r[:,None]-alpha[:,:,None]*mov
    clearance=np.linalg.norm(gap,axis=-1)
    contact=np.clip(1.5-clearance/.7,0,1)
    push=(1-alpha[:,:,None])*mov*contact[:,:,None]
    other=coord(q[:,-1]-sel[:,None])/.15;other_dist=np.linalg.norm(other,axis=-1)
    closest=np.argsort(other_dist,axis=1)[:,1];near=other[idx,closest]
    common=np.column_stack((r,goalvec,lag[:,-1],ap[:,-1],delta[idx,-1,c],seen[idx,-1,c],near,
                            np.minimum(4,other_dist[idx,closest]),np.minimum(4,np.linalg.norm(r,axis=1))))
    repeated=np.broadcast_to(common[:,None],(b,k,common.shape[-1]))
    feat=np.concatenate((repeated,options,options**2,mov,gap,alpha[:,:,None],contact[:,:,None],push,
                         options*r[:,None],options*contact[:,:,None],np.linalg.norm(options,axis=-1)[:,:,None],
                         np.broadcast_to(np.eye(k,dtype="f4")[None],(b,k,k))),-1)
    # Low-rank cross terms distinguish contact and approach without granting a hidden state or outcome.
    feat=np.concatenate((feat,options[:,:,:,None].__mul__(repeated[:,:,None,:]).reshape(b,k,-1)),axis=-1).astype("f4")
    out=dict(slots=slots,image=image.astype("f4"),location=(all_pos-CENTER)/.25,phi=feat,command=options.astype("f4"),
             focus=np.eye(5,dtype="f4")[c],basis=basis.astype("f4"))
    if "effect" in data:
        effect=coord(data["effect"])/.05
        out["y"]=np.concatenate((effect.reshape(b,k,10),data["gain"][:,:,None]/.04),-1).astype("f4")
        out["world"]=np.asarray(data["world"],"i4")
    return out

CENTER=np.array([.375,0.],"f4")
def sample_weights(groups):
    _,inv,ct=np.unique(groups,return_inverse=True,return_counts=True);w=1./ct[inv];return w/w.sum()

def raw_fit(r,lam=.015):
    b,k,f=r["phi"].shape;x=r["phi"].reshape(-1,f).astype("f8");y=r["y"].reshape(b*k,-1).astype("f8")
    w=np.repeat(sample_weights(r["world"])/k,k)
    mu=(x*w[:,None]).sum(0);sd=np.maximum(np.sqrt(((x-mu)**2*w[:,None]).sum(0)),.035)
    z=(x-mu)/sd;avg=(y*w[:,None]).sum(0)
    coef=np.linalg.solve(z.T@(z*w[:,None])+lam*np.eye(f),z.T@((y-avg)*w[:,None]))
    return dict(mean=mu.astype("f4"),std=sd.astype("f4"),coef=coef.astype("f4"),bias=avg.astype("f4"),lam=np.array(lam,"f4"))

def raw_predict(r,p):
    return ((r["phi"]-p["mean"])/p["std"])@p["coef"]+p["bias"]
def attach(r,prior):
    r["prior"]=prior.astype("f4");ii=prior[:,:,-1].argmax(1)
    r["raw_action"]=r["command"][np.arange(len(ii)),ii]

def subset(r,ix):return {k:x[ix] for k,x in r.items()}

def paired_regret(scores,y):
    return (y[:,:,-1].max(1)-y[np.arange(len(y)),scores.argmax(1),-1])*40.

def raw_setup(train,dev):
    options=[]
    for lam in (.001,.008,.04,.2):
        p=raw_fit(train,lam);yp=raw_predict(dev,p)
        regret=mean_group(paired_regret(yp[:,:,-1],dev["y"]),dev["world"])
        err=mean_group(np.abs(yp[:,:,-1]-dev["y"][:,:,-1]).mean(1)*40.,dev["world"])
        options.append((regret+.25*err,lam,p))
    _,lam,p=min(options,key=lambda x:x[:2]);groups=train["world"];assignment=groups%5
    oof=np.zeros_like(train["y"])
    for f in range(5):
        fit=raw_fit(subset(train,assignment!=f),lam);oof[assignment==f]=raw_predict(subset(train,assignment==f),fit)
    attach(train,oof);attach(dev,raw_predict(dev,p));return p,[(v,l) for v,l,_ in options]
