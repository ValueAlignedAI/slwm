"""Train-only analytic dynamics and real-case nuisance interventions."""
import numpy as np


def design(kin,plan,horizon):
    b=len(kin)
    hist=kin[:,:,2:].reshape(b,-1)/.05
    pose=(kin[:,-1,:2]-[.4,0])/.25
    u=plan[:,:horizon].reshape(b,-1)/.05
    return np.concatenate((np.ones((b,1)),pose,hist,u),1).astype(np.float64)


def identify(rows):
    x=rows["kin"];u=rows["plan"];y=rows["future"];groups=rows["group"]
    _,i,n=np.unique(groups,return_inverse=True,return_counts=True)
    weights=1./n[i];weights/=weights.sum()
    coeff={}
    for h in range(1,5):
        for name,m in (("active",h),("passive",0)):
            a=design(x,u,m);w=weights[:,None];mat=a.T@(a*w)
            # Small fixed regularizer protects singular duplicated history coordinates.
            mat+=.00003*np.eye(a.shape[1])
            coeff[f"{name}{h}"]=np.linalg.solve(mat,a.T@(y[:,h-1]*w)).astype("f4")
    return coeff


def reference(kin,plan,coef,active):
    return np.stack([design(kin,plan,h if active else 0)@coef[("active" if active else "passive")+str(h)] for h in range(1,5)],1).astype("f4")


def prepare(rows,coef):
    # Choose the same deterministic within-recording alternative plan for every processor.
    alt=np.empty_like(rows["plan"]);groups=rows["group"]
    for g in np.unique(groups):
        ix=np.flatnonzero(groups==g);order=np.lexsort((rows["tick"][ix],rows["episode"][ix]))
        ix=ix[order];alt[ix]=rows["plan"][np.roll(ix,max(1,len(ix)//2))]
    zero=np.zeros_like(alt)
    reset=np.broadcast_to(rows["kin"][:,-1:],rows["kin"].shape).copy();reset[:,:,2:]=0
    rows.update(alt_plan=alt,ref_active=reference(rows["kin"],rows["plan"],coef,True),
                ref_passive=reference(rows["kin"],zero,coef,False),ref_zero=reference(rows["kin"],zero,coef,True),
                ref_wrong=reference(rows["kin"],alt,coef,True),ref_reset=reference(reset,rows["plan"],coef,True),
                ref_pass_reset=reference(reset,zero,coef,False))
    return rows


def group_mean(values,groups):
    v=np.asarray(values);valid=np.isfinite(v)
    if not valid.any():return None
    _,idx,n=np.unique(groups[valid],return_inverse=True,return_counts=True)
    return float((np.bincount(idx,weights=v[valid])/n).mean())
