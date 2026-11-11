"""Paired 95% intervals over complete independent sources with optional equal family weighting."""
import hashlib
import numpy as np


class Groups:
    def __init__(self,groups,families=None):
        self.groups=np.asarray(groups);self.family=np.asarray(families) if families is not None else np.full(len(groups),"all")
        self.cache={}

    def score(self,value):
        a=np.asarray(value,np.float64);mask=np.isfinite(a)
        if not mask.any():return None
        key=mask.tobytes()
        if key not in self.cache:
            ids,first,inv,counts=np.unique(self.groups[mask],return_index=True,return_inverse=True,return_counts=True)
            fs=self.family[mask][first];families=np.unique(fs)
            W=np.zeros((4000,len(ids)),np.float32);mean=np.zeros(len(ids))
            for fam in families:
                j=np.flatnonzero(fs==fam);n=len(j);mean[j]=1/(len(families)*n)
                seed=int.from_bytes(hashlib.sha256(("G13/source-bootstrap/"+str(fam)+"/"+"/".join(map(str,ids[j]))).encode()).digest()[:8],"big")
                W[:,j]=np.random.default_rng(seed).multinomial(n,np.full(n,1/n),size=4000)/(n*len(families))
            self.cache[key]=(inv,counts,W,mean,len(families))
        inv,ct,w,m,nf=self.cache[key];v=np.bincount(inv,weights=a[mask])/ct;lo,hi=np.quantile(w@v,[.025,.975])
        return dict(value=float(m@v),lo=float(lo),hi=float(hi),groups=len(ct),n=int(mask.sum()),families=nf)
