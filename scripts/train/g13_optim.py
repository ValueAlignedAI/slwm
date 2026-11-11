"""Shared optimizer, immutable checkpoints and verified two-disk custody for the G13 studies."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import jax
import jax.numpy as jnp
from scripts.train.g9_storage import disks,write,sha

ROOT=Path("/mnt/slwm-data/SLWM/g13")


def roots(study):
    if study not in ("speech","speech-portable","robot","robot-corrected","holo","components"):raise ValueError("Unknown study")
    return tuple(p.parent.parent/"g13"/study for p in disks())


def update(p,m,v,grad,step,total,peak):
    norm=jnp.sqrt(sum(jnp.vdot(x,x) for x in jax.tree.leaves(grad)))
    grad=jax.tree.map(lambda x:x*jnp.minimum(1.,1.5/jnp.maximum(norm,1e-6)),grad)
    m=jax.tree.map(lambda a,b:.9*a+.1*b,m,grad);v=jax.tree.map(lambda a,b:.98*a+.02*b*b,v,grad)
    t=step.astype(jnp.float32);warm=min(48,total//8)
    frac=jnp.clip((t-warm)/max(1,total-warm),0,1)
    lr=peak*jnp.minimum(t/warm,1)*(.12+.88*.5*(1+jnp.cos(jnp.pi*frac)))
    p=jax.tree.map(lambda a,b,c:a-lr*(b/(1-.9**t)/(jnp.sqrt(c/(1-.98**t))+1e-8)+(.012*a if a.ndim>=2 else 0)),p,m,v)
    return p,m,v


def save_npz(file,d):
    tmp=file.with_suffix(".new.npz");np.savez_compressed(tmp,**{k:np.asarray(x) for k,x in d.items()});tmp.replace(file)


def save(folders,tag,step,p,m,v,config,curve):
    paths=[root/tag/f"step-{step:05d}" for root in folders]
    a=paths[0];a.mkdir(exist_ok=True,parents=True)
    save_npz(a/"weights.npz",p);save_npz(a/"optimizer.npz",{**{"m_"+k:x for k,x in m.items()},**{"v_"+k:x for k,x in v.items()}})
    write(a/"config.json",config)
    checks={k:sha(a/k) for k in ("weights.npz","optimizer.npz","config.json")}
    rec=dict(tag=tag,step=step,sha256=checks,curve=curve)
    for b in paths[1:]:
        b.mkdir(exist_ok=True,parents=True)
        for k,h in checks.items():
            part=b/(k+".copying");shutil.copyfile(a/k,part)
            if sha(part)!=h:raise ValueError("Checkpoint readback mismatch")
            part.replace(b/k)
        write(b/"verified.json",rec)
    write(a/"verified.json",rec)


def load(folders,tag,step,config):
    src=folders[0]/tag/f"step-{step:05d}"
    doc=json.loads((src/"verified.json").read_bytes())
    for folder in folders:
        p=folder/tag/src.name
        if json.loads((p/"verified.json").read_bytes())!=doc or json.loads((p/"config.json").read_bytes())!=config:raise ValueError("Checkpoint recipe mismatch")
        if any(sha(p/k)!=v for k,v in doc["sha256"].items()):raise ValueError("Checkpoint hash mismatch")
    with np.load(src/"weights.npz") as z:params={k:jnp.asarray(z[k]) for k in z.files}
    with np.load(src/"optimizer.npz") as z:
        m={k:jnp.asarray(z["m_"+k]) for k in params};v={k:jnp.asarray(z["v_"+k]) for k in params}
    return params,m,v,doc["curve"]


def complete(folders,tag,step):
    return all((f/tag/f"step-{step:05d}/verified.json").is_file() for f in folders)


def digest(doc):
    return hashlib.sha256(json.dumps(doc,sort_keys=True,separators=(",",":")).encode()).hexdigest()
