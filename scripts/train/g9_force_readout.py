"""Check frozen 116M contact readouts against the original uncapped three-axis forces."""
import argparse
import json
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from evals.g9_transfer import contact_features,records,ridge
from models.g9_shared import Spec
from scripts.train.g9_fit import numpy_weights,save_json


def force(path,expected):
    with np.load(path) as f:
        if list(f["names"])!=list(expected):raise ValueError("External trial order mismatch")
        offsets=np.cumsum(np.r_[0,f["lengths"]]);values=f["force"]
        return np.concatenate([values[a+1:b] for a,b in zip(offsets,offsets[1:])]),np.concatenate([values[a:b-1] for a,b in zip(offsets,offsets[1:])])


def main(root,bank,out):
    out.mkdir(parents=True,exist_ok=True);cal=records(bank/"roma-cal.npz");held=records(bank/"roma-held.npz")
    y,_=force(bank/"force-cal-uncapped.npz",cal);yh,now=force(bank/"force-held-uncapped.npz",held)
    wanted={f"confirm-{core}-z{field}-gelu-s{seed}-fresh" for core in ("hybrid","gpt2") for field in (128,512) for seed in (911,912)}
    while wanted:
        progressed=False
        for tag in sorted(wanted):
            stage=root/tag/"step-04096";dest=out/(tag+".json")
            if dest.exists():wanted.remove(tag);progressed=True;continue
            if not (stage/"verified.json").is_file():continue
            meta=json.loads((stage/"receipt.json").read_text());spec=Spec(**meta["spec"])
            p=jax.tree.map(jnp.asarray,numpy_weights(stage/"weights.npz"))
            (x,raw,_,_),_=contact_features(p,spec,cal);(xh,rh,_,_),groups=contact_features(p,spec,held)
            guesses=dict(frozen=ridge(x,y,xh),raw=ridge(raw,y,rh),constant=np.broadcast_to(y.mean(0),yh.shape),force_persistence=now)
            scenes=sorted(set(groups));result={k:[float(np.sqrt(np.mean((value[groups==s]-yh[groups==s])**2))) for s in scenes] for k,value in guesses.items()}
            save_json(dest,dict(tag=tag,sha256=meta["inference_sha256"],scenes=scenes,uncapped_rmse_newtons=result))
            print(tag,round(np.mean(result["frozen"]),3),flush=True);wanted.remove(tag);progressed=True
        if not progressed:time.sleep(15)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("root",type=Path);ap.add_argument("bank",type=Path);ap.add_argument("out",type=Path)
    a=ap.parse_args();main(a.root,a.bank,a.out)
