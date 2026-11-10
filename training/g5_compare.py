"""Compare locked G5 pairs with matched bootstraps over scene IDs and recording dates."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from training.g4_train import write
from training.g5_eval import ap_table


def aps_weights(values,y,group,draws):
    _,inverse=np.unique(group,return_inverse=True)
    order=np.argsort(-values,kind="stable")
    score=values[order]; label=y[order]; ix=inverse[order]
    # With few dates, repeated bootstrap compositions need only be integrated once.
    counts,ids=np.unique(draws,axis=0,return_inverse=True)
    results=np.array([ap_table(score,label,w[ix]) for w in counts])
    return results[ids]


def ci(delta):
    return [float(v) for v in np.quantile(delta,[.025,.975])]


def compare(root,scale,label,role):
    by_seed={}
    digests={}
    passed=True
    joint_win=True
    raw_win=True
    cfgref=None
    for seed in (511,512):
        metrics={}; pred={}; config={}
        for core in ("hybrid","gpt2"):
            name=f"g5-{scale}-{core}-{seed}"+("-"+label if label else "")
            run=root/name
            metrics[core]=json.loads((run/(role+"-metrics.json")).read_bytes())
            config[core]=json.loads((run/"config.json").read_bytes())
            digests[name]=metrics[core]["checkpoint"]
            done=json.loads((run/"complete.json").read_bytes())
            if digests[name]!=done["sha256"]["inference.npz"]:
                raise ValueError("Evaluated file is outside checkpoint lock")
            with np.load(run/(role+"-predictions.npz")) as f:
                pred[core]={k:np.array(f[k]) for k in f}
            m=metrics[core]
            passed &= m["impact"]["average_precision"]["correct"]>m["impact"]["average_precision"]["erase"]
            passed &= m["push"]["joint"]["endpoint_px"]<m["push"]["persistence"]["endpoint_px"]
        h,d=pred["hybrid"],pred["gpt2"]
        for k in ("impact_y","impact_group","sim_group"):
            np.testing.assert_array_equal(h[k],d[k])
        for k in ("params","batch","steps","data","data_seed","code","lr"):
            if config["hybrid"][k]!=config["gpt2"][k]:
                raise ValueError(f"Unmatched comparison: {k}")
        signature={k:config["hybrid"][k] for k in ("params","batch","steps","data","data_seed","code","lr")}
        if cfgref is not None and cfgref!=signature:
            raise ValueError("Unmatched seeds")
        cfgref=signature
        rng=np.random.default_rng(919)
        y=h["impact_y"]
        groups=h["impact_group"]
        g=len(np.unique(groups))
        date_draws=rng.multinomial(g,np.full(g,1/g),size=2000)
        n=len(h["sim_group"])
        sampled=rng.integers(0,n,(2000,n))
        gaps={}; basic={}
        for core,v in pred.items():
            basic[core]={"impact":metrics[core]["impact"]["average_precision"],"push":metrics[core]["push"]}
        for name,other,pk,ik in (("dense",d,"joint","correct"),("raw",h,"ridge","raw_motion"),("negative",h,"persistence","shift")):
            delta=other["push_"+pk]-h["push_joint"]
            gap=float(delta.mean())
            pci=ci(delta[sampled].mean(1))
            denom=float(other["push_"+pk].mean())
            imp_delta=aps_weights(h["impact_correct"],y,groups,date_draws)-aps_weights(other["impact_"+ik],y,groups,date_draws)
            ici=ci(imp_delta)
            a=basic["hybrid"]["impact"]["correct"]
            ref=basic["gpt2" if name=="dense" else "hybrid"]["impact"][ik]
            branch=other["branch_"+pk]-h["branch_joint"]
            gaps[name]={"endpoint_error_improvement_px":gap,"endpoint_paired_ci95":pci,
                        "impact_ap_gain":a-ref,"impact_ap_paired_ci95":ici,"impact_relative":a/ref-1,
                        "endpoint_relative":gap/denom,"branch_difference_improvement_px":float(branch.mean()),
                        "branch_difference_paired_ci95":ci(branch[sampled].mean(1))}
            if name=="dense":
                joint_win &= gap/denom>=.05 and a/ref-1>=.05 and ici[0]>0 and pci[0]>0
            if name=="negative":
                joint_win &= ici[0]>0 and pci[0]>0
            if name=="raw":
                raw_win &= gap>0 and a>ref and ici[0]>0 and pci[0]>0
        by_seed[str(seed)]={"controls":basic,"hybrid_gain":gaps,"examples":metrics["hybrid"]["examples"]}
    return {"scale":scale,"label":label,"split":role,"sha256":digests,"promote":bool(passed),"joint_architecture_win":bool(joint_win),
            "beats_fitted_raw_and_motion":bool(raw_win),"matched":cfgref,"seeds":by_seed}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True,type=Path); p.add_argument("--scale",default="pilot")
    p.add_argument("--label",default=""); p.add_argument("--role",choices=["dev","test"],default="dev")
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    result=compare(a.root,a.scale,a.label,a.role)
    write(a.output,result)
    print(json.dumps({k:result[k] for k in ("scale","label","split","promote","joint_architecture_win","beats_fitted_raw_and_motion")}))
