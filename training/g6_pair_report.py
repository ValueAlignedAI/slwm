"""Compare two independently archived, same-chip development checkpoints on identical observations."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from training.g5_compare import aps_weights,ci


def load(root,tag,step):
    stage=root/tag/f"step-{step:06d}"
    cfg=json.loads((stage/"config.json").read_bytes())
    rec=json.loads((stage/"verified.json").read_bytes())
    met=json.loads((stage/"dev-metrics.json").read_bytes())
    if rec["run"]!=tag or rec["step"]!=step or met["split"]!="dev" or met["checkpoint"]!=rec["sha256"]["inference.npz"] or len(set(rec["disk_uuids"]))!=2:
        raise ValueError("Endpoint or physical replica identity failed")
    with (stage/"dev-predictions.npz").open("rb") as fd:
        if hashlib.file_digest(fd,"sha256").hexdigest()!=rec["sha256"]["dev-predictions.npz"]:
            raise ValueError("Development rows changed")
    with np.load(stage/"dev-predictions.npz") as f:
        data={k:np.array(v) for k,v in f.items()}
    return cfg,rec,met,data


def compare(root,tags,step,kind):
    first,second=(load(root,t,step) for t in tags)
    ac,ar,am,a=first; bc,br,bm,b=second
    for key in ("batch","data","data_seed","seed","lr","steps","prior_sha256","tpu","code"):
        if ac[key]!=bc[key]:
            raise ValueError("Unmatched scientific recipe: "+key)
    if kind=="architecture" and (ac["params"]!=bc["params"] or ac["spec"]["core"]==bc["spec"]["core"]):
        raise ValueError("Architecture controls must be equal size and separate backbones")
    if kind=="size" and not (ac["params"]>bc["params"] and ac["spec"]["core"]==bc["spec"]["core"]):
        raise ValueError("Size controls must use the same backbone")
    if am["backend"]!=bm["backend"]:
        raise ValueError("Hardware mismatch")
    for key in ("sim_group","impact_group","impact_y"):
        np.testing.assert_array_equal(a[key],b[key])
    rng=np.random.default_rng(919)
    n=len(a["sim_group"]); sampled=rng.integers(0,n,(2000,n))
    y=a["impact_y"]; dates=a["impact_group"]; g=len(np.unique(dates))
    draws=rng.multinomial(g,np.full(g,1/g),size=2000)
    def gap(other,action,onset,apref):
        diff=other["push_"+action]-a["push_joint"]
        branch=other["branch_"+action]-a["branch_joint"]
        aps=aps_weights(a["impact_correct"],y,dates,draws)-aps_weights(other["impact_"+onset],y,dates,draws)
        ref=other["push_"+action].mean()
        return {"action_error_reduction_px":float(diff.mean()),"action_paired_ci95":ci(diff[sampled].mean(1)),
                "relative_action_improvement":float(diff.mean()/ref),
                "branch_difference_reduction_px":float(branch.mean()),"branch_paired_ci95":ci(branch[sampled].mean(1)),
                "contact_ap_gain":float(am["impact"]["average_precision"]["correct"]-apref),
                "contact_ap_paired_ci95":ci(aps),"relative_contact_improvement":float(am["impact"]["average_precision"]["correct"]/apref-1)}
    matched=gap(b,"joint","correct",bm["impact"]["average_precision"]["correct"])
    raw=gap(a,"ridge","raw_motion",am["impact"]["average_precision"]["raw_motion"])
    neg=gap(a,"persistence","shift",am["impact"]["average_precision"]["shift"])
    good=(matched["relative_action_improvement"]>=.05 and matched["relative_contact_improvement"]>=.05 and
          matched["action_paired_ci95"][0]>0 and matched["contact_ap_paired_ci95"][0]>0 and
          neg["action_paired_ci95"][0]>0 and neg["contact_ap_paired_ci95"][0]>0)
    return {"comparison":kind,"split":"dev","seed":ac["seed"],"step":step,"batch":ac["batch"],
            "examples_per_arm":step*ac["batch"],"positions_per_arm":step*ac["batch"]*64,"device":am["backend"],
            "observations":am["examples"],"first_minus_second":matched,"first_vs_raw":raw,"first_vs_negative":neg,
            "passes_both_endpoints_at_this_seed":bool(good),"arms":[{"run":c["run"],"parameters":c["params"],
                "tpu":c["tpu"],"action":m["push"],"contact":m["impact"],"inference_sha256":r["sha256"]["inference.npz"],
                "resume_sha256":r["sha256"]["resume.npz"],"optimizer_bytes":r["optimizer_bytes"],
                "inference_bytes":r["inference_bytes"],"verified_disk_uuids":r["disk_uuids"]}
                for c,r,m,_ in (first,second)]}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--root",type=Path,required=True)
    p.add_argument("--first",required=True); p.add_argument("--second",required=True); p.add_argument("--step",type=int,required=True)
    p.add_argument("--kind",choices=["architecture","size"],required=True)
    a=p.parse_args()
    print(json.dumps(compare(a.root,(a.first,a.second),a.step,a.kind),indent=2,sort_keys=True))
