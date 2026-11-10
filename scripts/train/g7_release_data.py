"""Build a compact, verified research release from G7 artifacts on Fedora."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.train.g6_artifacts import roots
from scripts.train.g7_readout_report import collect as readouts,read,write
from scripts.train.g7_scale_report import collect as scale
from scripts.train.g7_lock_heads import sha


def status(root):
    clocks=root/"physical-time"; archive=root/"archive-audit.json"
    return {"original_rungs":len(list(root.glob("g7-*/step-*/verified.json"))),
            "physical_readback":read(archive)["verified_rungs"] if archive.exists() else 0,
            "fresh_evaluations":len(list((root/"scale-eval").glob("*/*/*-signals.json"))),
            "alignment_heads":len(list((root/"alignment-audit").glob("*/*/waves.json"))),
            "supervised_tests":len(list((root/"supervised-confirmation").glob("policy*.json"))),
            "policy_tests":len(list((root/"rl").glob("*/test.json"))),"camera_tests":len(list((root/"camera").glob("*/test.json"))),
            "planner_tests":len(list((root/"planning").glob("*/test.json"))),"physical_timing_tests":len(list(clocks.glob("*/*/confirm.json"))),
            "live_examples":sum(len(read(f)) for f in (root/"live").glob("*/previews.json"))}


FINISH={"original_rungs":48,"physical_readback":48,"fresh_evaluations":80,"alignment_heads":8,"supervised_tests":20,
        "policy_tests":4,"camera_tests":6,"planner_tests":4,"physical_timing_tests":12,"live_examples":12}


def timing(root,data,out):
    d={}
    for p in sorted((root/"physical-time").glob("*/*/confirm.json")):
        x=read(p);d[x["tag"]+"/"+x["recipe"]]=x
    source={}
    for f in ("crema","events"):
        card=data/"physical-time"/f/"confirm/card.json"
        if card.is_file():
            c=read(card);source[f]={k:v for k,v in c.items() if k not in ("source_sha256","sha256")}
            source[f]["card_sha256"]=sha(card)
    diffs={}
    for seed in (711,712):
        for core in ("hybrid","gpt2","raw"):
            label=f"{core}-{seed}";folder=root/"physical-time"/label
            if label+"/both-signs" not in d or label+"/older-only" not in d:continue
            for family in ("crema","events"):
                with np.load(folder/"both-signs"/(family+"-predictions.npz")) as a,np.load(folder/"older-only"/(family+"-predictions.npz")) as b:
                    np.testing.assert_array_equal(a["uid"],b["uid"]);groups=a["group"];np.testing.assert_array_equal(groups,b["group"])
                    def wins(v):
                        ties=v==v.max(1,keepdims=True);return ties[:,0]/ties.sum(1)
                    delta=wins(a["scores"])-wins(b["scores"]);g,ix=np.unique(groups,return_inverse=True)
                    counts=np.bincount(ix);w=np.bincount(ix,weights=delta);draw=np.random.default_rng(77163).integers(len(g),size=(2000,len(g)))
                    ci=np.quantile(w[draw].sum(1)/counts[draw].sum(1),[.025,.975])
                    diffs[label+"/"+family]={"balanced_minus_older_only_four_way":float(delta.mean()),"group_ci95":ci.tolist(),"groups":len(g)}
    architectures={}
    for seed in (711,712):
        for first,second in (("raw","hybrid"),("raw","gpt2"),("hybrid","gpt2")):
            for family in ("crema","events"):
                a=root/"physical-time"/f"{first}-{seed}"/"both-signs"/(family+"-predictions.npz")
                b=root/"physical-time"/f"{second}-{seed}"/"both-signs"/(family+"-predictions.npz")
                if not a.is_file() or not b.is_file():continue
                with np.load(a) as aa,np.load(b) as bb:
                    np.testing.assert_array_equal(aa["uid"],bb["uid"]);g,ix=np.unique(aa["group"],return_inverse=True)
                    def win(v):
                        t=v==v.max(1,keepdims=True);return t[:,0]/t.sum(1)
                    change=win(aa["scores"])-win(bb["scores"]);m=np.bincount(ix);z=np.bincount(ix,weights=change)
                    sel=np.random.default_rng(77163).integers(len(g),size=(2000,len(g)))
                    architectures[f"{first}-minus-{second}-{seed}-{family}"]={"accuracy_delta":float(change.mean()),"ci95":np.quantile(z[sel].sum(1)/m[sel].sum(1),[.025,.975]).tolist()}
    write(out/"physical-timing.json",{"source_cards":source,"heads":d,"paired_sign_intervention":diffs,"paired_architectures":architectures})


def camera_pixels(data):
    def load(role):
        path=data/"reacher"/role
        x=np.load(path/"v.npy",mmap_mode="r")[:,:,-1].reshape(-1,256).astype(np.float64)/255
        y=np.load(path/"n.npy",mmap_mode="r")[:,:,-1,:2].reshape(-1,2)*21
        return x,y
    x,y=load("train");v,z=load("test");m=x.mean(0);mu=y.mean(0)
    coef=np.linalg.solve((x-m).T@(x-m)+.001*np.eye(256),(x-m).T@(y-mu))
    return {"current_fingertip_cm_ridge_over_256_pixels":float(np.linalg.norm((v-m)@coef+mu-z,axis=1).mean()),
            "training_mean_baseline_cm":float(np.linalg.norm(mu-z,axis=1).mean()),
            "pixels_with_training_sd_over_one_gray_level":int((x.std(0)*255>1).sum()),"new_starts":len(z)//4}


def main(a):
    root,replica=roots("g7");progress=status(root)
    if a.status:print(json.dumps(progress));return
    if a.final and progress!=FINISH:raise ValueError(f"Incomplete registered panels: {progress}")
    a.out.mkdir(exist_ok=True,parents=True);readouts(root,a.data,a.out);timing(root,a.data,a.out)
    write(a.out/"camera-observability.json",camera_pixels(a.data))
    if a.final:
        scale(root,a.out)
        audit=read(root/"archive-audit.json")
        if read(replica/"archive-audit.json")!=audit:raise ValueError("Physical audit receipts disagree")
        write(a.out/"checkpoint-inventory.json",audit)
    live={}
    for f in (root/"live").glob("*/media-verified.json"):
        d=read(f); previews={}
        for p in sorted((f.parent/"previews").glob("*")):previews[p.name]={"sha256":sha(p),"bytes":p.stat().st_size}
        live[f.parent.name]={**d,"previews":previews}
    write(a.out/"continuous-generation.json",live)
    write(a.out/"release-status.json",progress)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,default=Path("/mnt/slwm-data/SLWM/g7"));p.add_argument("--out",type=Path,default=Path("/mnt/slwm-checkpoints-a/SLWM/g7/public"))
    p.add_argument("--status",action="store_true");p.add_argument("--final",action="store_true");main(p.parse_args())
