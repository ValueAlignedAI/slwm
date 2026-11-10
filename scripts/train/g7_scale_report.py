"""Summarize matched exposure, common measured TPU time, fresh rooms/speakers and intervention contrasts."""
import argparse
from functools import lru_cache
import itertools
import json
from pathlib import Path
import numpy as np
from scripts.train.g7_scale_lock import cases
from scripts.train.g7_readout_report import write
from training.g5_compare import aps_weights


def read(f):return json.loads(f.read_bytes())
def ci(x):return np.quantile(x,[.025,.975]).tolist()
def metrics(m):
    ap=m["impact"]["average_precision"]
    return {"endpoint_px":m["push"]["joint"]["endpoint_px"],"branch_error_px":m["push"]["joint"]["branch_difference_px"],
            "video_endpoint_px":m["push"]["video"]["endpoint_px"],"state_only_px":m["push"]["state"]["endpoint_px"],
            "no_prospective_plan_px":m["push"]["unplanned"]["endpoint_px"],"persistence_px":m["push"]["persistence"]["endpoint_px"],"raw_ridge_px":m["push"]["ridge"]["endpoint_px"],
            "event_ap":ap["correct"],"erased_ap":ap["erase"],"shifted_ap":ap["shift"],"raw_motion_ap":ap["raw_motion"],
            "prevalence":m["impact"]["positive_prevalence"]}


@lru_cache(maxsize=52)
def bootstrap(folder,role):
    with np.load(Path(folder)/(role+"-predictions.npz")) as f:
        y=f["impact_y"];groups=f["impact_group"];sim=f["sim_group"]
        g=len(np.unique(groups));draws=np.random.default_rng(71151).multinomial(g,np.full(g,1/g),size=2000)
        ap=aps_weights(f["impact_correct"],y,groups,draws);raw=aps_weights(f["impact_raw_motion"],y,groups,draws)
        return f["push_joint"],f["branch_joint"],f["push_ridge"],ap,raw,sim,np.unique(groups)


def delta(aa,bb,role):
    a=bootstrap(str(aa),role);b=bootstrap(str(bb),role)
    np.testing.assert_array_equal(a[-2],b[-2]);np.testing.assert_array_equal(a[-1],b[-1])
    sample=np.random.default_rng(71152).integers(len(a[0]),size=(2000,len(a[0])))
    au,bu=(metrics(read(k/(role+"-metrics.json"))) for k in (aa,bb))
    return {"first_minus_second_endpoint_px":float((a[0]-b[0]).mean()),"endpoint_ci95":ci((a[0]-b[0])[sample].mean(1)),
            "first_minus_second_branches_px":float((a[1]-b[1]).mean()),"branches_ci95":ci((a[1]-b[1])[sample].mean(1)),
            "first_minus_second_event_ap":au["event_ap"]-bu["event_ap"],"event_ci95":ci(a[3]-b[3]),
            "first_minus_raw_ridge_px":float((a[0]-a[2]).mean()),"first_minus_raw_motion_ap":au["event_ap"]-au["raw_motion_ap"]}


def common_time(a,b,key):
    if len(a)!=4 or len(b)!=4:return None
    aa=np.array([r[key] for r in a],float);bb=np.array([r[key] for r in b],float)
    if np.any(~np.isfinite(aa)) or np.any(~np.isfinite(bb)) or np.any(np.diff(aa)<=0) or np.any(np.diff(bb)<=0):return None
    stop=float(min(aa[-1],bb[-1]))
    result={"seconds":stop,"first_interpolated_step":float(np.interp(stop,aa,[r["step"] for r in a])),
            "second_interpolated_step":float(np.interp(stop,bb,[r["step"] for r in b]))}
    for metric in ("endpoint_px","event_ap","branch_error_px"):
        result["first_minus_second_"+metric]=float(np.interp(stop,aa,[r["original_dev"][metric] for r in a])-np.interp(stop,bb,[r["original_dev"][metric] for r in b]))
    return result


def collect(root,out):
    out.mkdir(exist_ok=True,parents=True);parent=root.parent/"g6";records={};paths={}
    for tag,terminal in cases():
        fork=tag.startswith("g7-");base=root if fork else parent
        rows=[];wall=0.;update=0.
        for step in ((10240,12288,16384) if fork else (8192,)):
            stage=base/tag/f"step-{step:06d}";fresh=root/"scale-eval"/tag/stage.name
            if not (stage/"verified.json").is_file():continue
            cfg=read(stage/"config.json");v=read(stage/"verified.json");m=read(stage/"dev-metrics.json")
            if m["checkpoint"]!=v["sha256"]["inference.npz"]:raise ValueError(tag)
            if fork:
                prog=read(stage/"progress.json");tim=read(stage/"wall.json")
                update=prog["continuation_update_seconds"]
                wall=wall+tim["training_invocation_s"]+tim["evaluation_invocation_s"] if tim["training_invocation_s"] is not None and wall is not None else None
            row={"step":step,"params":cfg["params"],"examples_total":step*cfg["batch"],"continued_examples":(step-8192)*cfg["batch"],
                 "update_seconds":update,"training_and_evaluation_seconds":wall,"original_dev":metrics(m),"accelerator":m["backend"],
                 "weights":v["sha256"]["inference.npz"],"optimizer":v["sha256"]["resume.npz"],"mirrored_disk_count":len(set(v["disk_uuids"])),
                 "views":{"original_dev":m["examples"]}}
            for role in ("dev","test"):
                if (fresh/(role+"-metrics.json")).exists():
                    fm=read(fresh/(role+"-metrics.json"))
                    if fm["checkpoint"]!=m["checkpoint"]:raise ValueError("Fresh role changed its tested weight")
                    row["fresh_"+role]=metrics(fm);row["views"]["fresh_"+role]=fm["examples"]
                if (fresh/(role+"-signals.json")).is_file():
                    fields=read(fresh/(role+"-signals.json"))
                    if fields["checkpoint"]!=m["checkpoint"]:raise ValueError("Speech evaluator changed weights")
                    row["signals_"+role]=fields["metrics"]
            rows.append(row);paths[(tag,step)]=(stage,fresh)
        if rows:records[tag]=rows
    exposure={};controls={};timed={}
    def compare(name,first,second):
        ac=records.get(first,[]);bc=records.get(second,[])
        if not ac or not bc:return
        af=first.startswith("g7-");bf=second.startswith("g7-")
        ancestor=lambda t:"g6-"+t[3:].rsplit("-",1)[0]
        ax=(records.get(ancestor(first),[])+ac) if af else ac
        bx=(records.get(ancestor(second),[])+bc) if bf else bc
        if af and bf:
            time={key:common_time(ax,bx,key) for key in ("update_seconds","training_and_evaluation_seconds")}
            if all(time.values()):timed[name]={"first":first,"second":second,"matched_time":time}
        aa=ac[-1];bb=bc[-1]
        if aa["step"]!=(16384 if af else 8192) or bb["step"]!=(16384 if bf else 8192):return
        for domain,role,i in (("original_dev","dev",0),("fresh_test","test",1)):
            if domain in aa and domain in bb:
                controls.setdefault(name,{})[domain]=delta(paths[(first,aa["step"])][i],paths[(second,bb["step"])][i],role)
        exposure[name]={"first":first,"second":second,"steps":[aa["step"],bb["step"]]}
    for seed in (511,512):
        for size in ("anchor","larger"):
            for core in ("hybrid","gpt2"):
                prefix=f"v5e-{size}-{core}-{seed}"
                fresh="g7-"+prefix+"-fresh";replay="g7-"+prefix+"-replay";prev="g6-"+prefix
                compare("data-"+prefix,fresh,replay)
                for typ,tag in (("fresh",fresh),("replay",replay)):compare("continuation-"+prefix+"-"+typ,tag,prev)
            for arm in ("fresh","replay"):
                compare(f"architecture-{size}-{seed}-{arm}",f"g7-v5e-{size}-hybrid-{seed}-{arm}",f"g7-v5e-{size}-gpt2-{seed}-{arm}")
        for core,arm in itertools.product(("hybrid","gpt2"),("fresh","replay")):
            compare(f"size-{core}-{seed}-{arm}",f"g7-v5e-larger-{core}-{seed}-{arm}",f"g7-v5e-anchor-{core}-{seed}-{arm}")
    interactions={}
    for seed,core in itertools.product((511,512),("hybrid","gpt2")):
        tags=[f"g7-v5e-{size}-{core}-{seed}-{arm}" for size in ("larger","anchor") for arm in ("fresh","replay")]
        for domain,role,ix in (("original_dev","dev",0),("fresh_test","test",1)):
            if not all((t,16384) in paths and domain in records[t][-1] for t in tags):continue
            vals=[bootstrap(str(paths[(t,16384)][ix]),role) for t in tags]
            for other in vals[1:]:
                np.testing.assert_array_equal(vals[0][-2],other[-2]);np.testing.assert_array_equal(vals[0][-1],other[-1])
            def combine(k):return vals[0][k]-vals[1][k]-vals[2][k]+vals[3][k]
            diff=combine(0);draw=np.random.default_rng(71152).integers(len(diff),size=(2000,len(diff)))
            ap=[records[t][-1][domain]["event_ap"] for t in tags]
            interactions[f"{core}-{seed}-{domain}"]={"fresh_minus_replay_larger_less_anchor_endpoint_px":float(diff.mean()),
                "endpoint_ci95":ci(diff[draw].mean(1)),"event_ap":ap[0]-ap[1]-ap[2]+ap[3],"ap_ci95":ci(combine(3))}
    write(out/"scale-curves.json",{"models":records,"measured":"Cumulative continuation update timer includes first-block compilation. Stage wall sums training plus original evaluation; transport is billed separately in aggregate.",
          "full_continuation_endpoints":sum(v[-1]["step"]==16384 for t,v in records.items() if t.startswith("g7-"))})
    write(out/"matched-contrasts.json",{"labels":exposure,"size_by_data_interactions":interactions,"at_registered_endpoints":controls,"equal_tpu_time":timed,
          "time_interpolation":"Common longest bracketed time, piecewise linear original-development metric interpolation from the same zero-update parent. No extrapolation. Conditional on two training seeds.",
          "paired_intervals":"2,000 whole-scene or room/date resamples; positive event AP and negative endpoint errors favor first; the Tokyo held site contains two rooms."})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();collect(a.root,a.out)
