"""Paired transfer results and measured run cost, including failed comparisons."""
from collections import defaultdict
from pathlib import Path
import numpy as np

from training.joint_data import read
from training.joint_g2_anchor import HASHES
from training.joint_train import atomic


def effects(candidate, reference, key, other_key, task):
    """Pair exact items, resample speakers for FSDD and examples within class elsewhere."""
    with np.load(candidate,allow_pickle=False) as a, np.load(reference,allow_pickle=False) as b:
        for field in ("labels","ids","groups"):
            if not np.array_equal(a[field],b[field]):
                raise ValueError("Cannot pair different observations")
        y,group = a["labels"],a["groups"]
        delta = (a[key]==y).astype(float)-(b[other_key]==y).astype(float)
    rng = np.random.default_rng(20261109)
    classes = np.unique(y)
    if task=="fsdd":
        per_group = [np.mean([delta[(group==g)&(y==k)].mean() for k in classes]) for g in np.unique(group)]
        draws = rng.choice(per_group,(2000,len(per_group))).mean(1)
    else:
        draws = np.zeros(2000)
        for k in classes:
            vals = delta[y==k]
            draws += rng.choice(vals,(2000,len(vals))).mean(1)/len(classes)
    return {"gain":float(np.mean([delta[y==k].mean() for k in classes])),
            "paired_95_interval":np.quantile(draws,[.025,.975]).tolist(),"groups":len(np.unique(group))}


def summarize(cfg,step,role):
    root = Path(cfg["output"])
    models = [(n,s) for s in cfg["seeds"] for n in cfg["profiles"]]
    tasks = ["fsdd","fashion","banking"]
    rows, comparisons = {},{}
    text = ["# G3 joint transfer", "", f"Fixed {step:,}-update {role} comparison. Each row uses the same 32 labels per class within task.", "",
            "| Profile | Seed | Spoken digits | Fashion images | Banking intents | Text NLL | Active training min |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    def result(n,s,p):
        return root/f"{n}-{s}"/f"eval-{p}"
    for n,s in models:
        run = root/f"{n}-{s}"
        res = read(result(n,s,step)/f"{role}.json")
        card = read(run/f"step-{step}.json")
        acc = {t:res["tasks"][t]["scores"]["ordered-32"]["balanced_accuracy"] for t in tasks}
        rows[run.name] = {"balanced_accuracy":acc,"last":{t:res["tasks"][t]["scores"]["last-32"]["balanced_accuracy"] for t in tasks},
            "text_nll":res["diagnostics"]["language"]["nll"],"retrieval":res["diagnostics"]["retrieval"],
            "parameters":card["accounting"],"seconds":card["active_train_seconds"],"exposure_sha256":card["exposure_sha256"]}
        text.append(f"| {n} | {s} | "+" | ".join(f"{acc[t]:.1%}" for t in tasks)+f" | {rows[run.name]['text_nll']:.3f} | {card['active_train_seconds']/60:.1f} |")
    control_rows = {}
    text += ["", "| Frozen control | Seed | Spoken digits | Fashion images | Banking intents |",
             "|---|---:|---:|---:|---:|"]
    controls = [("raw",0,"raw")]+[(f"{n}",s,"ordered") for n,s in models]+[(n,0,"g2") for n in HASHES]
    for name,seed,kind in controls:
        item = read(result(name,seed,0)/f"{role}.json")
        values = {t:item["tasks"][t]["scores"][kind+"-32"]["balanced_accuracy"] for t in tasks}
        control_rows[f"{name}-{seed}"] = {"balanced_accuracy":values,"diagnostics":item["diagnostics"]}
        text.append(f"| {name} | {seed} | "+" | ".join(f"{values[t]:.1%}" for t in tasks)+" |")
    for s in cfg["seeds"]:
        hashes = {rows[f"{n}-{s}"]["exposure_sha256"] for n in cfg["profiles"]}
        if len(hashes)!=1:
            raise ValueError("Mismatched exposure across model variants")
        for n in ("signal-dense","signal-sparse","signal-wide"):
            baseline = "gpt2-wide" if n=="signal-wide" else "gpt2-dense"
            refs = [(baseline,s,step,"ordered"),("raw",0,0,"raw"),(n,s,0,"ordered")]
            if n=="signal-sparse":
                refs += [("gpt2-wide",s,step,"ordered"),("signal-wide",s,step,"ordered"),("signal-dense",s,step,"ordered")]
            for ref,rs,rp,kind in refs:
                label = f"{n}-{s}_vs_{ref}-{rs}-step{rp}"
                comparisons[label] = {t:effects(result(n,s,step)/f"{t}-{role}-predictions.npz",
                    result(ref,rs,rp)/f"{t}-{role}-predictions.npz","ordered-32",kind+"-32",t) for t in tasks}
    gates = {}
    for n in ("signal-dense","signal-wide","signal-sparse"):
        ref = "gpt2-wide" if n=="signal-wide" else "gpt2-dense"
        pass_by_task, worst = {},[]
        for task in tasks:
            diffs = []
            for s in cfg["seeds"]:
                bases = [f"{ref}-{s}-step{step}","raw-0-step0",f"{n}-{s}-step0"]
                if n=="signal-sparse":
                    bases += [f"gpt2-wide-{s}-step{step}",f"signal-dense-{s}-step{step}",f"signal-wide-{s}-step{step}"]
                diffs += [comparisons[f"{n}-{s}_vs_{b}"][task]["gain"] for b in bases]
            pass_by_task[task] = min(diffs)>0
            worst.append(min(diffs))
        gates[n] = {"gains_on_tasks_both_seeds":pass_by_task,"passes":sum(pass_by_task.values())>=2 and min(worst)>=-.02}
    text += ["", "Probes fit only labelled support. The few-shot curves, raw and unadapted controls, paired uncertainty, G2 anchor and retrieval are available in the metric files. Development outcomes remain separate from endpoint confirmation. FSDD has two final speakers.", ""]
    out = root/f"summary-{step}-{role}.json"
    atomic(out,{"stage":step,"role":role,"rows":rows,"controls":control_rows,"comparisons":comparisons,"gates":gates})
    out.with_suffix(".md").write_text("\n".join(text))
