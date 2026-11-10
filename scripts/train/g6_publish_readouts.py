"""Publish frozen cross-view and prospective closed-loop evidence as each matched pair completes."""
import argparse
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

BASE=Path("experiments/multimodal/t6/tpu-scale-20261110")
BRANCH="training/g5-world-dynamics-v6e-20261110"
REMOTE="/mnt/slwm-checkpoints-a/SLWM/g6"
COLLECT="""
import json
from pathlib import Path
root=Path('/mnt/slwm-checkpoints-a/SLWM/g6')
def read(p): return json.loads(p.read_bytes())
probes={p.stem:read(p) for p in (root/'shared-probes').glob('g6-*.json')}
terminal={}
for p in (root/'terminal').glob('*/decisions.json'):
    d=p.parent
    if (d/'speech-dev.json').is_file() and (d/'weights.json').is_file():
        terminal[d.name]={'decisions':read(p),'speech':read(d/'speech-dev.json'),'weights':read(d/'weights.json')}
print(json.dumps({'probes':probes,'terminal':terminal},separators=(',',':')))
"""


def paired(h,d):
    for k in ("step","seed","scale_params","development_clips","development_speakers","classes"):
        if h[k]!=d[k]:
            raise ValueError("Unmatched frozen comparison")
    if h["development_speakers"]!=4: raise ValueError("Speaker split changed")
    scores={}
    for shots in ("8","32"):
        a=h["shots"][shots]; b=d["shots"][shots]
        group=a["cross_view_by_speaker"]
        if set(group)!=set(b["cross_view_by_speaker"]): raise ValueError("Speaker membership mismatch")
        delta=[group[g]-b["cross_view_by_speaker"][g] for g in sorted(group)]
        rng=random.Random(602048)
        boot=sorted(statistics.mean(rng.choices(delta,k=len(delta))) for _ in range(4000))
        scores[shots]={"hybrid":a["mean_cross_view"],"dense":b["mean_cross_view"],"gap":a["mean_cross_view"]-b["mean_cross_view"],
                       "equal_speaker_mean_gap":statistics.mean(delta),"paired_speaker_ci95":[boot[99],boot[3899]]}
    eight=scores["8"]
    return {"kind":"frozen","params":h["scale_params"],"step":h["step"],"seed":h["seed"],"shots":scores,
            "registered_seed_pass":eight["hybrid"]>.25 and eight["gap"]>=.03 and eight["paired_speaker_ci95"][0]>0,
            "arms":[h,d]}


def write_new(path,doc):
    if path.exists():
        if json.loads(path.read_bytes())!=doc: raise ValueError("Conflicting archived evidence "+path.name)
        return
    tmp=path.with_suffix(".pending"); tmp.write_text(json.dumps(doc,sort_keys=True,indent=2)+"\n"); tmp.replace(path)


def report(path):
    results=[json.loads(p.read_bytes()) for p in sorted(path.glob("*.json"))]
    frozen=[r for r in results if r["kind"]=="frozen"]
    lines=["# Speaker transfer and sustained decisions","","A single fixed ridge head is fitted on eight training examples per command from one modality, then applied to another with no adaptation. The table averages six directional swaps on four development speakers. Balanced chance is 25%. The intervals resample whole speakers.","",
           "| Parameters | Seed | Hybrid | Dense GPT-2 | Hybrid gain, pp | Paired interval, pp |",
           "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in sorted(frozen,key=lambda v:(v["params"],v["seed"])):
        e=r["shots"]["8"]; gen,scale=r["arms"][0]["run"].split("-")[1:3]; seed=r["seed"]
        link=f"supplemental/frozen-{gen}-{scale}-{seed}.json"
        lo,hi=e["paired_speaker_ci95"]
        lines.append(f"| {r['params']:,} | [{seed}]({link}) | {e['hybrid']:.2%} | {e['dense']:.2%} | {e['gap']*100:.2f} | {lo*100:.2f} to {hi*100:.2f} |")
    for size in sorted(set(v["params"] for v in frozen)):
        both={v["seed"]:v for v in frozen if v["params"]==size}
        if set(both)=={511,512}:
            passed=all(v["registered_seed_pass"] for v in both.values())
            lines+=["",f"At {size:,} parameters the two-seed shared-content gate {'passed' if passed else 'was unmet'}. Each linked result includes all nine directional cells, source-specific raw baselines, within-speaker shuffles and retrieval."]
    terminal=[r for r in results if r["kind"]=="terminal"]
    if terminal:
        lines+=["","## Final checkpoints on 32 unseen goal-reaching scenes","","All rules ran 24 successive decisions per scene. Lower final distance is better. The speech diagnostic uses the same frozen recognizer and held development speakers as G5.","",
                "| Parameters | Seed | Hybrid, px | Dense, px | Prior, px | Hybrid reset, px | Dense reset, px |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in sorted(terminal,key=lambda v:(v["decisions"]["params"],v["decisions"]["seed"])):
            v=r["decisions"]; c=v["controls"]; seed=v["seed"]
            loc=f"supplemental/terminal-{v['generation']}-{v['scale']}-{seed}.json"
            lines.append(f"| {v['params']:,} | [{seed}]({loc}) | {c['hybrid']['mean_final_px']:.2f} | {c['gpt2']['mean_final_px']:.2f} | {c['prior']['mean_final_px']:.2f} | {c['hybrid-reset']['mean_final_px']:.2f} | {c['gpt2-reset']['mean_final_px']:.2f} |")
    return "\n".join(lines+["","[Frozen readout protocol](shared-state.md) · [Fresh decision protocol](decisions.md) · [Conditional generation protocol](generation.md)"])+"\n"


def run(a):
    os.chdir(a.repo); dest=BASE/"supplemental"; dest.mkdir(exist_ok=True,parents=True)
    while time.time()<a.deadline:
        try:
            command=["ssh","-o","BatchMode=yes","-o","ConnectTimeout=8","db@192.168.1.200","python3","-"]
            payload=json.loads(subprocess.check_output(command,input=COLLECT,text=True,timeout=40))
            nterm=0; nprobe=0
            for seed in (511,512):
                for gen in ("v5e","v6e"):
                    for scale in ("anchor","larger"):
                        key=f"{gen}-{scale}-{seed}"; names=[f"g6-{gen}-{scale}-{c}-{seed}" for c in ("hybrid","gpt2")]
                        probes=[payload["probes"].get(k) for k in names]
                        def hashes(step):
                            files=[a.control/"receipts"/f"{k}-{step}.json" for k in names]
                            return [json.loads(f.read_bytes())["sha256"]["inference.npz"] for f in files] if all(f.is_file() for f in files) else []
                        if all(probes) and hashes(2048)==[p["inference_sha256"] for p in probes]:
                            write_new(dest/f"frozen-{key}.json",paired(*probes)); nprobe+=1
                        t=payload["terminal"].get(key)
                        if t and t["decisions"]["complete"] and hashes(8192)==[t["weights"][c] for c in ("hybrid","gpt2")]:
                            if t["weights"]!=t["decisions"]["weights_sha256"]: raise ValueError("Speech and decision checkpoints differ")
                            t["kind"]="terminal"; write_new(dest/f"terminal-{key}.json",t); nterm+=1
            if nprobe or nterm:
                md=BASE/"transfer-and-decisions.md"; md.write_text(report(dest))
                paths=[str(md),*[str(p) for p in sorted(dest.glob("*.json"))]]
                if subprocess.check_output(["git","branch","--show-current"],text=True).strip()!=BRANCH: raise ValueError("Study branch changed")
                if subprocess.check_output(["git","status","--porcelain","--",*paths],text=True).strip():
                    subprocess.run(["git","add","--",*paths],check=True)
                    subprocess.run(["git","commit","--only","-m","Publish verified shared-state and live readouts from matched TPU models","--",*paths],check=True)
                env=dict(os.environ,GIT_SSH_COMMAND="ssh -o BatchMode=yes -o UserKnownHostsFile=/Users/db/dev/slwm-g2/artifacts/predictive_world/g2/github_known_hosts -o StrictHostKeyChecking=yes")
                if subprocess.check_output(["git","rev-list","--count","@{upstream}..HEAD"],text=True).strip()!="0":
                    subprocess.run(["git","push","origin",BRANCH],env=env,check=True,timeout=100)
            if nterm>=4 and nprobe>=4: return
        except (OSError,ValueError,KeyError,subprocess.SubprocessError) as e:
            print(type(e).__name__,str(e)[:200],flush=True)
        time.sleep(75)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--repo",required=True,type=Path)
    p.add_argument("--control",required=True,type=Path); p.add_argument("--deadline",required=True,type=float)
    run(p.parse_args())
