"""Paired, person-clustered decisions; never selects on test performance."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from training.t2_real_av import write_json,cluster_interval
from training.t2_prepare_cremad import sha256


def paired(new,old):
    a,b=new["per_clip"],old["per_clip"]
    if [x["sample_id"] for x in a]!=[x["sample_id"] for x in b]: raise ValueError("Paired test IDs differ")
    nv=np.array([r["correct_hole_mse"] for r in a]); ov=np.array([r["correct_hole_mse"] for r in b]); gains=(ov-nv)/ov.clip(1e-9)
    interval=cluster_interval(gains,[r["actor"] for r in a]); mean=float(gains.mean())
    return {"mean_per_clip_gain":mean,"person_cluster_95_interval":interval,"aggregate_mse_reduction":float(1-nv.mean()/ov.mean()),
        "passes_5percent_positive_interval":bool(mean>=.05 and interval[0]>0)}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args(); cfg=json.loads(Path(a.config).read_text()); root=Path(cfg["artifacts"])
    selection=json.loads((root/"selection.json").read_text()); primary={}; gates=[]
    def load(run,split,gap=8): return json.loads((root/run/(split+f"_gap{gap}.json")).read_text())
    for split in ("crema_test","rav_test"):
        primary[split]={}
        for seed in ("61","62"):
            video=load(selection["primary_video"][seed],split); audio=load(selection["audio"][seed],split); anchor=load(selection["anchor"][seed],split)
            comparisons={"vs_audio":paired(video,audio),"vs_anchor":paired(video,anchor)}; controls=video["control_gains"]
            for v in comparisons.values(): gates.append(v["passes_5percent_positive_interval"])
            for v in controls.values():
                v["passes_2percent_positive_interval"]=bool(v["mean_per_clip_gain"]>=.02 and v["person_cluster_95_interval"][0]>0); gates.append(v["passes_2percent_positive_interval"])
            primary[split][seed]={"samples":video["samples"],"excluded_short":video["excluded_short"],"audio_run":selection["audio"][seed],"video_run":selection["primary_video"][seed],
                "metrics":{"video_hole_mse":video["metrics"]["correct_hole_mse"],"audio_hole_mse":audio["metrics"]["correct_hole_mse"],"anchor_hole_mse":anchor["metrics"]["correct_hole_mse"],"interpolation_hole_mse":video["metrics"]["interpolation_hole_mse"]},
                "paired_comparisons":comparisons,"motion_controls":controls}
    table=[]
    for run in cfg["anchors"]+cfg["runs"]:
        complete=json.loads((root/run["id"]/"training_complete.json").read_text()); counts=json.loads((root/run["id"]/"counts.json").read_text())
        row={"id":run["id"],"arm":run.get("arm","anchor"),"seed":run["seed"],"selected_step":complete["selected_step"],"steps":complete["steps"],"positive_windows":complete["positive_windows"],
            "wall_seconds":complete["wall_seconds"],"counts":counts,"checkpoint_sha256":complete["checkpoint_sha256"],"test":{}}
        for split in ("crema_test","rav_test"):
            row["test"][split]={str(g):load(run["id"],split,g)["metrics"]["correct_hole_mse"] for g in (4,8,12)}
        table.append(row)
    wer=json.loads((root/"exploration/intelligibility.json").read_text()); feature=bool(all(gates)); practical=bool(wer["practical_video_gate_passes"])
    result={"sprint":"T2-V1","feature_video_gate_passes":feature,"practical_speech_video_gate_passes":practical,"video_claim_supported":feature and practical,
        "decision":("Video passes the registered feature and speech diagnostic gates" if feature and practical else "Video improves held-out gap-spectrum prediction, but useful speech repair is not yet established; retain the visual artifact and resolve the speech/decoder bottleneck" if feature else "Video fails the registered contribution gate; retain trained audio controls and report the negative or mixed result"),
        "primary":primary,"runs":table,"matrix_training_wall_minutes":sum(r["wall_seconds"] for r in table)/60,"wer":wer["summary"],
        "selection_sha256":sha256(root/"selection.json"),"report_source_sha256":sha256(Path(__file__)),"scope":"Offline supervised speech repair on constrained corpora; signal proxies, no general world-model or novel-vocabulary claim"}
    write_json(root/"summary.json",result)
    fmt=lambda v:f"{v:.5f}"
    lines=["# T2-V1 — Video versus audio for true-gap speech repair", "",result["decision"]+".","",
        f"Eleven registered RTX 3090 runs; {result['matrix_training_wall_minutes']:.2f} measured GPU-process minutes including validation. Checkpoints selected on equal-corpus validation only; audio61 comparison is {selection['audio']['61']}.","",
        "The speech expert is frozen. A new audio anchor adapts it to honest waveform gaps; the visual branch then learns only bounded residual corrections on that frozen anchor. Video uses original96×64 mouth pixels, differences, a temporal CNN, local cross-attention and observed-only confidence. Audio controls have approximately matched graph-connected trainable capacity and the same positive windows. Video receives additional negative examples and therefore more compute.","",
        "Every input STFT window touching hidden samples is removed. Actual WAV repairs erase originals before any transform and preserve observed PCM samples exactly. This is a new waveform-passthrough task, distinct from the older full-synthesis protocol.","",
        "## Primary320ms results", "", "Relative effects are means of paired per-clip reductions, with2,000-draw person-cluster95% intervals. Positive means video improves the baseline. The registered gate needs≥5% and a positive interval in both seeds on both sets.","",
        "| Test | Seed | Audio MSE | Video MSE | Video gain vs audio [95%] | vs anchor [95%] |", "|---|---:|---:|---:|---:|---:|"]
    for split,seeds in primary.items():
        for seed,row in seeds.items():
            def effect(key):
                x=row["paired_comparisons"][key]; lo,hi=x["person_cluster_95_interval"]; return f"{100*x['mean_per_clip_gain']:.2f}% [{100*lo:.2f}, {100*hi:.2f}]"
            lines.append(f"| {split} (n={row['samples']}) | {seed} | {fmt(row['metrics']['audio_hole_mse'])} | {fmt(row['metrics']['video_hole_mse'])} | {effect('vs_audio')} | {effect('vs_anchor')} |")
    lines += ["","## Temporal and pair controls","","Correct video must beat each control by≥2% with a positive interval. Wrong video is the same person saying a different utterance. Shifts use real±240ms frames, with length≥50 eligibility. Missing video returns the frozen audio anchor exactly; that fallback is not claimed equal to the stronger two-branch audio control.","", "| Test / seed | Wrong | +240ms | −240ms | Static |", "|---|---:|---:|---:|---:|"]
    for split,seeds in primary.items():
        for seed,row in seeds.items():
            effects=[]
            for control in ("wrong","shift_plus","shift_minus","static"):
                x=row["motion_controls"][control]; lo,hi=x["person_cluster_95_interval"]; effects.append(f"{100*x['mean_per_clip_gain']:.2f}% [{100*lo:.2f}, {100*hi:.2f}]")
            lines.append(f"| {split}/{seed} | "+" | ".join(effects)+" |")
    lines += ["","## All trained artifacts","","Total count includes the frozen speech expert/anchor, all frontend/decoder weights and inactive auxiliary heads. Active trainable counts measure graph participation in the first optimizer update; no inactive padding is used to claim a capacity match.","", "| ID | Variant | Selected / total updates | Active trainable | Total | Minutes | C320 MSE | R320 MSE |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in table: lines.append(f"| {r['id']} | {r['arm']} | {r['selected_step']} / {r['steps']} | {r['counts']['graph_connected_trainable']:,} | {r['counts']['total']:,} | {r['wall_seconds']/60:.2f} | {r['test']['crema_test']['8']:.5f} | {r['test']['rav_test']['8']:.5f} |")
    lines += ["","Full per-clip metrics, confidence gates,160/480ms secondary results, actual PCM outputs, optimizer snapshots, configs and hashes are retained alongside this report. EXP-T2-619 has twice the update exposure; it is a stronger audio control, not data-budget matched.","","## Speech diagnostic","","Pinned Whisper-tiny.en (37,760,256 external evaluation parameters) evaluates the32 fixed examples. Its text is never SLWM output or training supervision. Mean per-clip WER can exceed100%. The speech gate needs≥5% relative WER reduction and positive person-cluster intervals against both selected audio and silence, in both seeds and both sets. Oracle spectrum is an evaluation-only decoder ceiling: true clean mel, observed-only phase, never a model input or candidate.","", "| Test | Original | Silence | Interpolation | Oracle spectrum | Audio61 | Video61 | Audio62 | Video62 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split,w in wer["summary"].items():
        m=w["mean_clip_wer"]; lines.append(f"| {split} (n={w['samples']}) | "+" | ".join(f"{100*m[c]:.2f}%" for c in ("reference","silence","interpolation","oracle_spectrum","audio61","video61","audio62","video62"))+" |")
    lines += ["",f"Feature gate: **{'PASS' if feature else 'FAIL'}**. Speech diagnostic gate: **{'PASS' if practical else 'FAIL'}**.","",
        "## Data, limitations and next decision","",
        "CREMA:5,061 training clips/62people,210 validation/7people,240 reused test/8people. Fresh RAVDESS:240 training clips from07,08,19,20;128 validation clips from earlier R1 test actors03,04,05,06,15,16,17,18;128 new test clips from09,10,11,12,21,22,23,24. Only length≥50 clips enter the main shifted-control comparison; exclusions and all-clip diagnostic scores are saved. Both corpora use acted speech and restricted phrases. RAV is new people in a partly trained domain, not new vocabulary. All normalizers/PCA stay fixed from CREMA training. No external generator supplies trained outputs.","",
        "The frozen expert includes the previous CREMA/Libri-trained R2 core and auxiliary content readout; this cycle is additional adaptation, not training from scratch. The visual branch is a diagnostic audio-conditioned correction head, not a newly demonstrated shared multimodal world model or learned output policy. Confidence and no-video fallback are engineering safeguards, not calibrated uncertainty or evidence of understanding.","",
        "Keep the true-gap masking, exact observation preservation, trained checkpoints and ablation record. A failed video gate calls for a new preregistered hypothesis and genuinely new holdout before claiming improvement; increasing parameter count alone is not justified. See the protocol and training guide for reproduction and license limits."
    ]
    (root/"report.md").write_text("\n".join(lines)+"\n")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout="constrained")
    for ax,split in zip(axes,("crema_test","rav_test")):
        labels=[]; values=[]; lower=[]; upper=[]
        for seed in ("61","62"):
            for key,label in (("vs_audio","extra audio"),("vs_anchor","audio anchor")):
                x=primary[split][seed]["paired_comparisons"][key]; v=100*x["mean_per_clip_gain"]; lo,hi=np.array(x["person_cluster_95_interval"])*100
                labels.append(seed+" vs "+label); values.append(v); lower.append(v-lo); upper.append(hi-v)
        pos=np.arange(4); ax.errorbar(values,pos,xerr=[lower,upper],fmt="o",capsize=4,color="#225b8f"); ax.axvline(0,color="black",lw=.8); ax.axvline(5,color="#9b652c",ls="--",label="5% gate")
        ax.set_yticks(pos,labels); ax.invert_yaxis(); ax.set_xlabel("Video reduction in gap MSE (%)"); ax.set_title(split+" ·320ms"); ax.grid(axis="x",alpha=.2); ax.legend(loc="best")
    fig.suptitle("Does video improve repair beyond audio?\nPaired per-clip effects; person-cluster95% intervals",fontsize=13)
    fig.savefig(root/"video_vs_audio.png",dpi=180); plt.close(fig)
    print(json.dumps({"feature_gate":feature,"speech_gate":practical,"minutes":result["matrix_training_wall_minutes"]}))


if __name__=="__main__": main()
