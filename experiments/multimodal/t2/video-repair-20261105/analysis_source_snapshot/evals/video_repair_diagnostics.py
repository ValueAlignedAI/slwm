"""Post-hoc heterogeneity and harmful-video analysis; no new selection/gates."""
import argparse
import json
from pathlib import Path
import numpy as np
from training.t2_real_av import write_json,cluster_interval
from training.t2_prepare_cremad import sha256


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args(); cfg=json.loads(Path(a.config).read_text()); root=Path(cfg["artifacts"])
    selection=json.loads((root/"selection.json").read_text()); result={"scope":"Post-hoc descriptive analysis; no model, selection, threshold or primary gate changes","source_sha256":sha256(Path(__file__)),"splits":{}}
    lines=["# T2-V1 post-hoc failure analysis","","These diagnostics explain the frozen result. They do not change model selection, primary gates or the fixed listening subset. Positive control harm below means that condition performs worse than the validation-selected additional audio branch.",""]
    plots={}
    for split in ("crema_test","rav_test"):
        result["splits"][split]={}; plots[split]={}; lines += ["## "+split,"", "| Seed / input video | Relative MSE harm vs selected audio [person-cluster95%] | Mean gate |","|---|---:|---:|"]
        for seed in ("61","62"):
            video=json.loads((root/selection["primary_video"][seed]/(split+"_gap8.json")).read_text()); audio=json.loads((root/selection["audio"][seed]/(split+"_gap8.json")).read_text())
            vr,ar=video["per_clip"],audio["per_clip"]; assert [r["sample_id"] for r in vr]==[r["sample_id"] for r in ar]
            baseline=np.array([r["correct_hole_mse"] for r in ar]); actors=[r["actor"] for r in vr]; modes={}
            for mode in ("correct","wrong","shift_plus","shift_minus","static","occluded","gap_shuffle","no_video"):
                loss=np.array([r[mode+"_hole_mse"] for r in vr]); harm=(loss-baseline)/baseline.clip(1e-9); interval=cluster_interval(harm,actors); gate=video["metrics"][mode+"_mean_gate"]
                modes[mode]={"mean_per_clip_relative_harm":float(harm.mean()),"person_cluster_95_interval":interval,"mean_gate":gate}
                if mode in ("correct","wrong","shift_plus","shift_minus","static"):
                    lines.append(f"| {seed} / {mode} | {100*harm.mean():.2f}% [{100*interval[0]:.2f}, {100*interval[1]:.2f}] | {gate:.3f} |")
            per_person={}
            loss=np.array([r["correct_hole_mse"] for r in vr]); gains=(baseline-loss)/baseline.clip(1e-9)
            for person in sorted(set(actors)):
                use=np.asarray(actors)==person; per_person[person]={"clips":int(use.sum()),"video_gain_vs_audio":float(gains[use].mean()),"video_mse":float(loss[use].mean()),"audio_mse":float(baseline[use].mean())}
            result["splits"][split][seed]={"control_harm":modes,"per_person":per_person}; plots[split][seed]=per_person
        lines += ["","All people, including failures (descriptive averages; no within-person independence claim):","","| Person | Seed61 video gain vs audio | Seed62 video gain vs audio |","|---|---:|---:|"]
        for person in plots[split]["61"]:
            lines.append(f"| {person} | {100*plots[split]['61'][person]['video_gain_vs_audio']:.2f}% | {100*plots[split]['62'][person]['video_gain_vs_audio']:.2f}% |")
        lines += [""]
    lines += ["A difference between correct and shifted video establishes sensitivity, not safe rejection of bad input. Compare every shifted/wrong condition against the strong audio baseline: a positive harm means the trained confidence gate has allowed damaging corrections. The exact no-video fallback alone does not resolve this. Calibrating rejection on training/validation mismatches is a future hypothesis; no threshold is tuned on these test actors.","","The main report retains all architecture ablations,160/480ms gaps, all-clip diagnostics, original speech, silence, interpolation and the reference-spectrum reconstruction. Those comparisons are needed before attributing limitations solely to fusion or model size."]
    write_json(root/"failure_analysis.json",result); (root/"failure_analysis.md").write_text("\n".join(lines)+"\n")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout="constrained")
    for ax,split in zip(axes,("crema_test","rav_test")):
        people=list(plots[split]["61"]); y=np.arange(len(people))
        for seed,offset,color in (("61",-.10,"#225b8f"),("62",.10,"#b16429")):
            ax.scatter([100*plots[split][seed][person]["video_gain_vs_audio"] for person in people],y+offset,label="Seed "+seed,color=color,s=26)
        ax.set_yticks(y,people); ax.invert_yaxis(); ax.axvline(0,color="black",lw=.8); ax.axvline(5,color="gray",ls="--",lw=.8); ax.set_title(split); ax.set_xlabel("Video reduction in gap MSE (%)"); ax.grid(axis="x",alpha=.2); ax.legend(loc="best")
    fig.suptitle("Generalization varies by person\nPost-hoc descriptive means ·320ms gaps",fontsize=13); fig.savefig(root/"effects_by_person.png",dpi=180); plt.close(fig)
    print(json.dumps({"posthoc_failure_analysis":str(root/"failure_analysis.md")}))


if __name__=="__main__": main()
