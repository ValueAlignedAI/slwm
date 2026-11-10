"""Frozen primary gates, coverage/harm accounting, and descriptive artifacts."""
import argparse
import json
from pathlib import Path
import numpy as np
from training.t2_real_av import write_json,cluster_interval
from training.research_sources import file_sha


def interval(values,actors):
    return {"mean":float(np.mean(values)),"person_cluster_95_interval":cluster_interval(np.asarray(values),actors)}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/t2/content_sync.json"); a=p.parse_args(); cfg=json.loads(Path(a.config).read_text()); root=Path(cfg["artifacts"])
    selection=json.loads((root/"selection.json").read_text()); calibration=json.loads((root/"calibration.json").read_text()); probe=json.loads((root/"word_evaluation_v2/probe_summary.json").read_text()); asr=json.loads((root/"exploration/intelligibility.json").read_text()); results={}; content_gates=[]; reject_gates=[]; video_gates=[]
    corrected=json.loads((root/"word_evaluation_v2/head_metrics.json").read_text())
    def load(run):
        value=json.loads((root/run/"test_gap8.json").read_text()); value["metrics"].update(corrected[run]); return value
    for seed,plain,content in (("71","EXP-T2-701","EXP-T2-702"),("72","EXP-T2-703","EXP-T2-704")):
        old,new=load(plain),load(content); aa=load(cfg["audio_by_seed"][seed]); vv=load(cfg["video_by_seed"][seed]); actors=[r["actor"] for r in vv["per_clip"]]
        for data in (old,new,aa): assert [r["sample_id"] for r in data["per_clip"]]==[r["sample_id"] for r in vv["per_clip"]]
        om=np.array([r["correct_hole_mse"] for r in old["per_clip"]]); nm=np.array([r["correct_hole_mse"] for r in new["per_clip"]]); am=np.array([r["correct_hole_mse"] for r in aa["per_clip"]]); vm=np.array([r["correct_hole_mse"] for r in vv["per_clip"]])
        cg=interval((om-nm)/om.clip(1e-9),actors); word=probe["comparisons"][content+"_vs_"+plain]["all"]; clean_ratio=new["metrics"]["clean_mse"]/max(old["metrics"]["clean_mse"],1e-9)
        content_pass=bool(cg["mean"]>=.03 and cg["person_cluster_95_interval"][0]>0 and clean_ratio<=1.05 and word["passes_5pp_positive_interval"]); content_gates.append(content_pass)
        vg=interval((am-vm)/am.clip(1e-9),actors); video_pass=bool(vg["mean"]>=.03 and vg["person_cluster_95_interval"][0]>0); video_gates.append(video_pass)
        coverage=interval([r["correct_accepted"] for r in vv["per_clip"]],actors); controls={}; reject_pass=coverage["mean"]>=.5 and calibration[cfg["video_by_seed"][seed]]["coverage_gate_passes"]
        for mode in ("shift6","shift-6","shift4","shift-4","wrong","static","blank","missing"):
            accept=interval([r[mode+"_accepted"] for r in vv["per_clip"]],actors); loss=np.array([r[mode+"_hole_mse"] for r in vv["per_clip"]]); raw=np.array([r[mode+"_raw_hole_mse"] for r in vv["per_clip"]]); harm=interval((loss-am)/am.clip(1e-9),actors); rawharm=interval((raw-am)/am.clip(1e-9),actors)
            passed=bool(accept["mean"]<=.05 and accept["person_cluster_95_interval"][1]<=.1 and harm["mean"]<=.01 and harm["person_cluster_95_interval"][1]<=.03)
            controls[mode]={"acceptance":accept,"relative_harm_vs_strong_audio":harm,"raw_relative_harm_vs_strong_audio":rawharm,"passes_rejection_condition":passed}
            if mode in ("shift6","shift-6"): reject_pass=reject_pass and passed
        reject_gates.append(bool(reject_pass)); results[seed]={"samples":len(actors),"content_gain":cg,"field_word_readout_gain":word,"clean_mse_ratio":clean_ratio,"content_gate_passes":content_pass,"video_gain":vg,"video_gate_passes":video_pass,"aligned_coverage":coverage,"controls":controls,"rejection_gate_passes":bool(reject_pass),"metrics":{"plain":old["metrics"],"content":new["metrics"],"audio":aa["metrics"],"video":vv["metrics"]}}
    table=[]
    for run in cfg["base_runs"]+cfg["candidate_runs"]:
        complete=selection[run["id"]]; test=load(run["id"]); table.append({**complete,"test_metrics":test["metrics"]})
    summary={"sprint":"T2-C1","evaluation_revision":2,"word_scoring":"Lexical words only; sil and sp excluded. Original ridge coefficients, all neural weights, thresholds, spectra and waveforms unchanged.","content_gate_passes":all(content_gates),"rejection_gate_passes":all(reject_gates),"video_gate_passes":all(video_gates),"practical_speech_gate_passes":asr["practical_gate_passes"],"seeds":results,"runs":table,"training_wall_minutes":sum(r["wall_seconds"] for r in table)/60,"speech_diagnostic":asr["mean_clip_wer"],"source_sha256":file_sha(Path(__file__)),"scope":"New GRID people and sentence compositions, restricted six-word grammar. Offline repair, not general world modeling or calibrated semantic truth."}
    write_json(root/"summary_v2.json",summary)
    lines=["# T2-C1 — speech content and mistimed-video rejection (word evaluation revision2)","","The original word metric included19 short-pause frames. Revision2 excludes both sil and sp, reusing the same training-fitted ridge coefficients. All spectra, weights, thresholds and waveforms remain unchanged; original metrics and report are retained.","", "All choices use training, selection or independent calibration people. Final people and exact sentence combinations were reserved before fitting. Primary gates remain fixed, including a coverage requirement so rejecting every video cannot pass.","",
        "| Gate | Both-seed outcome |","|---|---|",f"| Field content + spectral/clean guardrails | {'PASS' if summary['content_gate_passes'] else 'FAIL'} |",f"| Mistimed-video rejection + aligned coverage | {'PASS' if summary['rejection_gate_passes'] else 'FAIL'} |",f"| Useful calibrated video beyond strong audio | {'PASS' if summary['video_gate_passes'] else 'FAIL'} |",f"| Independent speech diagnostic | {'PASS' if summary['practical_speech_gate_passes'] else 'FAIL'} |","",
        "## Content and video effects","","Effects are paired per-clip means, with2,000-draw person-cluster95% intervals. Word readouts are separate training-fitted ridge classifiers of processed audio fields, with no raw bypass. Text labels are never supplied during practical audio/video inference.","", "| Seed | Content gap-MSE gain | Field word-accuracy gain | Clean-MSE ratio | Calibrated video gain vs audio | Aligned coverage |","|---|---:|---:|---:|---:|---:|"]
    def fmt(x):
        lo,hi=x["person_cluster_95_interval"]; return f"{100*x['mean']:.2f}% [{100*lo:.2f}, {100*hi:.2f}]"
    for seed,s in results.items():
        w=s["field_word_readout_gain"]; lo,hi=w["person_cluster_95_interval"]
        lines.append(f"| {seed} | {fmt(s['content_gain'])} | {100*w['absolute_accuracy_gain']:.2f}pp [{100*lo:.2f}, {100*hi:.2f}] | {s['clean_mse_ratio']:.3f} | {fmt(s['video_gain'])} | {fmt(s['aligned_coverage'])} |")
    lines += ["","## Rejection with usefulness accounted for","","The video model stores the frozen strong audio candidate. Every rejection returns that prediction and its unit/latent state exactly. Acceptance requires zero predicted offset and an observed-only quality threshold fixed on three separate calibration people. Audio/video candidates have approximately matched graph-connected trained capacity; the deployed video model has additional frozen fallback storage and extra negative-example compute.","", "| Seed / video | Accepted | Repair harm vs strong audio | Raw unfiltered harm |","|---|---:|---:|---:|"]
    for seed,s in results.items():
        for mode,c in s["controls"].items(): lines.append(f"| {seed} / {mode} | {fmt(c['acceptance'])} | {fmt(c['relative_harm_vs_strong_audio'])} | {fmt(c['raw_relative_harm_vs_strong_audio'])} |")
    lines += ["","Positive harm is worse than audio alone. ±240ms is primary; ±160ms is an untrained shift magnitude. Wrong utterances, static/blank video and missing-video behavior are diagnostics. Wrong sources differing only within an erased word can be observationally ambiguous; timing rejection is not calibrated semantic truth.","","## Every trained artifact","", "| Run | Variant / seed | Selected / updates | Trainable / frozen | Total | Train minutes | Test320 MSE | Missing-word accuracy |","|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in table:
        run=r["run"]; c=r["counts"]; word=r["test_metrics"]["correct_word_accuracy"]; variant=run.get("arm","base")+" units="+str(run["unit_weight"])
        lines.append(f"| {run['id']} | {variant} / {run['seed']} | {r['selected_step']} / {r['updates']} | {c['trainable']:,} / {c['frozen']:,} | {c['total']:,} | {r['wall_seconds']/60:.2f} | {r['test_metrics']['correct_hole_mse']:.5f} | {100*word:.2f}% |")
    lines += ["",f"Measured training wall time on the RTX3090: {summary['training_wall_minutes']:.2f} minutes, including validation. Historical R2 pretraining and corpus preparation are additional. All best optimizer/RNG states, final weights, counts, sources/configs, curves, per-clip scores and160/480ms secondary results are retained.","","## Actual speech and text diagnostics","","All32 listening IDs were fixed before final scores. Observed PCM is preserved on the video-aligned decoded timeline; initial source stream trimming is recorded. Practical model features and phase erase original hidden samples before transforms. The true-spectrum and timed-text conditions deliberately receive reference information and are clearly labeled diagnostics, not practical repair results. Whisper-tiny.en is pinned, external and never trains a generator.","", "| Condition | Mean clip WER |","|---|---:|"]
    for name,value in asr["mean_clip_wer"].items(): lines.append(f"| {name} | {100*value:.2f}% |")
    lines += ["","See `exploration/intelligibility.json` for paired intervals and raw WER. Digit/letter splitting and single-digit number normalization are identical for reference and hypotheses; no grammar/homophone correction is applied. The `word_probes.json` files instead contain actual SLWM closed-vocabulary outputs for the context, not recognizer text and not a learned output-policy commitment.","","## Scope, data and failures","",
        "Training uses GRID people01–16 on sentence-hash buckets0–7. Selection17–20 and calibration22–24 use bucket8. Final25–32 use bucket9: both people and complete sentence combinations are new. Speaker21 has no video;33/34 remain unopened. The vocabulary/grammar stays restricted, so this is not novel-vocabulary or natural-world validation. CREMA/Libri training windows retain broader acoustics; their final checks are reused retention diagnostics. No codec is fitted to GRID holdouts.","",
        "The0.5% source alarm stopped preparation. Raw CRC/frame-count review verified90 genuinely short video files, all from training person08 (88train/2training-person diagnostic). The exact reviewed list passes the alarm; the50-packet input rule and all holdout people stay unchanged. Failed preparation logs and source review are retained.","",
        "The content objective is compared to the same architecture without its explicit unit CE; both have the timed-text auxiliary reconstruction interface. This is not an ablation removing all text supervision. Video pooling/translation, lexical objectives and the admission detector are an engineering package, with no claim that any one module alone causes all gains. Ungated results on the same weights isolate the admission decision.","",
        "T1-S2 separately measures causal English prediction at GPT-2 parameter scale on a much larger corpus. These are distinct trained weights. Neither study establishes a unified124M multimodal world model, future physical dynamics, general reasoning or a learned policy. A failed gate remains a negative/mixed result; further changes require a new preregistered holdout."
    ]
    (root/"report_v2.md").write_text("\n".join(lines)+"\n")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout="constrained")
    labels=["71","72"]; gains=[results[s]["video_gain"]["mean"]*100 for s in labels]; lo=[(results[s]["video_gain"]["mean"]-results[s]["video_gain"]["person_cluster_95_interval"][0])*100 for s in labels]; hi=[(results[s]["video_gain"]["person_cluster_95_interval"][1]-results[s]["video_gain"]["mean"])*100 for s in labels]
    axes[0].errorbar(gains,np.arange(2),xerr=[lo,hi],fmt="o",capsize=4); axes[0].set_yticks(range(2),["Seed "+s for s in labels]); axes[0].axvline(0,color="black",lw=.8); axes[0].axvline(3,color="gray",ls="--"); axes[0].set_xlabel("Calibrated video reduction in gap MSE (%)"); axes[0].set_title("Fresh people + new sentence combinations")
    x=np.arange(2); axes[1].bar(x-.18,[results[s]["aligned_coverage"]["mean"]*100 for s in labels],.36,label="Aligned accepted"); axes[1].bar(x+.18,[max(results[s]["controls"][m]["acceptance"]["mean"] for m in ("shift6","shift-6"))*100 for s in labels],.36,label="Mistimed accepted"); axes[1].axhline(50,color="gray",ls="--",lw=.8); axes[1].set_xticks(x,["Seed "+s for s in labels]); axes[1].set_ylabel("Acceptance (%)"); axes[1].set_ylim(0,105); axes[1].legend(); axes[1].set_title("Rejection must preserve useful coverage")
    fig.suptitle("Content-aware repair and observed-only video admission"); fig.savefig(root/"content_and_rejection_v2.png",dpi=180); plt.close(fig)
    print(json.dumps({k:summary[k] for k in ("content_gate_passes","rejection_gate_passes","video_gate_passes","practical_speech_gate_passes","training_wall_minutes")}))


if __name__=="__main__": main()
