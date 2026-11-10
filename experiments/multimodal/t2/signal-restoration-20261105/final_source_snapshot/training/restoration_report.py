"""Frozen R1/R2 tables, gates, readout evidence and scientific comparison plot."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from training.restoration_other import expanded_config
from training.t2_real_av import write_json,cluster_interval,stable_hash
from training.t2_prepare_cremad import sha256


def read(path): return json.loads(Path(path).read_text())


def paired(a,b,key):
    if [r["sample_id"] for r in a]!=[r["sample_id"] for r in b]: raise ValueError("Unpaired reports")
    gain=np.array([(y[key]-x[key])/max(y[key],1e-9) for x,y in zip(a,b)])
    return {"mean_per_clip_gain":float(gain.mean()),"person_cluster_95_interval":cluster_interval(gain,[r["actor"] for r in a])}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--r1",required=True); parser.add_argument("--r2",required=True); args=parser.parse_args()
    c1=read(args.r1); c2=expanded_config(args.r2); p1,p2=Path(c1["artifacts"]),Path(c2["artifacts"])
    m1={s:{r["id"]:read(p1/r["id"]/(s+"_metrics.json")) for r in c1["runs"]} for s in ("crema_test","libri_test","ravdess_test")}
    m2={s:{r["id"]:read(p2/r["id"]/(s+"_metrics.json")) for r in c2["runs"]} for s in ("crema_test","libri_test","ravdess_test","other_test")}
    probes1=read(p1/"content_probe_summary.json"); probes2=read(p2/"content_probe_summary.json"); asr1=read(p1/"exploration/intelligibility.json"); asr2=read(p2/"exploration/intelligibility.json")
    silence=read(p1/"exploration/matched_silence_control.json"); complete1={r["id"]:read(p1/r["id"]/"training_complete.json") for r in c1["runs"]}; complete2={r["id"]:read(p2/r["id"]/"training_complete.json") for r in c2["runs"]}
    comparisons1={}
    for s,models in m1.items():
        comparisons1[s]={}
        for key in ("EXP-T2-501","EXP-T2-502"):
            comparisons1[s][key]={base:paired(models[key]["per_clip"],models[base]["per_clip"],"paired_hole_mse") for base in ("EXP-T2-503","EXP-T2-507","EXP-T2-508","EXP-T2-509")}
    clean_gate=all(m1[s][k]["metrics"]["clean_audio_mel_mse"]<=1.05*m1[s]["EXP-T2-507"]["metrics"]["clean_audio_mel_mse"] for s in ("crema_test","libri_test") for k in ("EXP-T2-501","EXP-T2-502"))
    repair_gate=all(1-m1["crema_test"]["EXP-T2-501"]["metrics"]["paired_hole_mse"]/m1["crema_test"][b]["metrics"]["paired_hole_mse"]>=.05 and comparisons1["crema_test"]["EXP-T2-501"][b]["person_cluster_95_interval"][0]>0 for b in ("EXP-T2-503","EXP-T2-508"))
    repair_gate=repair_gate and all(m1["crema_test"][k]["metrics"]["paired_hole_mse"]<m1["crema_test"]["EXP-T2-508"]["metrics"]["paired_hole_mse"] for k in ("EXP-T2-501","EXP-T2-502"))
    multimodal_gate=True
    for key in ("EXP-T2-501","EXP-T2-502"):
        x=m1["crema_test"][key]
        multimodal_gate=multimodal_gate and all(1-x["metrics"]["paired_hole_mse"]/x["metrics"][b+"_hole_mse"]>=threshold and x["paired_gains"][b]["person_cluster_95_interval"][0]>0 for b,threshold in (("masked_audio",.05),("wrong_video",.02)))
        ext=m1["ravdess_test"][key]["metrics"]
        multimodal_gate=multimodal_gate and ext["paired_hole_mse"]<min(ext["masked_audio_hole_mse"],ext["wrong_video_hole_mse"])
    alignment_gate=all(m1[s][key]["synchronization"]["accuracy"]>=threshold for s,threshold in (("crema_test",.9),("ravdess_test",.4)) for key in ("EXP-T2-501","EXP-T2-502"))
    gates1={"clean_output_retention":bool(clean_gate),"repair_over_raw_av_and_vanilla":bool(repair_gate),"positive_multimodal_contribution":bool(multimodal_gate),"timing_retention":bool(alignment_gate)}
    gates2={}; probe_gate=True; preserve_gate=True; repair2=True; timing2=True
    for key,base in (("EXP-T2-521","EXP-T2-501"),("EXP-T2-522","EXP-T2-502")):
        for s in ("libri_test","other_test"):
            ours=probes2["runs"][key]["metrics"][s]["mse"]; original=probes2["references"][base]["metrics"][s]["mse"]; old=probes2["references"]["EXP-T2-509"]["metrics"][s]["mse"]
            probe_gate=probe_gate and ours<=.8*original and ours<=1.05*old and probes2["paired_comparisons"][key][s][base]["person_cluster_95_interval"][0]>0
            ref=read(p2/"reference_metrics"/(base+"_other_test.json")) if s=="other_test" else m1[s][base]
            raw=read(p2/"reference_metrics/EXP-T2-507_other_test.json") if s=="other_test" else m1[s]["EXP-T2-507"]
            clean=m2[s][key]["metrics"]["clean_audio_mel_mse"]; preserve_gate=preserve_gate and clean<=1.05*ref["metrics"]["clean_audio_mel_mse"] and clean<=1.05*raw["metrics"]["clean_audio_mel_mse"]
        for s in ("crema_test","libri_test"):
            repair2=repair2 and m2[s][key]["metrics"]["paired_hole_mse"]<=1.05*m1[s][base]["metrics"]["paired_hole_mse"]
            preserve_gate=preserve_gate and m2[s][key]["metrics"]["clean_audio_mel_mse"]<=1.05*min(m1[s][base]["metrics"]["clean_audio_mel_mse"],m1[s]["EXP-T2-507"]["metrics"]["clean_audio_mel_mse"])
        timing2=timing2 and m2["crema_test"][key]["synchronization"]["accuracy"]>=.9 and m2["ravdess_test"][key]["synchronization"]["accuracy"]>=.4
    gates2={"field_content_conservation":bool(probe_gate),"clean_output_guardrail":bool(preserve_gate),"repair_guardrail":bool(repair2),"timing_retention":bool(timing2)}
    parity={s:abs(m2[s]["EXP-T2-523"]["metrics"]["paired_hole_mse"]-m1[s]["EXP-T2-501"]["metrics"]["paired_hole_mse"]) for s in ("crema_test","libri_test","ravdess_test")}
    summary1={"sprint":"T2-R1","gates":gates1,"paired_core_comparisons":comparisons1,"runs":complete1,"total_training_seconds":sum(r["wall_seconds"] for r in complete1.values()),
        "interpretation":"Preserve raw observations and timing; reliable positive contribution from video is not established. Linear field readout reveals a bypass incentive problem.","recognizer":asr1["summary"],"matched_silence_control":silence["summary"]}
    summary2={"sprint":"T2-R2","gates":gates2,"runs":complete2,"total_training_seconds":sum(r["wall_seconds"] for r in complete2.values()),"zero_auxiliary_control_absolute_mse_differences":parity,
        "probe_comparisons":probes2["paired_comparisons"],"recognizer_mean_clip_wer":asr2["mean_clip_wer"],"all_primary_gates_pass":bool(all(gates2.values()))}
    write_json(p1/"summary.json",summary1); write_json(p2/"summary.json",summary2)
    for cfg,root,complete in ((c1,p1,complete1),(c2,p2,complete2)):
        for run in cfg["runs"]:
            key=run["id"]; c=complete[key]
            if sha256(root/key/"best.pt")!=c["checkpoint_sha256"]: raise ValueError("Checkpoint changed")
            write_json(root/key/"registry.json",{"experiment_id":key,"sprint":cfg["sprint"],"config_hash":stable_hash(cfg),**c,"status":"completed","claim_scope":"simultaneous signal-packet restoration and representation diagnostics; no policy commit"})
    common=["", "## Data, observations and claim limits", "",
        "CREMA-D keeps 5,061 train / 210 validation / 240 test clips, with disjoint people; its test is reused. LibriSpeech dev-clean adds 1,278 training clips from 32 people and 64 validation clips from eight other people. Long training clips use the first 5.12s; no transcript is a training target. Every update draws 24 CREMA + 8 Libri windows, 1,600×32 updates, exactly 32 packets per learned context. Half of CREMA video inputs are dropped. Only head/data/new update budgets match: total core parameters, pretraining and compute differ. Both signal and vanilla cores warmstart from their earlier corresponding 1,600-update M1 runs; raw controls have no such pretraining. Old unused reconstruction heads are removed. No wider cached world features enter this cycle.", "",
        "R1 first reserves 48 complete test-clean utterances from 12 new Libri people and 128 RAVDESS clips from eight new people. These guide the separately labeled R2 repair and become reused diagnostics. R2 reserves 48 complete test-other utterances from 12 different Libri people as its fresh, harder recording-condition check. Neither external set fits statistics, weights, readouts or checkpoint selection. All codec/PCA/statistical values remain from CREMA training only (249,376 fitted values). Libri archives are checked against official MD5; RAVDESS uses selected ZIP-member CRC and individual media SHA256, with archive MD5 declarations retained rather than recomputed for the unfetched full ZIPs. Sources: [LibriSpeech](https://www.openslr.org/12), Panayotov et al. (2015), CC BY 4.0; [RAVDESS](https://zenodo.org/records/1188976), Livingstone & Russo (2018), CC BY-NC-SA 4.0, noncommercial evaluation only.", "",
        "Two four-packet holes with observed boundary anchors remove 25% of each middle window. Deltas are computed after observed-only interpolation. Hidden target changes cannot affect any input or prediction. Video is simultaneous context, not a future observation. Reconstruction uses hidden-packet weight 4 and temporal L1; independent alignment never sees its paired modality. Gain ±12dB and smooth four-knot EQ ±6dB affect BOTH observations and target, so the model preserves the observed channel/level rather than guessing an unknowable original gain. These are mel-domain approximations, not real room simulation.", "",
        "Known CREMA prompt IDs only exclude contrastive false negatives, and person groups define controlled negative examples. No transcript words or speaker ID is an encoder input; this is not text-generation training or a fully unlabeled data claim.", "",
        "Important scope: 64ms centered codec windows can straddle packet boundaries. This withholds mel packets, not sample-exact waveform spans. Actual audio uses inverse mel and newly estimated Griffin-Lim phase; no original waveform samples supply model output. Listening uses one global mask before overlapping windows, preventing a hidden packet from returning in a neighboring window. Global interpolation can use observed anchors outside a render window equally for all models; primary window metrics use only their 32-packet context. Models have not learned policy/uncertainty commitment, free text or autonomous world dynamics.", "",
        "Checkpoint selection uses ONLY the equal-corpus mean validation hole MSE every 400 updates. All seeds, checkpoints, last weights, optimizer-bearing best states, curves, hashes and failures remain. Paired confidence intervals resample people (2,000 draws), not frames. Ridge readouts fit 512 CREMA + 512 Libri training clips only, fixed regularization, 16,448 coefficient/bias values plus 256 input means, and see ONLY frozen world fields. Lower readout MSE is linear decodability, not irreversible information capacity or semantic understanding. Raw input already contains the observed coarse target, so copying is a trivial perfect baseline for that assay.", "",
        "Research basis: [masked audiovisual representation learning](https://arxiv.org/abs/2201.02184) and [speech inpainting](https://www.isca-archive.org/interspeech_2023/montesinos23_interspeech.html). No pretrained generator from those works is used. The only external neural evaluator is pinned Whisper-tiny.en (37,760,256 weights, revision `87c7102498dcde7456f24cfd30239ca606ed9063`), never a training/generation component. Recognizer text is not model text; WER is fallible and can exceed 100%."]
    lines=["# T2-R1 — shared-field speech-packet restoration", "", "**Reliable benefit from adding video is not established.** The model repairs spectral features and preserves timing, but wrong-video sensitivity is not sufficient: correct video must also beat sound alone. The frozen field readout motivates the separate R2 content objective.", "", "## Frozen gates", "", "| Gate | Result |", "|---|---|"]
    lines += [f"| {key} | {'PASS' if value else 'FAIL'} |" for key,value in gates1.items()]
    lines += ["", "## Models and compute", "", "| ID | Arm | Total neural weights | Trainable | Frozen | Selected step | Train seconds |", "|---|---|---:|---:|---:|---:|---:|"]
    for run in c1["runs"]:
        c=complete1[run["id"]]; n=c["provenance"]["counts"]; lines.append(f"| {run['id']} | {run['arm']} | {n['total']:,} | {n['trainable']:,} | {n['frozen']:,} | {c['selected_step']} | {c['wall_seconds']:.1f} |")
    lines += ["",f"New decoder/core training: {summary1['total_training_seconds']/60:.2f} RTX 3090 minutes; data preparation, readout fitting, scoring and rendering excluded. Update budgets are matched, not wall time or total historical pretraining compute."]
    for split in m1:
        lines += ["",f"## {split}", "", "| ID | Paired hole MSE | Sound-only hole MSE | Clean-sound MSE | Timing accuracy |", "|---|---:|---:|---:|---:|"]
        for run in c1["runs"]:
            r=m1[split][run["id"]]; m=r["metrics"]; accuracy=r.get("synchronization",{}).get("accuracy"); sync="—" if accuracy is None else f"{100*accuracy:.1f}%"
            lines.append(f"| {run['id']} | {m['paired_hole_mse']:.5f} | {m['masked_audio_hole_mse']:.5f} | {m['clean_audio_mel_mse']:.5f} | {sync} |")
        lines += ["", "Correct-video versus sound-only/wrong-video mean per-clip gains (cluster intervals):", ""]
        for key in ("EXP-T2-501","EXP-T2-502"):
            g=m1[split][key]["paired_gains"]; items=[]
            for b in ("masked_audio","wrong_video"):
                ci=g[b]["person_cluster_95_interval"]; items.append(f"{b}: {100*g[b]['mean_per_clip_gain']:+.2f}% [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]")
            lines.append(f"- {key}: "+"; ".join(items))
        if split in {"crema_test","ravdess_test"}:
            lines += ["", "Independent timing controls (correct / wrong sound / zero sound / energy baseline):", ""]
            for key in ("EXP-T2-501","EXP-T2-502"):
                t=m1[split][key]["synchronization"]; ci=t["person_cluster_accuracy_95_interval"]
                lines.append(f"- {key}: {100*t['accuracy']:.1f}% / {100*t['wrong_audio_accuracy']:.1f}% / {100*t['zero_audio_accuracy']:.1f}% / {100*t['energy_baseline_accuracy']:.1f}%; correct interval [{100*ci[0]:.1f}, {100*ci[1]:.1f}]%, chance 14.3%.")
        wer=asr1["summary"][split]["mean_clip_wer"]
        lines += ["",f"Recognizer mean clip WER: reference {100*wer['reference']:.1f}%, target codec {100*wer['codec100']:.1f}%, interpolation {100*wer['interpolation']:.1f}%, full seeds {100*wer['EXP-T2-501']:.1f}%/{100*wer['EXP-T2-502']:.1f}%, raw audio {100*wer['EXP-T2-507']:.1f}%, observation-matched silence {100*silence['summary'][split]['mean_clip_wer']:.1f}%."]
    lines += ["", "## Mechanism and failure analysis", "",
        "The bypass protects decoded observations while allowing the field to specialize in corrections. The full skip model's linear coarse-sound readout is worse than the old frozen core; the no-skip arm's readout is better even though its generated signal is worse. This is evidence about decodability/incentives, not proof of irreversible information deletion. R2 tests an explicit observed-content gradient.", "",
        "A matched silence control was added after the initial listening pass. The original `silence_gaps` retains 100Hz target detail outside holes and therefore has richer observations than the models; it remains as a diagnostic, never a fair model baseline. `coarse_silence.wav` instead blanks the same 25Hz-derived interpolation available to every model. Separate original and matched-control transcripts/results remain intact."]+common
    (p1/"report.md").write_text("\n".join(lines)+"\n")
    decision="Retain the explicit content objective: both primary seeds pass its registered gates." if all(gates2.values()) else "The content objective does not pass every registered guardrail."
    lines=["# T2-R2 — conserve observed content in the shared field", "", "**"+decision+"** These are content/output/timing guardrails; the R1 positive-video-contribution gate remains failed.", "", f"Primary auxiliary=.5 two-seed gates: `{json.dumps(gates2,sort_keys=True)}`.", "",
        "The intervention adds only 16,448 linear-head weights and an observed-packet MSE objective. The primary decoder and raw preservation path are unchanged. All runs restart from the same original M1 source and use the exact R1 training data, sampled windows and hyperparameters. R1 already exposed test-clean/RAVDESS; only test-other is fresh here. Coefficient 2 is a predeclared sensitivity run, not a winner selected after test inspection.", "",
        "## Models and compute", "", "| ID | Auxiliary weight | Total neural weights | Selected step | Train seconds |", "|---|---:|---:|---:|---:|"]
    for run in c2["runs"]:
        c=complete2[run["id"]]; lines.append(f"| {run['id']} | {run['aux_weight']} | {c['provenance']['counts']['total']:,} | {c['selected_step']} | {c['wall_seconds']:.1f} |")
    lines += ["",f"Training total {summary2['total_training_seconds']/60:.2f} RTX 3090 minutes, preparation/scoring excluded. Zero-auxiliary control reproduces R1 paired-hole MSE with absolute differences: `{json.dumps(parity,sort_keys=True)}`. The unused auxiliary head is retained in the parameter count for that control.", "",
        "## Frozen-field decodability: audio-only core inputs, no raw bypass", "", "| Model | CREMA | test-clean (reused) | RAVDESS (reused) | test-other (fresh) |", "|---|---:|---:|---:|---:|"]
    for key in ("EXP-T2-509","EXP-T2-501","EXP-T2-502","EXP-T2-506"):
        d=probes2["references"][key]["metrics"]; lines.append(f"| R1 {key} | "+" | ".join(f"{d[s]['mse']:.5f}" for s in ("crema_test","libri_test","ravdess_test","other_test"))+" |")
    for run in c2["runs"]:
        d=probes2["runs"][run["id"]]["metrics"]; lines.append(f"| {run['id']} | "+" | ".join(f"{d[s]['mse']:.5f}" for s in ("crema_test","libri_test","ravdess_test","other_test"))+" |")
    lines += ["", "Paired mean per-clip readout improvements over the corresponding R1 full seed (ratios of aggregate MSE can differ):", "", "| Model | Set | Relative gain | Person-cluster 95% interval |", "|---|---|---:|---:|"]
    for key,base in (("EXP-T2-521","EXP-T2-501"),("EXP-T2-522","EXP-T2-502")):
        for s in ("libri_test","other_test"):
            v=probes2["paired_comparisons"][key][s][base]; ci=v["person_cluster_95_interval"]
            lines.append(f"| {key} | {s} | {100*v['mean_relative_gain']:.1f}% | [{100*ci[0]:.1f}, {100*ci[1]:.1f}]% |")
    lines += ["", "## Primary output and timing guardrails", "", "| Model | CREMA hole MSE | test-clean hole MSE | Fresh test-other hole MSE | Fresh clean MSE | CREMA timing | RAVDESS timing |", "|---|---:|---:|---:|---:|---:|---:|"]
    for run in c2["runs"]:
        key=run["id"]; a=m2["other_test"][key]["metrics"]
        lines.append(f"| {key} | {m2['crema_test'][key]['metrics']['paired_hole_mse']:.5f} | {m2['libri_test'][key]['metrics']['paired_hole_mse']:.5f} | {a['paired_hole_mse']:.5f} | {a['clean_audio_mel_mse']:.5f} | {100*m2['crema_test'][key]['synchronization']['accuracy']:.1f}% | {100*m2['ravdess_test'][key]['synchronization']['accuracy']:.1f}% |")
    lines += ["", "## Fresh test-other listening diagnostic", "", "| Condition | Mean clip WER |", "|---|---:|"]
    for key,value in asr2["mean_clip_wer"].items(): lines.append(f"| {key} | {100*value:.1f}% |")
    lines += ["", "No intelligibility claim follows a feature-MSE improvement alone. Compare the same-observation silence control and the untouched original/codec controls. All WAV files are actual predictions with estimated phase, and the fresh set has only 12 people. Reports retain all predeclared coefficients and both primary seeds, including failed gates."]+common
    (p2/"report.md").write_text("\n".join(lines)+"\n")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid"); fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout="constrained")
    ids=("EXP-T2-509","EXP-T2-501","EXP-T2-506","EXP-T2-521","EXP-T2-522","EXP-T2-524"); x=np.arange(len(ids)); colors=("#247898","#d98247")
    for j,s in enumerate(("libri_test","other_test")):
        values=[(probes2["references"] if k in probes2["references"] else probes2["runs"])[k]["metrics"][s]["mse"] for k in ids]
        axes[0].bar(x+(j-.5)*.36,values,.36,label="clean (reused)" if j==0 else "harder (fresh)",color=colors[j])
    axes[0].set_xticks(x,["old core","skip","no skip","aux .5 seed 51","aux .5 seed 52","aux 2 (one seed)"],rotation=35,ha="right"); axes[0].set_ylabel("Linear sound-readout MSE"); axes[0].set_title("ONLY the internal field; lower better"); axes[0].legend()
    cases=[("interpolation",m1["crema_test"]["EXP-T2-501"]["metrics"]["interpolation_hole_mse"]),("raw audio",m1["crema_test"]["EXP-T2-507"]["metrics"]["paired_hole_mse"]),("vanilla",m1["crema_test"]["EXP-T2-503"]["metrics"]["paired_hole_mse"]),("signal",m1["crema_test"]["EXP-T2-501"]["metrics"]["paired_hole_mse"]),("aux .5",m2["crema_test"]["EXP-T2-521"]["metrics"]["paired_hole_mse"])]
    axes[1].bar(np.arange(len(cases)),[v for _,v in cases],color="#247898"); axes[1].set_xticks(np.arange(len(cases)),[k for k,_ in cases],rotation=35,ha="right"); axes[1].set_ylabel("CREMA hidden-packet MSE"); axes[1].set_title("Repair must beat strong controls")
    names=("EXP-T2-509","EXP-T2-503","EXP-T2-501","EXP-T2-521"); x=np.arange(4)
    for j,s in enumerate(("crema_test","ravdess_test")):
        values=[100*(m1[s][k] if k in m1[s] else m2[s][k])["synchronization"]["accuracy"] for k in names]
        axes[2].bar(x+(j-.5)*.36,values,.36,label="CREMA" if j==0 else "RAVDESS",color=colors[j])
    axes[2].axhline(100/7,color="gray",ls="--",label="chance"); axes[2].set_xticks(x,["old core","vanilla","signal","aux .5"],rotation=35,ha="right"); axes[2].set_ylabel("Correct timing offset (%)"); axes[2].set_title("Alignment is separate from useful video"); axes[2].legend()
    fig.suptitle("T2-R1/R2: signal fidelity, missing-packet repair and correspondence",fontsize=14); fig.savefig(p2/"results.png",dpi=160); plt.close(fig)
    print(json.dumps({"r1_gates":gates1,"r2_gates":gates2,"training_minutes":(summary1['total_training_seconds']+summary2['total_training_seconds'])/60}),flush=True)


if __name__=="__main__": main()
