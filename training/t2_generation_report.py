"""Auditable tables, gates and scientific figures from frozen T2-G1 results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from training.t2_real_av import write_json,cluster_interval,stable_hash
from training.t2_prepare_cremad import sha256


LABELS={"EXP-T2-401":"frozen audio field", "EXP-T2-402":"frozen video field", "EXP-T2-403":"raw 25 Hz audio", "EXP-T2-404":"raw video", "EXP-T2-405":"observed 100 Hz audio (richer input)",
        "EXP-T2-411":"audio field + seed", "EXP-T2-412":"raw audio + seed", "EXP-T2-413":"time + seed", "EXP-T2-414":"audio field + seed, no flow", "EXP-T2-415":"wrong-audio training + seed", "EXP-T2-416":"audio field + seed, seed 42"}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text()); root=Path(cfg["artifacts"])
    read=lambda p:json.loads(p.read_text())
    complete={r["id"]:read(root/r["id"]/"training_complete.json") for r in cfg["runs"]}
    metrics={s:{r["id"]:read(root/r["id"]/(s+"_metrics.json")) for r in cfg["runs"]} for s in ("validation","test","external")}
    for split in metrics:
        corrected=root/"EXP-T2-415"/("corrected_"+split+"_metrics.json")
        if corrected.exists(): metrics[split]["EXP-T2-415"]=read(corrected)
    asr=read(root/"exploration/intelligibility.json"); controls={s:read(root/(s+"_audio_controls.json")) for s in metrics}
    alignment=read(root/"external_frozen_alignment.json")
    visual={}; audio={}; comparisons={}
    for split in ("test","external"):
        rows=metrics[split]; vm={k:v["metrics"] for k,v in rows.items()}; visual[split]={}; audio[split]={}
        for run in cfg["runs"]:
            key=run["id"]; measured=rows[key]; m=vm[key]
            if run["task"]=="visual":
                persist_gain=1-m["correct_roi_mse"]/m["persistence_roi_mse"]
                time_gain=1-m["correct_roi_mse"]/vm["EXP-T2-413"]["correct_roi_mse"]
                wrong_gain=measured["correct_vs_same_person_wrong_relative_gain"]
                ci=measured["gain_person_cluster_95_interval"]
                passed=persist_gain>=.05 and time_gain>=.05 and wrong_gain>=.02 and ci[0]>0
                visual[split][key]={"roi_mse":m["correct_roi_mse"],"whole_frame_mse":m["correct_pixel_mse"],"temporal_mse":m["correct_temporal_mse"],
                    "motion_energy_ratio":m["motion_energy_ratio"],"improvement_vs_persistence":persist_gain,"improvement_vs_time_only":time_gain,
                    "correct_vs_wrong_gain":wrong_gain,"wrong_gain_person_cluster_95_interval":ci,"gate_pass":passed}
            else:
                reference="old_video_interpolated" if run["input"] in {"field_video","video_coarse"} else "old_audio_interpolated"
                old=controls[split]["metrics"]; wer=asr["summary"][split]["mean_clip_wer"]; oldwer="old_video" if reference.startswith("old_video") else "old_audio"
                wer_gain=1-wer[key]/max(wer[oldwer],1e-9)
                passed=wer_gain>=.2 and m["correct_mel_mse"]<old[reference+"_mel_mse"] and m["correct_temporal_mse"]<old[reference+"_temporal_mse"]
                if oldwer=="old_video": passed=passed and measured["correct_vs_same_person_wrong_relative_gain"]>0 and measured["gain_person_cluster_95_interval"][0]>0 and m["correct_mel_mse"]<m["null_mel_mse"]
                audio[split][key]={"mel100_mse":m["correct_mel_mse"],"temporal_mse":m["correct_temporal_mse"],"mean_clip_wer":wer[key],
                    "wer_relative_improvement_vs_old":wer_gain,"gate_pass":bool(passed),"richer_input_diagnostic":run.get("richer_input_diagnostic",False),
                    "correct_vs_wrong_gain":measured["correct_vs_same_person_wrong_relative_gain"],"wrong_gain_person_cluster_95_interval":measured["gain_person_cluster_95_interval"]}
        # Paired per-clip differences do not equate a frozen core with a raw
        # decoder's total parameter/compute cost; both are reported separately.
        comparisons[split]={}
        for learned,baseline,metric in (("EXP-T2-401","EXP-T2-403","mel_mse"),("EXP-T2-402","EXP-T2-404","mel_mse"),("EXP-T2-411","EXP-T2-412","roi_mse"),("EXP-T2-411","EXP-T2-413","roi_mse"),("EXP-T2-416","EXP-T2-413","roi_mse")):
            a,b=rows[learned]["per_clip"],rows[baseline]["per_clip"]
            if [r["sample_id"] for r in a]!=[r["sample_id"] for r in b]: raise ValueError("Unpaired reports")
            delta=np.array([(bb["correct_"+metric]-aa["correct_"+metric])/max(bb["correct_"+metric],1e-9) for aa,bb in zip(a,b)])
            comparisons[split][learned+"_vs_"+baseline]={"mean_per_clip_relative_gain":float(delta.mean()),"person_cluster_95_interval":cluster_interval(delta,[r["actor"] for r in a])}
    records=[]
    for run in cfg["runs"]:
        key=run["id"]; r=complete[key]
        if sha256(root/key/"best.pt")!=r["checkpoint_sha256"]: raise ValueError("Checkpoint changed")
        registry={"sprint":"T2-G1","experiment_id":key,"run":run,"config_hash":stable_hash(cfg),"selected_step":r["selected_step"],
            "checkpoint_sha256":r["checkpoint_sha256"],"accounting_mode":"core frozen; decoder trained, separate counts",**r["provenance"],
            "wall_seconds":r["wall_seconds"],"windows_seen":r["windows_seen"],"metrics_files":[("corrected_" if key=="EXP-T2-415" and (root/key/("corrected_"+s+"_metrics.json")).exists() else "")+s+"_metrics.json" for s in metrics],
            "claim_scope":"diagnostic simultaneous conditional generation, no policy commit"}
        write_json(root/key/"registry.json",registry); records.append(registry)
    summary={"sprint":"T2-G1","config_hash":stable_hash(cfg),"runs":records,"visual":visual,"audio":audio,"paired_core_comparisons":comparisons,
        "independent_recognizer":asr["summary"],"external_frozen_alignment":alignment["synchronization"],
        "total_decoder_training_wall_seconds":sum(r["wall_seconds"] for r in complete.values()),"core_sha256":cfg["checkpoint_sha256"],
        "limitations":["CREMA test reused in earlier research cycles", "external set has four people and two fixed prompts", "one real visual seed is additional conditioning",
            "100Hz input diagnostic has richer observations", "Whisper tiny.en can hallucinate; text is evaluator output, never model transcription",
            "grid_sample CUDA backward is not bitwise deterministic", "these are short simultaneous translations, not autonomous future rollouts", "no policy or uncertainty gate trained"]}
    write_json(root/"summary.json",summary)
    lines=["# T2-G1 — conditional generation and frozen-world decoding", "", "## Research question and scope", "",
        "Can finer-rate sound targets and a single observed appearance frame repair output heads without changing the aligned core? Eleven preregistered decoder runs retain the 4,741,089-parameter frozen M1 core. The core is counted in addition to each decoder. Raw-feature controls use the same decoder and observations but do not pay the core cost; the 100 Hz audio-input diagnostic has richer observations and is not a matched core comparison.", "",
        "This is diagnostic simultaneous conditional generation. All sound in a window is observed before generating its mouth motion. No policy commits, autonomous future-prediction or general world-model claims are tested.", "", "## Data, compute and provenance", "",
        "Comparison limit: head parameters, sampled targets and update budgets are matched within each task. Cached frozen fields overlap 32-frame encodes across the full observed recording and can include context outside a decoder's 32-frame crop; raw corrections see that crop directly. Core arms therefore have extra processing/context as well as extra weights. These are not strictly matched total-compute or temporal-receptive-field comparisons. The failed core-benefit result does not establish that every possible shared-core objective is ineffective.", "",
        "CREMA-D: 5,061 train / 210 validation / 240 test clips, 62 / 7 / 8 disjoint people, 3.636 training hours. This test split was evaluated in M1, so its results are exploratory. External RAVDESS: 64 clips from actors 01,02,13,14, two new utterances, no fitting, normalization, tuning or checkpoint selection. File SHA256 and ZIP member CRC verified. The fixed external crop was chosen from framing before scores. [RAVDESS source](https://zenodo.org/records/1188976), Livingstone & Russo (2018), CC BY-NC-SA 4.0; evaluation-only noncommercial research.", "",
        "Each arm trains 1,600 updates on the same recorded training data. Audio: batch 32, 51,200 sampled 1.28s windows; visual: batch 16, 25,600 windows. Budgets are matched within each task, not across tasks. Validation selects every 200 steps. Continuous pixel/mouth/edge/temporal losses exclude the supplied seed. All visual heads initialize at persistence. No pretrained neural generator or source waveform supplies generated output. Audio phase is newly estimated by 48-step Griffin-Lim, independently of the recorded phase.", "",
        f"Decoder-training total: {summary['total_decoder_training_wall_seconds']/60:.2f} minutes on one RTX 3090 (data preparation, field extraction, scoring and rendering excluded). Codec fit: 249,248 old PCA/statistical values + 128 new 100 Hz normalization values, training people only. Evaluator-only recognizer: 37,760,256 weights, pinned revision `87c7102498dcde7456f24cfd30239ca606ed9063`.", "",
        "| ID | Conditioning | New trainable weights | Frozen core | Selected step | Train seconds |", "|---|---|---:|---:|---:|---:|"]
    for run in cfg["runs"]:
        r=complete[run["id"]]; p=r["provenance"]["parameter_accounting"]
        lines.append(f"| {run['id']} | {LABELS[run['id']]} | {p['decoder_trainable']:,} | {p['frozen_core']:,} | {r['selected_step']} | {r['wall_seconds']:.1f} |")
    for split in ("test","external"):
        lines += ["",f"## {'Reused CREMA test' if split=='test' else 'New external RAVDESS'}: visual results", "",
            "Lower MSE is better; positive gains favor correct input. All visual predictions receive exactly one real seed. Motion ratio 0 is static persistence; 1 matches recorded motion energy but does not establish correct timing. Metrics use the fixed 32-frame middle window and exclude the seed from pixel errors. CUDA flow backward has nondeterminism; both seeds and actual checkpoint hashes are retained.", "",
            "| ID | Mouth MSE | Gain vs seed | Gain vs time only | Correct vs wrong audio | Motion ratio | Gate |", "|---|---:|---:|---:|---:|---:|---|"]
        for key,v in visual[split].items():
            ci=v["wrong_gain_person_cluster_95_interval"]
            lines.append(f"| {key} | {v['roi_mse']:.6f} | {100*v['improvement_vs_persistence']:+.1f}% | {100*v['improvement_vs_time_only']:+.1f}% | {100*v['correct_vs_wrong_gain']:+.2f}% [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] | {v['motion_energy_ratio']:.3f} | {'PASS' if v['gate_pass'] else 'FAIL'} |")
        lines += ["", "The frozen visual gate requires ≥5% lower mouth MSE than both persistence and time-only, ≥2% correct-versus-wrong audio benefit and a positive person-cluster 95% interval. External direction must persist without tuning. Full-frame and temporal errors and all per-clip values are in each metrics JSON. Four external people limit confidence.", "",
            f"## {'Reused CREMA test' if split=='test' else 'New external RAVDESS'}: audio results", "",
            "All mel errors are measured on the same 100 Hz physical grid with training-only normalization. Old 25 Hz heads are linearly interpolated onto that grid, holding the final three subframes. WER uses the preregistered first four sample IDs per person: 32 CREMA / 16 external clips. Alphabetical CREMA selection concentrates on the DFA prompt; it is a narrow repeated-prompt diagnostic, not a vocabulary benchmark. Recognizer text is measurement, never text produced by SLWM. WER may exceed 100% through insertions.", "",
            "| Condition | 100 Hz mel MSE | Temporal MSE | Mean clip WER | Audio gate |", "|---|---:|---:|---:|---|"]
        old=controls[split]["metrics"]; wer=asr["summary"][split]["mean_clip_wer"]
        for key,c in (("codec25_interpolated","codec25"),("old_audio_interpolated","old_audio"),("old_video_interpolated","old_video")):
            lines.append(f"| {key} | {old[key+'_mel_mse']:.4f} | {old[key+'_temporal_mse']:.4f} | {100*wer[c]:.1f}% | control |")
        for key,a in audio[split].items(): lines.append(f"| {key}: {LABELS[key]} | {a['mel100_mse']:.4f} | {a['temporal_mse']:.4f} | {100*a['mean_clip_wer']:.1f}% | {'PASS' if a['gate_pass'] else 'FAIL'}{' (richer input)' if a['richer_input_diagnostic'] else ''} |")
        lines += ["",f"Original/25 Hz codec/100 Hz codec WER: {100*wer['reference']:.1f}% / {100*wer['codec25']:.1f}% / {100*wer['codec100']:.1f}%. Video-head wrong-video/null-video WER: {100*wer['wrong_video']:.1f}% / {100*wer['null_video']:.1f}%.", "",
            "The frozen audio gate requires ≥20% relative recognizer-WER improvement versus the old rendered head, plus lower mel and temporal errors. Cross-modal audio must additionally improve over mean/null and same-person wrong-input controls with positive person-cluster input-dependence interval. A codec or same-modality reconstruction win is not a video-to-speech win. Null errors and paired WER changes/intervals are retained in the JSON reports.", ""]
    sync=alignment["synchronization"]
    lines += ["## Additional frozen alignment audit", "",
        f"On RAVDESS, the original small frozen core selects the correct shift on {100*sync['accuracy']:.1f}% of {sync['eligible_clips']} clips, versus {100*sync['motion_energy_baseline_accuracy']:.1f}% for the energy/motion control and 14.3% chance. This is an exploratory extension to M1; it never selected or tuned G1 checkpoints.", "",
        "Evaluation correction: EXP-T2-415's original inference inherited its shuffled-training flag. Tables now use `corrected_*_metrics.json`, evaluating correct/wrong/null audio normally. All original metrics and media remain unchanged; separately labeled corrected media and `evaluation_correction.json` record the fix. The originally selected checkpoint is retained without reselection (its validation inputs were shuffled). The full-versus-time-only gate is unaffected.", "",
        "## Paired core versus simpler decoder comparisons", "", "Intervals resample people (2,000 bootstrap draws). These are mean per-clip relative improvements; ratios of aggregate MSE above can differ.", "", "| Dataset | Comparison | Relative gain | Person-cluster 95% interval |", "|---|---|---:|---:|"]
    for split,items in comparisons.items():
        for key,c in items.items():
            ci=c["person_cluster_95_interval"]; lines.append(f"| {split} | {key} | {100*c['mean_per_clip_relative_gain']:+.2f}% | [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]% |")
    lines += ["", "## Artifacts and reproducibility", "",
        "`summary.json` and each experiment's `registry.json`, configuration, curves, validation/test/external metrics, optimizer-bearing checkpoint and SHA256 retain every arm, including failures. `source_snapshot/` and `source_hashes.json` freeze training implementation. Corpus `*_manifest.json` files bind all array and normalization hashes. `exploration/gallery.json` lists validation low/median/high visual cases, fixed test/external cases and all 48 listening cases with original/codec/model/wrong/null controls. The real seed is saved separately. GIF/WebM and WAV files contain actual decoded predictions. Source artifacts, raw media and decoded caches remain on Fedora; reusable heads and media are mirrored locally.", "",
        "Reproduce: `bash scripts/train/run_conditional_generation_fedora.sh`. Standalone inference: `python -m exploration.generation_infer --help`. No inference updates weights. There is no generated free text or policy/uncertainty head in this stage.", "",
        "Research basis: [SadTalker](https://arxiv.org/abs/2211.12194) motivates separating reference appearance and motion; [EnCodec](https://arxiv.org/abs/2210.13438) motivates treating compression quality separately. Neither pretrained generator is used. Full interpretation and next allowed research step are in `docs/research/findings.md`."]
    (root/"report.md").write_text("\n".join(lines)+"\n")
    # Standalone scientific figure; no interactive product or synthetic media.
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid"); fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout="constrained")
    colors=["#246c91","#a4c5d5"]
    visual_ids=["EXP-T2-411","EXP-T2-412","EXP-T2-413","EXP-T2-415","EXP-T2-416"]; x=np.arange(len(visual_ids))
    for j,split in enumerate(("test","external")):
        axes[0].bar(x+(j-.5)*.38,[visual[split][k]["roi_mse"] for k in visual_ids],width=.38,label=split,color=colors[j])
    axes[0].set_xticks(x,["field","raw","time only","shuffled","field s42"],rotation=25,ha="right"); axes[0].set_ylabel("Mouth-region MSE (lower better)"); axes[0].set_title("One real seed; controls remain competitive"); axes[0].legend()
    audio_ids=["EXP-T2-401","EXP-T2-402","EXP-T2-403","EXP-T2-404","EXP-T2-405"]
    for j,split in enumerate(("test","external")):
        axes[1].bar(np.arange(5)+(j-.5)*.38,[audio[split][k]["mel100_mse"] for k in audio_ids],width=.38,label=split,color=colors[j])
    axes[1].set_xticks(np.arange(5),["field A","field V","raw A","raw V","100Hz A*"],rotation=25,ha="right"); axes[1].set_ylabel("100 Hz mel MSE (lower better)"); axes[1].set_title("*Richer input is a codec diagnostic")
    names=["reference","codec100","old_audio","EXP-T2-401","EXP-T2-403","EXP-T2-405","old_video","EXP-T2-402"]
    for j,split in enumerate(("test","external")):
        axes[2].plot(np.arange(len(names)),[100*asr["summary"][split]["mean_clip_wer"][k] for k in names],"o-",label=split,color=colors[j])
    axes[2].set_xticks(np.arange(len(names)),["real","codec","old A","field A","raw A","100Hz A*","old V","field V"],rotation=40,ha="right")
    axes[2].set_ylabel("Whisper-tiny mean clip WER (%)"); axes[2].set_title("48 fixed clips; fallible recognizer")
    fig.suptitle("T2-G1: measured conditional generation, frozen 4.74M core",fontsize=14)
    fig.savefig(root/"results.png",dpi=170); plt.close(fig)
    print(json.dumps({"report":str(root/"report.md"),"decoder_training_minutes":summary["total_decoder_training_wall_seconds"]/60,
        "visual_gates":{s:{k:v["gate_pass"] for k,v in values.items()} for s,values in visual.items()},"external_sync":sync["accuracy"]}),flush=True)


if __name__=="__main__": main()
