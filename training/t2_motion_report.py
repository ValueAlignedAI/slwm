"""Report architecture, scaling, controls and actual conditional outputs."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from training.t2_real_av import write_json,cluster_interval


def report(path: str) -> Path:
    cfg=json.loads(Path(path).read_text()); root=Path(cfg["artifacts"])
    runs=[json.loads((root/r["id"]/"metrics.json").read_text()) for r in cfg["runs"]]
    card=json.loads((Path(cfg["corpus"])/"dataset_card.json").read_text())
    causal=json.loads((root/"causal_checks.json").read_text()) if (root/"causal_checks.json").exists() else {}
    gallery=json.loads((root/"exploration/gallery.json").read_text())["examples"] if (root/"exploration/gallery.json").exists() else []
    large=[r for r in runs if r["run"]["arm"]=="signal_large"]
    decision={"synchronization_pass_both_large_seeds":all(r["gates"]["sync_pass"] for r in large),
              "video_to_audio_pass_both_large_seeds":all(r["gates"]["video_to_audio_pass"] for r in large),
              "audio_to_motion_pass_both_large_seeds":all(r["gates"]["audio_to_motion_pass"] for r in large),
              "causal_audio_checks_pass_both_large_seeds":all(causal.get(r["experiment_id"],{}).get("causal_audio_contribution",False) for r in large),
              "completed_runs":len(runs),"training_seconds":sum(r["training_seconds"] for r in runs)}
    decision["multimodal_translation_gate_pass"]=all(decision[k] for k in ("synchronization_pass_both_large_seeds","video_to_audio_pass_both_large_seeds","audio_to_motion_pass_both_large_seeds","causal_audio_checks_pass_both_large_seeds"))
    small=runs[0]
    for r in large:
        if r["run"]["seed"]==41:
            one={p["sample_id"]:p for p in small["test"]["synchronization"]["per_clip"]}
            larger=r["test"]["synchronization"]["per_clip"]
            difference=np.array([p["correct"]-one[p["sample_id"]]["correct"] for p in larger])
            decision["scale_seed41"]={"synchronization_absolute_gain":float(difference.mean()),
                "person_cluster_gain_interval":cluster_interval(difference,[p["actor"] for p in larger]),
                "video_to_audio_relative_error_gain":1-r["test"]["errors"]["video_input_audio_mel"]/small["test"]["errors"]["video_input_audio_mel"],
                "train_time_ratio":r["training_seconds"]/small["training_seconds"],
                "limits":"Small model has one seed; scaling evidence is a controlled pilot, not a two-seed scaling law"}
    write_json(root/"decision.json",decision)
    table=[]
    for r in runs:
        s=r["test"]["synchronization"]; e=r["test"]["errors"]
        table.append({"id":r["experiment_id"],"arm":r["run"]["arm"],"seed":r["run"]["seed"],"parameters":r["parameter_counts"]["total"],
            "sync_accuracy":s["accuracy"],"sync_energy_control":s["motion_energy_baseline_accuracy"],"v2a_mel_mse":e["video_input_audio_mel"],
            "a2motion_mse":e["audio_input_video_motion"],"a2v_r1":r["test"]["retrieval"]["audio_to_video_r1"],
            "seconds":r["training_seconds"],"best_step":r["best_step"]})
    with (root/"comparison.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    headline=("Physical-time alignment passes both larger-model seeds and the input-corruption checks. "
              if decision["synchronization_pass_both_large_seeds"] and decision["causal_audio_checks_pass_both_large_seeds"] else
              "The synchronization evidence is incomplete or fails a required control. ")
    headline+=("The two cross-modal decoding directions also pass the registered gates."
               if decision["multimodal_translation_gate_pass"] else
               "Cross-modal generation remains incomplete; do not treat playable output as a useful general world model.")
    lines=["# SLWM motion alignment, modest scaling and conditional outputs — 2026-11-04","","## Research decision","",headline,"",
        f"Completed **{len(runs)}** registered RTX 3090 runs, following the user's authorization to redesign the previous failed recipe and explore audio/video-conditioned outputs. "
        f"The small model has {small['parameter_counts']['total']:,} trainable parameters; the larger model {large[0]['parameter_counts']['total']:,}. "
        "No pretrained neural encoder or external media generator supplies the results.","","## What changed","",
        "A fixed lower-face codec retains appearance and frame-to-frame motion; 25 Hz physical timestamps replace normalized clip time. "
        "Audio and visual temporal adapters enter a shared field at corresponding times. Independent alignment removes the other modality and omits absolute positions. "
        "A same-recording shifted-window objective explicitly trains synchronization. Missing-modality prediction trains the two output directions. "
        "The legacy packing and no-local-alignment arms isolate these changes, while vanilla and no-spectral arms test whether signal blocks are necessary.","",
        "Research references: [AV-HuBERT](https://arxiv.org/abs/2201.02184) motivates masked multimodal representation learning; "
        "[SyncNet](https://github.com/joonson/syncnet_python) motivates same-recording timing negatives. This is an independently implemented small pilot, not a reproduction of their published results.","",
        "## Data and accounting","",
        f"[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D), source `{card['source_revision']}`, ODbL database / DbCL contents. "
        f"Usable split counts: `{card['split_counts']}`. People: `{ {s:len(v) for s,v in card['actor_splits'].items()} }`. "
        f"Selected {card['selected']} recordings; excluded {len(card['failures'])} failed/short clips. Retained hours: `{card['hours']}`. "
        "All old test people remain excluded from training and validation; all eight new test people were untouched in previous cycles.","",
        f"The codec's {card['feature_spec']['fitted_codec_values']:,} PCA/statistical values fit only training people and are counted separately. "
        "Feature dimensions are audio128 (log-mel and changes), visual160 (appearance PCA96 and motion PCA64). "
        "The core predicts simultaneous missing modalities; these metrics do not claim causal future prediction. "
        "Sentence metadata only excludes false negatives among repeated prompts in contrastive training; it is never an encoder input. "
        "This is weakly informed alignment, not a claim of fully label-free learning.","",
        f"Each run uses {cfg['training']['steps']} optimizer steps × {cfg['training']['batch_size']} independent clips, 32-frame random physical-time windows, "
        "BF16 autocast with FP32 weights/AdamW. Parameter, data and step budgets are explicit; wall time/FLOPs are not claimed equal. "
        f"Measured train-loop time totals {decision['training_seconds']/60:.1f} minutes; preparation, probes and rendering are additional.","",
        "## Frozen confirmation results","",
        "| Run | Architecture | Seed | Parameters | Timing correct | Energy control | Video→audio MSE ↓ | Audio→motion MSE ↓ | Seconds | Best step |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append(f"| {row['id']} | {row['arm']} | {row['seed']} | {row['parameters']:,} | {row['sync_accuracy']:.1%} | {row['sync_energy_control']:.1%} | {row['v2a_mel_mse']:.4f} | {row['a2motion_mse']:.4f} | {row['seconds']:.1f} | {row['best_step']} |")
    lines += ["", "Timing selection chooses one of seven independently encoded valid equal-length windows at shifts −9, −6, −3, 0, 3, 6, 9 frames. "
        "Chance is 14.3%. Only clips long enough for every candidate enter this metric; counts are saved per run. "
        "Identical padding/duration cannot reveal the answer. Retrieval uses independent encodes and all test clips as candidates; "
        "true-recording and sentence-level matches are separately reported in metrics. No fused retrieval is used.","",
        "## Gates and input dependence","",
        f"Synchronization passes both large seeds: **{decision['synchronization_pass_both_large_seeds']}**. "
        f"Video→audio decoding: **{decision['video_to_audio_pass_both_large_seeds']}**. "
        f"Audio→motion decoding: **{decision['audio_to_motion_pass_both_large_seeds']}**. "
        f"Audio corruption checks: **{decision['causal_audio_checks_pass_both_large_seeds']}**.","",
        "Registered decoding gates require ≥5% improvement over the training mean plus ≥2% gain from the correct input over a within-person wrong clip, "
        "with a positive 95% person-cluster interval. Confidence intervals resample people, not frames. See each run's `gates` and contribution intervals."]
    if causal:
        lines += ["","| Model | Original audio | Zeroed | Wrong clip, same person | Reversed time | Envelope only |","|---|---:|---:|---:|---:|---:|"]
        for key,value in causal.items():
            c=value["conditions"]
            lines.append(f"| {key} | {c['original']['accuracy']:.1%} | {c['zeroed_audio']['accuracy']:.1%} | {c['same_person_wrong_audio']['accuracy']:.1%} | {c['reversed_audio']['accuracy']:.1%} | {c['envelope_only_audio']['accuracy']:.1%} |")
        lines += ["","These are frozen input-corruption diagnostics. They qualify the timing claim without tuning any model or threshold. "
            "Even a genuine synchronization gain may reflect speech energy/onset cues rather than semantic understanding."]
    scale=decision.get("scale_seed41",{})
    if scale:
        lo,hi=scale["person_cluster_gain_interval"]
        lines += ["","## Does the additional capacity help?","",
            f"Seed 41 larger-versus-small timing gain: {scale['synchronization_absolute_gain']:+.1%}, person-cluster 95% interval [{lo:+.1%}, {hi:+.1%}]. "
            f"Video→audio error improvement: {scale['video_to_audio_relative_error_gain']:+.1%}. Train time ratio: {scale['train_time_ratio']:.2f}×. "
            "The small model has one seed; this is a controlled pilot, not a scaling law. Larger full/vanilla models have two seeds. "
            "Do not attribute the gain to parameter count when stronger features or the local objective explain it."]
    lines += ["","## What the model generates","",
        "The saved model directly predicts log-mel and lower-face features. Audio input produces a coarse mouth reconstruction and a separate rendering of predicted motion on a training-mean face. "
        "Video input produces predicted mel features rendered as a waveform using newly estimated Griffin-Lim phase. "
        "PCA reconstruction and same-modality reconstruction are codec/model controls. The original audio or video is never substituted into a generated output. "
        "The results are internal diagnostic proposals, not policy-committed behavior, realistic people, a neural vocoder, or open language generation.","",
        "Models used in the gallery are selected by validation: small full model, best-validation larger full seed, and its matched larger vanilla baseline. "
        "Examples include lowest/median/highest validation prediction error and fixed clips from three new test people. No test cherry-picking drives model choice.",""]
    if gallery:
        gallery_lines=["# Conditional output gallery — X0-M1","",
            "Actual saved-model outputs, with original/codec/null controls. All synthesized WAVs use peak normalization for listening; do not infer calibrated loudness. "
            "Mouth appearance is low-resolution inverse PCA. Audio-derived identity is unknown. Recognized sentences are one of 12 known prompts, not open text generation.",""]
        for example in gallery:
            case=Path(example["folder"]).relative_to(root/"exploration")
            gallery_lines += [f"## {example['experiment_id']} · {example['sample_id']} · {example['selection']}","",
                f"Expected prompt: {example['actual_sentence']} · acted emotion {example['actual_emotion']} · {example['duration_seconds']:.2f}s.","",
                f"- [Real audio]({case}/reference_audio.wav) / [codec audio]({case}/codec_audio.wav) / [model audio reconstruction]({case}/audio_reconstruction.wav).",
                f"- [Video→generated audio]({case}/video_to_audio.wav) / [no-input audio]({case}/null_audio.wav).",
                f"- [Audio→mouth]({case}/audio_to_mouth.gif) / [audio→integrated motion]({case}/audio_to_motion.gif).",
                f"- [Real mouth]({case}/reference_mouth.gif) / [codec mouth]({case}/codec_mouth.gif) / [model video reconstruction]({case}/video_reconstruction.gif) / [no-input mouth]({case}/null_mouth.gif).",""]
            for modality in ("audio","video"):
                p=example["probes"][modality+"_sentence"]
                gallery_lines.append(f"{modality.capitalize()} closed-set readout: “{p['text']}” ({p['confidence']:.0%} calibrated class probability).")
            if (Path(example["folder"])/"audio_linear_mouth.gif").exists():
                gallery_lines += ["",f"Additional frozen linear-head diagnosis: [audio→mouth]({case}/audio_linear_mouth.gif), "
                    f"[video→mouth reconstruction]({case}/video_linear_mouth.gif), [audio reconstruction]({case}/audio_linear_audio.wav), "
                    f"[video→audio]({case}/video_linear_audio.wav). These separately fitted heads do not replace the original model outputs."]
            gallery_lines.append("")
        (root/"exploration/gallery.md").write_text("\n".join(gallery_lines))
        chosen=next(e for e in gallery if e["experiment_id"]==min(large,key=lambda r:r["validation"]["selection_score"])["experiment_id"] and e["selection"]=="median_error")
        relative=Path(chosen["folder"]).relative_to(root)
        lines += [f"Example `{chosen['sample_id']}` from `{chosen['experiment_id']}`:","",
            f"- [Audio→mouth video]({relative}/audio_to_mouth.gif) and [integrated predicted motion]({relative}/audio_to_motion.gif).",
            f"- [Video→generated audio]({relative}/video_to_audio.wav).",
            f"- [Original audio]({relative}/reference_audio.wav), [codec audio]({relative}/codec_audio.wav), [audio reconstruction]({relative}/audio_reconstruction.wav), [no-input audio]({relative}/null_audio.wav).",
            f"- [Original mouth]({relative}/reference_mouth.gif), [codec mouth]({relative}/codec_mouth.gif), [video reconstruction]({relative}/video_reconstruction.gif), [no-input mouth]({relative}/null_mouth.gif).",
            "","[Complete output gallery](exploration/gallery.md). The machine-readable gallery index `exploration/gallery.json` and each `example.json` contain exact provenance, conditioning, expected prompt and frozen-probe outputs. "
            "Standalone inference accepts either `--audio` or `--video`; its default output head selects the other modality."]
    lines += ["","## Frozen recognition probes","",
        "The text readout is a separately fitted, frozen-core classifier over 12 known sentence prompts and six emotion categories. "
        "It is closed-set recognition, not transcription or free text generation. Ridge probes and validation-only temperature calibration are saved separately. "
        "Raw-feature, shuffled-training-label and class-prior controls limit interpretation.","",
        "| Model | Input | Task | Learned test accuracy | Raw feature control | Shuffled labels | Class prior | ECE |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    for folder in sorted((root/"exploration").glob("EXP-T2-*")) if (root/"exploration").exists() else []:
        file=folder/"probe_metrics.json"
        if not file.exists(): continue
        probes=json.loads(file.read_text())
        for key,value in probes["probes"].items():
            modality,task=key.split("_")
            lines.append(f"| {folder.name} | {modality} | {task} | {value['learned']['test']['accuracy']:.1%} | {value['raw_feature_control_test']['accuracy']:.1%} | {value['shuffled_training_labels_test']['accuracy']:.1%} | {value['prior_test_accuracy']:.1%} | {value['learned']['test']['ece']:.3f} |")
    lines += ["","## Failure cases, artifacts and limits","",
        "All per-clip errors and confidence intervals remain in run metrics. Coarse PCA mouth output cannot establish an unseen person's appearance. "
        "Integrating predicted motion can drift; MSE often favors a generic face or nearly zero motion. Griffin-Lim audio can be distorted even when its mel error improves. "
        "Listen to the codec ceiling alongside generated sound. Timing correspondence does not establish intelligible speech, grounded captioning, physical prediction, "
        "general world modeling, or hallucination reduction. Fixed face crops and acted speech constrain external validity.","",
        "Each run retains exact config, source/config/data hashes, optimizer/RNG state, validation-selected checkpoint, curves and registry. "
        "The source snapshot preserves the frozen training code. X0 records its own source, checkpoint and codec hashes. "
        "`comparison.csv`, `decision.json`, `causal_checks.json`, recognition probe weights, media and gallery are all retained.","",
        "## Next research decision","",
        ("Retain the alignment module and validate on a second, more diverse physical-time audiovisual domain at the smallest effective size. "
         "Prioritize intelligibility and causal motion prediction before another parameter increase."
         if decision["synchronization_pass_both_large_seeds"] and decision["causal_audio_checks_pass_both_large_seeds"] else
         "Resolve the timing controls before expanding model size or making a generation capability claim."),""]
    asr_file=root/"exploration/intelligibility.json"
    linear_file=root/"exploration/linear_decoder_metrics.json"
    if linear_file.exists():
        linear=json.loads(linear_file.read_text())
        lines += ["## Exploratory output-head diagnosis","",
            "After seeing the averaged mouth outputs, fixed-ridge linear heads were fitted to the frozen world field using training frames only. "
            "Core weights and all preregistered results stay unchanged. These supervised diagnostic readouts add separately counted coefficients; "
            "they test whether an alternative head can recover detail, not whether a new world model passed confirmation.","",
            "| Model | Head | Extra fitted parameters | Audio reconstruction MSE | Silent video→audio MSE | Audio→motion MSE | Visible-video appearance MSE |",
            "|---|---|---:|---:|---:|---:|---:|"]
        for identity,value in linear.items():
            original=next(r for r in runs if r["experiment_id"]==identity)["test"]["errors"]
            for name,e,count in (("original",original,0),("frozen linear probe",value["test"],value["fitted_parameters"])):
                lines.append(f"| {identity} | {name} | {count:,} | {e['audio_input_audio_mel']:.4f} | {e['video_input_audio_mel']:.4f} | {e['audio_input_video_motion']:.4f} | {e['video_input_video_appearance']:.4f} |")
        lines += ["","Original heads are already included in the model budget; the extra-count column counts only additional probe weights. "
            "The alternative readout improves some observed-signal reconstruction but does not resolve audio-conditioned motion or unknown appearance. "
            "The corresponding `audio_linear_*` and `video_linear_*` media are saved beside each selected median validation example."]
    if asr_file.exists():
        asr=json.loads(asr_file.read_text())
        lines += ["","## Optional external speech-intelligibility diagnostic","",
            "[Whisper tiny.en](https://huggingface.co/openai/whisper-tiny.en) is used only as an external, fallible recognizer of already-saved waveforms; "
            "it never generates, repairs or supplies the model audio or the closed-set recognition probe.",""]
        if asr.get("status")=="completed":
            lines += [f"External evaluation-only model parameters: {asr['external_evaluation_parameters']:,}; revision `{asr['revision']}`. "
                "These are exploratory macro word-error rates on the selected gallery, not a representative ASR benchmark or human listening judgment.","",
                "| Audio condition | Mean recognizer word-error rate ↓ |","|---|---:|"]
            for name,value in asr["conditions_mean_wer"].items(): lines.append(f"| {name} | {value:.1%} |")
            lines += ["","Original and codec errors distinguish recognizer/codec limitations from model errors. A distorted or null waveform can make the recognizer "
                "invent a phrase; those transcripts are labeled recognizer outputs and must not be represented as generated model text."]
        else: lines += ["The optional recognizer check was unavailable. Its error is retained; no primary result depends on it."]
    (root/"report.md").write_text("\n".join(lines)+"\n")
    plot(root,runs,cfg,causal)
    return root/"report.md"


def plot(root,runs,cfg,causal):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout="constrained")
    for r in runs:
        curves=json.loads((root/r["experiment_id"]/"curves.json").read_text())["curves"]
        points=[c for c in curves if "validation" in c]
        label=r["run"]["arm"]+" / "+str(r["run"]["seed"])
        axes[0].plot([p["step"] for p in points],[p["validation"]["synchronization"]["accuracy"]*100 for p in points],label=label)
        axes[1].plot([p["step"] for p in points],[p["validation"]["errors"]["video_input_audio_mel"] for p in points])
        axes[2].plot([p["step"] for p in points],[p["validation"]["errors"]["audio_input_video_motion"] for p in points])
    for ax,title,ylabel in zip(axes,("Same-clip timing alignment","Silent video → audio","Audio → motion"),("Correct offset (%) ↑","Validation mel MSE ↓","Validation motion MSE ↓")):
        ax.set(title=title,xlabel="Optimizer steps",ylabel=ylabel)
    axes[0].axhline(100/7,ls="--",color="gray",lw=1)
    axes[0].legend(fontsize=7,ncol=2)
    fig.suptitle("SLWM physical-time motion architecture · validation selects checkpoints")
    fig.savefig(root/"learning_curves.png",dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout="constrained")
    labels=[r["run"]["arm"].replace("_"," ")+"\nseed "+str(r["run"]["seed"]) for r in runs]
    accuracy=[r["test"]["synchronization"]["accuracy"]*100 for r in runs]
    colors=["#315c9b" if r["run"]["arm"].startswith("signal") else "#91a6ae" for r in runs]
    axes[0].bar(np.arange(len(runs)),accuracy,color=colors)
    intervals=[r["test"]["synchronization"]["person_cluster_accuracy_interval"] for r in runs]
    errors=np.array([[max(0,acc-100*ci[0]) for acc,ci in zip(accuracy,intervals)],
                     [max(0,100*ci[1]-acc) for acc,ci in zip(accuracy,intervals)]])
    axes[0].errorbar(np.arange(len(runs)),accuracy,yerr=errors,fmt="none",ecolor="#243745",capsize=3,lw=1)
    axes[0].axhline(100/7,ls="--",color="#b77d3d",label="Chance: 14.3%")
    axes[0].axhline(runs[0]["test"]["synchronization"]["motion_energy_baseline_accuracy"]*100,ls=":",color="#47846b",label="Untrained energy/motion control")
    axes[0].set(xticks=np.arange(len(runs)),xticklabels=labels,ylabel="Correct timing offset (%) ↑",title="Independent confirmation on new people",ylim=(0,100))
    axes[0].tick_params(axis="x",labelrotation=65,labelsize=7); axes[0].legend(fontsize=8,loc="lower right")
    for r in runs:
        e=r["test"]["errors"]
        marker="o" if r["run"]["model"]["processor"]=="signal" else "s"
        axes[1].scatter(r["parameter_counts"]["total"]/1e6,e["video_input_audio_mel"],s=65,marker=marker,
                        color="#315c9b" if r["run"]["arm"].startswith("signal") else "#91a6ae")
        if r["run"]["arm"]=="signal_small":
            axes[1].annotate("SLWM 4.74M",(r["parameter_counts"]["total"]/1e6,e["video_input_audio_mel"]),fontsize=8,xytext=(6,8),textcoords="offset points")
    axes[1].set(xlabel="All trainable parameters (millions)",ylabel="Silent-video audio MSE ↓",title="Capacity is measured separately from alignment")
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[Line2D([],[],marker="o",ls="none",color="#315c9b",label="Full SLWM"),
                           Line2D([],[],marker="s",ls="none",color="#91a6ae",label="Vanilla Transformer"),
                           Line2D([],[],marker="o",ls="none",color="#91a6ae",label="Ablation / control")],loc="upper right",fontsize=8)
    fig.suptitle("Alignment and scaling · same real data and optimizer-step budgets")
    fig.savefig(root/"alignment_and_scaling.png",dpi=170); plt.close(fig)
    if causal:
        chosen=[r for r in runs if r["run"]["arm"] in {"signal_small","signal_large"}]
        conditions=["original","zeroed_audio","same_person_wrong_audio","reversed_audio","envelope_only_audio"]
        fig,ax=plt.subplots(figsize=(9,4.3),layout="constrained")
        x=np.arange(len(conditions)); width=.24
        for i,r in enumerate(chosen):
            values=[causal[r["experiment_id"]]["conditions"][c]["accuracy"]*100 for c in conditions]
            ax.bar(x+(i-1)*width,values,width=width,label=r["experiment_id"]+" · "+str(round(r["parameter_counts"]["total"]/1e6,2))+"M")
        ax.axhline(100/7,color="gray",ls="--")
        ax.set(xticks=x,xticklabels=["Original\naudio","Zeroed\naudio","Wrong clip\nsame person","Reversed\ntime","Energy\nprofile only"],ylabel="Correct offset (%) ↑",ylim=(0,100),title="Does alignment depend on the actual sound?")
        ax.legend(fontsize=8); fig.savefig(root/"input_dependence.png",dpi=170); plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True)
    print(report(parser.parse_args().config))
