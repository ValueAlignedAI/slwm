"""Curated G1 findings from frozen outputs; no fitting, filtering or new gates."""
import html
import json
from pathlib import Path

import numpy as np

from training.general_sources import write
from training.general_speech_eval import paired_wer
from training.research_sources import file_sha


def load(path):return json.loads(Path(path).read_text())


def report():
    cfg=load("configs/t2/general_transfer.json");root=Path(cfg["root"])
    transfer=load(root/"readouts/evaluation_summary.json");speech=load(root/"speech/summary.json");alignment=load(root/"alignment/summary.json");long=load(root/"long_media/summary.json")
    completion={r["id"]:load(root/r["id"]/"complete.json") for r in cfg["runs"]}
    recognized={r["id"]:load(root/"speech/evaluation"/r["id"]/"recognized.json") for r in cfg["runs"] if (root/"speech/evaluation"/r["id"]/"recognized.json").exists()}
    def primary(rows):return [r for r in rows if r["family"]=="speech" and r["condition"]=="audio" and r["gap_seconds"]==.16]
    specialist=primary(recognized["G1-111"]);noninferiority={};processor={}
    for name in ("G1-101","G1-103"):
        rows=primary(recognized[name]);effect=paired_wer(rows,specialist)
        fixed=[r for r in recognized[name] if r["family"]=="speech" and r["condition"]=="interpolation" and r["gap_seconds"]==.16]
        baseline=paired_wer(rows,fixed)
        noninferiority[name]={"versus_speech_specialist":effect,"versus_interpolation":baseline,"within_one_point_with_group_interval":bool(effect["gain_95ci"][0]>=-.01 and baseline["gain_95ci"][0]>=-.01)}
    for a,b in [("G1-101","G1-102"),("G1-103","G1-104")]:processor[a+"_vs_"+b]=paired_wer(primary(recognized[a]),primary(recognized[b]))
    broad=bool(transfer["transfer_gates"]["mhealth/activity"] and transfer["transfer_gates"]["droid/actor"] and all(v["within_one_point_with_group_interval"] for v in noninferiority.values()))
    gates={"new_source_activity_both_slwm_seeds":transfer["transfer_gates"]["mhealth/activity"],"new_task_actor_both_slwm_seeds":transfer["transfer_gates"]["droid/actor"],"speech_within_one_point_both":all(v["within_one_point_with_group_interval"] for v in noninferiority.values()),"registered_broad_representation_gate":broad,"speech_improvement_both":speech["both_slwm_seeds_speech"],"video_adds_content_and_rejects_mistiming_both":speech["both_slwm_seeds_video"]}
    operational={r["id"]:load(root/"operational"/r["id"]/"summary.json") for r in cfg["runs"] if (root/"operational"/r["id"]/"summary.json").exists()}
    cost={name:{"parameters":r["parameters"],"committed_optimizer_updates":r["optimizer_updates"],"schedule_position":r["position"],"target_reached":r["target_reached"],"active_training_hours":r["training_seconds"]/3600,"validation_hours":r["validation_seconds"]/3600,"selected_position":r["selected_position"],"actual_family_calls":r["family_calls"]} for name,r in completion.items()}
    angular=load(root/"angular_pilot_v2/summary.json") if (root/"angular_pilot_v2/summary.json").exists() else None
    result={"gates":gates,"speech_noninferiority":noninferiority,"paired_processor_speech_effects":processor,"cost":cost,"operational":operational,"long_media":long,"selection_only_angular_repair_pilot":angular,
            "source_evaluation_lock_sha256":file_sha(root/"evaluation_lock.json"),"vocoder_freshness_scope":"Fresh relative to shared cores and R1; upstream LibriTTS-vocoder speaker exposure unverified",
            "scope":"Tests of reusable representations, offline imitation and recorded consequences; not a general world-model or physical closed-loop certification"}
    write(root/"findings.json",result);write(root/"gates.json",gates)
    lines=["# G1 cross-source shared-state findings","", "The primary question is whether one frozen processor transfers beyond its training tasks. Speaker-disjoint speech alone cannot answer it.","", "| Registered check | Result |","|---|---|"]
    lines += ["| "+name+" | "+("PASS" if value else "FAIL / insufficient evidence")+" |" for name,value in gates.items()]
    lines += ["","## Frozen new-source/new-task readouts","","| Task / model | Primary error | Strongest selection baseline | Paired gain95% interval |","|---|---:|---|---|"]
    for task in ("mhealth/activity","droid/actor"):
        value=transfer["tasks"][task]
        for name in ("G1-101","G1-102","G1-103","G1-104", "G1-101-aligned","G1-101-pooled"):
            effect=value["paired_effects"][name];interval=effect["gain_95ci"]
            lines.append(f"| {task} / {name} | {effect['candidate_error']:.4f} | {value['strongest_selection_baseline']} | [{interval[0]:.4f}, {interval[1]:.4f}] |")
    lines += ["","Activity error is one minus macro balanced accuracy. Actor error is mean absolute command error normalized with training-target scales; paired regression effects use independent environment-group means. Raw physical translation/SO(3)/gripper errors, both readout seeds, all controls and the forecast task remain in `readouts/evaluation_summary.json`. The aligned/pooled probes are prospective secondary conditions and do not replace a failed primary gate.","","## Intelligibility and alignment","","| Run | Primary speech WER | Gain over interpolation | Aligned-video WER | Audio-only meeting WER | Worst shifted-video harm |","|---|---:|---:|---:|---:|---:|"]
    for name in ("G1-101","G1-102","G1-103","G1-104","G1-111"):
        item=speech["runs"][name];s=item["speech_gain_over_interpolation"];av=item["correct_video_gain_over_audio"]
        lines.append(f"| {name} | {100*s['candidate_wer']:.2f}% | {100*s['gain']:.2f}pp | {100*av['candidate_wer']:.2f}% | {100*av['baseline_wer']:.2f}% | {100*item['worst_mistimed_video_wer_harm']:.2f}pp |")
    lines += ["","These are fallible independent-recognizer scores. Libri intervals resample people; AMI intervals resample connected meeting components. Owner-empty windows retain separate insertion counts and may contain other speakers. All fixed quiet/overlap cases and64/320ms diagnostics remain. Window-local CTC and whole-recording actual model CTC outputs are labelled separately from recognizer text.","","Freshness applies to our cores/R1, with no held final voice used for fitting or calibration. The external Vocos renderer was trained on LibriTTS derived from LibriSpeech; its exact held-speaker exposure is unverified. See the upstream exposure audit. All architectures and fixed controls share that decoder; core phone/CTC results supply a separate content check.","","## Unseen modality combinations and decisions","","`alignment/summary.json` compares independently encoded sound→image semantic categories with two specialists joined through fixed text prototypes. This is category agreement across unpaired datasets, not identification of matched physical events. Rare categories and all frozen counts remain visible.","","`consequences/` compares actual-plan predictions at1/4/8 ordinal steps with persistence, last velocity and commanded-pose-as-state, plus removed/shuffled plans. Wrong-plan inputs test prediction dependence; the actual counterfactual physical outcomes are unobserved. Image-motion error is separate from static background error. A next-command readout is offline imitation, not proof of useful planning or reward optimization.","","`operational/` records over40s of paced repeated observations/proposals, stale-packet rejection, latency/deadlines and exact weight invariance. The10Hz consumer is chosen for replay; source robot cadence is unverified. Recorded future inputs do not respond to proposals. A real ongoing controller still needs causal feedback, goals/costs, calibrated uncertainty and intervention tests.","","## Scale, compute and next decisions","","| Run | Parameters | Updates | Training+validation hours |","|---|---:|---:|---:|"]
    for name,item in cost.items():lines.append(f"| {name} | {item['parameters']:,} | {item['committed_optimizer_updates']:,} | {item['active_training_hours']+item['validation_hours']:.2f} |")
    if broad:lines += ["","The registered reuse gate passes on these sources/tasks. The next confirmation is longer memory and action-dependent feedback on new environments, with an explicit predictive bottleneck and independent goals. This limited study does not establish general world understanding."]
    else:lines += ["","The registered broad reuse gate does not pass. Preserve any task-specific gains, but do not promote the current representation as general. Inspect source-wise negative transfer, the matched pooling/alignment controls and action/decoder ceilings before increasing parameter count. Use new held data for a changed architecture's confirmation."]
    lines += ["","Compare the nonlinear SLWM path with the matched latent Transformer and fixed-modality Transformer, including its added time/memory cost. Two sizes, with one small-model seed, do not support a scaling law. Specialists match target-family calls and processor capacity but have lower per-model training cost; their aggregate storage is a different comparison. The independent500M-token language models are not shared multimodal weights.","","The current forecaster conditions each future query on that query's command, rather than the entire preceding action prefix. Accumulated action effects need a causal plan encoder/recurrent rollout next. An advantage over the Transformer attributes the whole added nonlinear path, not Fourier coordinates alone; a direct-time FIR parameterization is a further necessary control.","","The separate angular-repair pilot uses training/selection data only and does not replace the registered primary models or gates. Its first attempt stopped before any optimizer update because a diagnostic covariance operation entered BF16; the versioned precision fix and failure record are retained.","","Every endpoint, selected position, target exposure, source snapshot and final lock is retained. EMA teachers, the289,860-parameter R1 edge,13,531,650-parameter Vocos decoder and37,760,256-parameter evaluation-only Whisper scorer remain separate from core inference counts. No untrained action head is presented as a policy.",""]
    (root/"REPORT.md").write_text("\n".join(lines))
    page=["<!doctype html><meta charset=utf-8><title>G1 raw long outputs</title><style>body{font:16px system-ui;margin:32px;max-width:1200px}article{display:inline-block;vertical-align:top;max-width:290px;margin:8px;padding:12px;background:#eee}video{width:256px;image-rendering:pixelated}p{max-width:1000px}</style><h1>G1 raw40.96-second outputs</h1><p>All fixed cases, including failures. Native generated video is16×16 RGB. Audio→video uses the reference audio soundtrack for display. Video→audio keeps the input video for display and synthesizes speech with a fixed training-mean voice. Text-only generation and audio→image are diagnostic directions; no quality claim is assumed. Oracle decoder videos are privileged ceilings.</p>"]
    for item in load(root/"long_media/cases.json"):
        folder=Path(item["folder"]);relative=folder.relative_to(root);label=item["run"]+" / "+item["id"]+" / "+item["mode"]+" / "+item["control"]
        page.append('<article><h3>'+html.escape(label)+'</h3><video controls preload="none" src="'+html.escape(str(relative/"output.mp4"))+'"></video><p>'+html.escape(item.get("provided_text",item.get("reference","")))+'</p><small>Actual field text: '+html.escape(item.get("field_text",""))+'</small></article>')
    (root/"watch.html").write_text("\n".join(page))
    write(root/"study_complete.json",{"all_registered_phases_complete":True,"findings_sha256":file_sha(root/"findings.json"),"report_sha256":file_sha(root/"REPORT.md"),"gates":gates,"no_new_fit_after_final_lock":True})


if __name__=="__main__":report()
