"""All-case, person-clustered reporting; this module never selects new weights."""
from collections import defaultdict
import html
import json
from pathlib import Path
import numpy as np
from training.t2_real_av import write_json


def interval(values,people,seed=291):
    people=np.asarray(people); values=np.asarray(values,float); unique=sorted(set(people)); means=np.asarray([values[people==p].mean() for p in unique]); rng=np.random.default_rng(seed); draws=means[rng.integers(len(means),size=(10000,len(means)))].mean(1)
    return {"person_mean":float(means.mean()),"person_cluster_95_interval":np.quantile(draws,[.025,.975]).tolist(),"people":len(unique),"cases":len(values)}


def recognition_summary(rows):
    """Keep no-reference hallucinations explicit instead of hiding them in WER."""
    with_words=[r for r in rows if r["normalized_reference_words"]>0]
    without_words=[r for r in rows if r["normalized_reference_words"]==0]
    return {"cases":len(rows),"nonempty_reference_cases":len(with_words),
            "mean_clip_wer":float(np.mean([r["wer"] for r in rows])) if rows else None,
            "nonempty_reference_mean_clip_wer":float(np.mean([r["wer"] for r in with_words])) if with_words else None,
            "nonempty_reference_corpus_wer":sum(r["normalized_edit_count"] for r in with_words)/sum(r["normalized_reference_words"] for r in with_words) if with_words else None,
            "no_reference_cases":len(without_words),
            "no_reference_insertions":sum(r["normalized_edit_count"] for r in without_words),
            "reference_words":sum(r["normalized_reference_words"] for r in rows)}


def content_summaries(root,recognized,long_cases):
    field_rows=json.loads((root/"evaluation/field_ctc_scores.json").read_text()); grouped=defaultdict(list)
    for row in field_rows:
        key="|".join(str(row.get(k)) for k in ("run","group","split","mode","control","gap_seconds"))
        grouped[key].append({**row,"normalized_edit_count":row["normalized_edits"]})
    ctc={k:recognition_summary(v) for k,v in grouped.items()}
    write_json(root/"field_content_summary.json",{"method":"Actual learned field CTC, separate from external waveform recognition; text-conditioned cases disclose supplied content","groups":ctc})
    groups=defaultdict(list)
    for row in recognized:
        if row["group"] in ("long","long_control","text"): groups[row["group"]+"|"+row["condition"]].append(row)
    recognition={k:recognition_summary(v) for k,v in groups.items()}
    conditions=defaultdict(list)
    for row in long_cases: conditions[row["run"]+"|"+row["mode"]+"|"+row["control"]].append(row)
    effects={}
    for key,rows in conditions.items():
        people=[r["actor"] for r in rows]
        effects[key]={"cases":len(rows),"code_mse":interval([r["video_code_mse"] for r in rows],people),
                      "motion_mse":interval([r["motion_error_mse"] for r in rows],people),
                      "code_gain_over_provided_static":interval([r["video_persistence_mse"]-r["video_code_mse"] for r in rows],people),
                      "motion_gain_over_static":interval([r["static_motion_error_mse"]-r["motion_error_mse"] for r in rows],people)}
    write_json(root/"long_condition_summary.json",{"waveform_recognition":recognition,"all_world_mode_controls":effects})
    return recognition,ctc


def speech_summary(cfg,root):
    records=[r for r in json.loads((root/"evaluation/recognizer_results.json").read_text()) if r["group"]=="speech"]; result={}
    for split in ("test-clean","test-other"):
        result[split]={}
        for width in (.16,.32):
            rows=[r for r in records if r["split"]==split and r["gap_seconds"]==width]; grouped=defaultdict(list)
            for r in rows: grouped[r["condition"]].append(r)
            for condition in grouped: grouped[condition].sort(key=lambda r:r["id"])
            means={k:float(np.mean([r["wer"] for r in v])) for k,v in grouped.items()}; corpus={k:sum(r["normalized_edit_count"] for r in v)/sum(r["normalized_reference_words"] for r in v) for k,v in grouped.items()}; baseline=min(("silence","interpolated_mel"),key=lambda k:means[k]); comparisons={}
            for run in cfg["runs"]:
                name=run["id"]; people=[r["actor"] for r in grouped[name]]; delta=np.asarray([r["wer"] for r in grouped[baseline]])-np.asarray([r["wer"] for r in grouped[name]]); effect=interval(delta,people); effect["relative_reduction"]=effect["person_mean"]/max(means[baseline],1e-9); effect["passes_primary_gate"]=bool(means[baseline]>0 and effect["relative_reduction"]>=.05 and effect["person_cluster_95_interval"][0]>0); comparisons[name]=effect
            result[split][str(width)]={"mean_clip_wer":means,"corpus_wer":corpus,"stronger_registered_baseline":baseline,"paired_effects":comparisons}
    write_json(root/"speech_summary.json",result); return result


def policy_summary():
    root=Path("experiments/policy/continuous-sensing-20261105"); cfg=json.loads(Path("configs/policy/stream_sensing.json").read_text()); summary={}
    for run in cfg["world_runs"]:
        rows=[]
        for p in (root/run).glob("*_*.json"):
            if p.name=="paced_cpu.json": continue
            value=json.loads(p.read_text())
            if isinstance(value,dict) and "rows" in value: rows.extend(value["rows"])
        grouped=defaultdict(list)
        for r in rows: grouped[r["strategy"]].append(r)
        means={k:float(np.mean([r["total_cost"] for r in v])) for k,v in grouped.items()}; best=min((k for k in grouped if k not in ("learned","raw")),key=lambda k:means[k]); person={}
        for strategy in grouped:
            person[strategy]={p:float(np.mean([r["total_cost"] for r in grouped[strategy] if r["actor"]==p])) for p in sorted({r["actor"] for r in rows})}
        people=list(person["learned"]); gain=[person[best][p]-person["learned"][p] for p in people]; effect=interval(gain,people); coverage=float(np.mean([r["committed"] for r in grouped["learned"]])); committed=[r["error"] for r in grouped["learned"] if r["committed"]]; action_counts={a:sum(r["action"]==a for r in grouped["learned"]) for a in cfg["actions"]}; summary[run]={"mean_total_cost":means,"strongest_fixed":best,"effect":effect,"coverage":coverage,"conditional_committed_error":float(np.mean(committed)) if committed else None,"actions":action_counts,"passes_gate":bool(effect["person_mean"]>0 and effect["person_cluster_95_interval"][0]>0 and coverage>=.5),"person_costs":person,"goals":{g:{k:float(np.mean([r["total_cost"] for r in v if r["goal"]==g])) for k,v in grouped.items()} for g in cfg["goals"]}}
        summary[run]["bootstrap_sensor_cost_per_episode"]=.1; summary[run]["risk_absolute_calibration_error"]=float(np.mean([abs(r["predicted_error"]-r["error"]) for r in grouped["learned"]])); summary[run]["activity_conditioned_tick_means"]={}
        representation=interval([person["raw"][p]-person["learned"][p] for p in people],people); summary[run]["field_vs_direct_sensor_statistics"]=representation; summary[run]["field_representation_gate"]=bool(representation["person_cluster_95_interval"][0]>0 and coverage>=.5)
        for active in (False,True):
            conditional={}
            for strategy,v in grouped.items():
                selected=[r for r in v if r["future_speaker_active_evaluation_only"]==active]; conditional[strategy]={"ticks":len(selected),"mean_total_cost":float(np.mean([r["total_cost"] for r in selected])) if selected else None}
            summary[run]["activity_conditioned_tick_means"]["active" if active else "quiet"]=conditional
    write_json(root/"summary.json",summary); lines=["# P1-S1 — continuous sensing decisions from frozen shared fields","","Every automatic tick counts toward utility, including uncommitted predictions. A shared0.1 bootstrap sensor cost per episode is additional and equal across strategies. Actions select recorded sensors; they do not control physical motion. All world weights remain fixed and calibration people are separate from training, selection and final test people.","","| World run | Learned total cost | Direct-sensor actor | Strongest fixed / cost | Paired cost gain95% interval | Commitment coverage | Gate |","|---|---:|---:|---|---|---:|---|"]
    for run,v in summary.items():
        lo,hi=v["effect"]["person_cluster_95_interval"]; lines.append(f"| {run} | {v['mean_total_cost']['learned']:.4f} | {v['mean_total_cost']['raw']:.4f} | {v['strongest_fixed']} / {v['mean_total_cost'][v['strongest_fixed']]:.4f} | {v['effect']['person_mean']:+.4f} [{lo:+.4f},{hi:+.4f}] | {100*v['coverage']:.1f}% | {'PASS' if v['passes_gate'] else 'FAIL'} |")
    lines += ["","The strongest fixed baseline is the lowest cost across all registered fixed strategies. The actor receives current processed-field/memory means, model risk and previous sensor bits; future labels enter only train-only risk fitting and final scoring. Standardized128-hidden heads predict counterfactual future error after a2-packet guard. Goals are explicit audio/video/both enums, not learned language instructions. Summary JSON additionally reports active/quiet tick-conditional means and risk calibration error. Own-speaker activity is an evaluation-only label; primary utility retains every tick.","","A physical-action world model still needs action-conditioned physical transitions, goals/reward, multi-step planning and embodiment-specific data. This recorded-data sensing loop cannot establish those capabilities. See the paced CPU audit for automatic action ticks, deadlines and exact weight invariance."]
    (root/"report.md").write_text("\n".join(lines)+"\n"); return summary


def gallery(root):
    base=root/"evaluation"; rows=json.loads((base/"long_cases.json").read_text()); texts=json.loads((base/"text_cases.json").read_text()); page=["<!doctype html><meta charset=utf-8><title>SLWM continuous research outputs</title><style>body{font:16px system-ui;background:#111827;color:#eee;margin:32px}a{color:#93c5fd}video{width:256px;image-rendering:pixelated}article{display:inline-block;vertical-align:top;background:#1f2937;padding:16px;margin:8px;max-width:290px}small{display:block;color:#aaa}summary{cursor:pointer}p{max-width:1000px}</style><h1>Continuous shared-field research outputs</h1><p>All fixed cases, including failures. Video is64×64 predicted motion over a declared provided training appearance. Audio→video uses original supplied audio; video→audio generates the entire waveform using a frozen external Vocos renderer. Text is a phonetic edge with estimated timing. These are not arbitrary scene videos or evidence of robot control.</p>"]
    for label,items in (("Text → audio/video",texts),("All fresh-person audio/video cases",rows)):
        page.append("<h2>"+label+"</h2>")
        for row in items:
            folder=Path(row["folder"]); relative=folder.relative_to(root).as_posix(); title=row.get("id","prompt "+str(row.get("prompt")))+" · "+row["run"]+" · "+row["mode"]+" · "+row.get("control","normal"); page.append(f'<article><b>{html.escape(title)}</b><small>{row["waveform"]["seconds"]:.2f}s ·64×64 native</small><video controls preload="none" src="{relative}/output.mp4"></video><p><a href="{relative}/output.wav">PCM waveform</a> · <a href="{relative}/case.json">Metrics/provenance</a> · <a href="{relative}/trace.json">Continuous state trace</a></p><details><summary>Reference / provided text</summary>{html.escape(row["transcript"])}</details><details><summary>Actual learned field CTC output</summary>{html.escape(row["field_text"])}</details></article>')
    page.append("<h2>Speech repair</h2><p>Every original, silenced, linearly repaired, rendered control and learned repair is retained in evaluation/speech, with exact observed-PCM assertions, masks, field text and the pinned independent recognizer results.</p>"); (root/"watch.html").write_text("\n".join(page))


def av_repair_summary(recognized,root):
    cases=[r for r in recognized if r["group"]=="av_repair"]; result={}
    for run in ("EXP-T2-801","EXP-T2-802","EXP-T2-803","EXP-T2-804"):
        grouped=defaultdict(list)
        for r in cases:
            if r["run"]==run: grouped[r["control"]].append(r)
        for condition in grouped: grouped[condition].sort(key=lambda r:r["id"])
        people=[r["actor"] for r in grouped["audio_only"]]; audio=np.asarray([r["wer"] for r in grouped["audio_only"]]); aligned=np.asarray([r["wer"] for r in grouped["aligned_video"]]); effects={}
        for condition,rows in grouped.items(): effects[condition+"_minus_aligned"]=interval(np.asarray([r["wer"] for r in rows])-aligned,people)
        result[run]={"mean_clip_wer":{k:float(np.mean([r["wer"] for r in rows])) for k,rows in grouped.items()},"recognition":{k:recognition_summary(v) for k,v in grouped.items()},"paired_changes":effects,"nonempty_reference_people":sum(r["normalized_reference_words"]>0 for r in grouped["audio_only"]),"reference_words_by_person":{r["actor"]:r["normalized_reference_words"] for r in grouped["audio_only"]},"scope":"Secondary same-model fresh-person transfer probe, no new timing-rejection policy or primary gate"}
    write_json(root/"av_repair_summary.json",result); return result


def main():
    cfg=json.loads(Path("configs/t2/broad_stream.json").read_text()); root=Path(cfg["artifacts"]); selection=json.loads((root/"selection.json").read_text()); speech=speech_summary(cfg,root); policy=policy_summary(); field={r["id"]:json.loads((root/"evaluation"/(r["id"]+"_fields.json")).read_text()) for r in cfg["runs"]}; long=json.loads((root/"evaluation/long_cases.json").read_text()); recognized=json.loads((root/"evaluation/recognizer_results.json").read_text()); summaries={}; codec=dict(np.load(Path(cfg["corpus"])/"codec.npz")); fitted=sum(v.size for v in codec.values())+4
    for r in cfg["runs"]:
        key=r["id"]; f=field[key]; retrieval={}
        for direction in ("audio_to_video","audio_to_text","video_to_text"):
            rows=[x for x in f["retrieval"] if x["direction"]==direction]; people=[x["actor"] for x in rows]; retrieval[direction]={**interval([x["accuracy"] for x in rows],people),"person_mean_chance":float(np.mean([x["chance"] for x in rows])),"above_chance":interval([x["accuracy"]-x["chance"] for x in rows],people)}
        future=np.asarray([x["future_errors"] for x in f["windows"]]); persistence=np.asarray([x["persistence_errors"] for x in f["windows"]]); prior=np.asarray([x["training_mean_prior_errors"] for x in f["windows"]]); people=[x["actor"] for x in f["windows"]]
        summaries[key]={"selection":selection[key],"retrieval":retrieval,"future_mean_audio_mae_video_mse":future.mean(0).tolist(),"persistence_mean_audio_mae_video_mse":persistence.mean(0).tolist(),"training_mean_prior_audio_mae_video_mse":prior.mean(0).tolist(),"future_gain_over_persistence":[interval((persistence-future)[:,i],people) for i in range(2)],"future_gain_over_training_mean":[interval((prior-future)[:,i],people) for i in range(2)],"phone_accuracy_active":float(np.mean([x["phone_accuracy_active"] for x in f["windows"] if x["phone_accuracy_active"] is not None]))}
    write_json(root/"representation_summary.json",summaries); long_summary={}
    for run in ("EXP-T2-801","EXP-T2-802","EXP-T2-803","EXP-T2-804"):
        rows=[r for r in long if r["run"]==run and r["control"]=="normal"]; videos=[r for r in rows if r["mode"]=="audio"]; wer=[r["wer"] for r in recognized if r["group"]=="long" and r["condition"]==run+"_video_normal"]; long_summary[run]={"audio_to_video_code_mse":float(np.mean([r["video_code_mse"] for r in videos])),"provided_static_code_mse":float(np.mean([r["video_persistence_mse"] for r in videos])),"audio_to_video_motion_mse":float(np.mean([r["motion_error_mse"] for r in videos])),"static_motion_mse":float(np.mean([r["static_motion_error_mse"] for r in videos])),"video_to_audio_mean_clip_wer":float(np.mean(wer)),"native_generated_unique_frames":int(np.sum([r["video"]["unique_frames"] for r in videos])),"cases":len(rows)}
    recognition,ctc=content_summaries(root,recognized,long)
    for run,v in long_summary.items(): v["video_to_audio_recognition"]=recognition["long|"+run+"_video_normal"]
    write_json(root/"long_summary.json",long_summary); av=av_repair_summary(recognized,root); gates={"speech_slwm_both_seeds_both_fresh_sets":all(speech[s]["0.16"]["paired_effects"][r]["passes_primary_gate"] for s in speech for r in ("EXP-T2-801","EXP-T2-803")),"continuous_sensing_slwm_both_seeds":all(policy[r]["passes_gate"] for r in ("EXP-T2-801","EXP-T2-803")),"field_actor_beats_direct_sensor_actor_both_slwm_seeds":all(policy[r]["field_representation_gate"] for r in ("EXP-T2-801","EXP-T2-803"))}; write_json(root/"gates.json",gates)
    lines=["# T2-B1 — broad speech and continuous multimodal fields","","Randomly initialized shared audio/video/phonetic-text models versus paired Transformers, with matched data, window plans, optimizers and2000-update limits. Selection uses independent validation only; final speakers are disjoint from all prior SLWM exposures. Gates and every failed case remain visible.","",f"Fitted codec/duration values: {fitted:,}; external frozen Vocos:13,531,650 neural weights, separately counted. CMU pronunciation is a fixed external dictionary, not a learned semantic text encoder. The124/125M English language models have separate weights.","","| Run | Variant / seed | Neural weights | Selected / total updates | Validation loss | Measured run minutes |","|---|---|---:|---:|---:|---:|"]
    for r in cfg["runs"]:
        d=selection[r["id"]]; lines.append(f"| {r['id']} | {r['variant']} / {r['seed']} | {d['parameters']:,} | {d['selected_step']} / {d['updates']} | {d['selected_validation_loss']:.4f} | {d['wall_seconds']/60:.2f} |")
    lines += ["","## Fresh-speaker true-gap speech","","Primary160ms gap, all48 clips/person-disjoint12-speaker set; actual waveform erasure before FFT. Mean clip WER is primary, corpus WER is also saved. Each baseline is scored identically by pinned external Whisper-tiny.en. Observed PCM is exact outside every gap. CIs resample people10,000 times.","","| Set | Condition | Mean clip WER | Change versus stronger registered baseline |95% interval | Gate |","|---|---|---:|---:|---|---|"]
    for split,v in speech.items():
        primary=v["0.16"]
        for name,wer in primary["mean_clip_wer"].items():
            effect=primary["paired_effects"].get(name)
            if effect:
                lo,hi=effect["person_cluster_95_interval"]; lines.append(f"| {split} | {name} | {100*wer:.2f}% | {100*effect['relative_reduction']:+.2f}% relative | [{100*lo:+.2f},{100*hi:+.2f}]pp | {'PASS' if effect['passes_primary_gate'] else 'FAIL'} |")
            else: lines.append(f"| {split} | {name} | {100*wer:.2f}% | — | — | control |")
    lines += ["","## Same-model video contribution on new AMI people","","These secondary40.96s cases compare identical weights/input gaps with audio only, correct video and wrong/static/±240ms video. Empty-reference and quiet cases remain explicit; this is not a fitted timing-rejection gate. All values are mean clip WER.","","| Run | Audio only | Aligned video | Static video | Wrong time | −240ms | +240ms |","|---|---:|---:|---:|---:|---:|---:|"]
    for name,v in av.items(): lines.append("| "+name+" | "+" | ".join(f"{100*v['mean_clip_wer'][c]:.2f}%" for c in ("audio_only","aligned_video","static_video","wrong_time_video","minus240ms","plus240ms"))+" |")
    lines += ["","## Shared representation and future prediction","","Blind audio/video/text embeddings are encoded independently, with within-person candidates and no appearance input. Phonetic accuracy excludes silence/unknown targets. Forecasts receive both observed sensors and predict beyond the2-packet guard; persistence uses the same last observations.","","| Run | Audio/video retrieval | Audio/text retrieval | Video/text retrieval | Future audio MAE / persistence | Future video MSE / persistence |","|---|---:|---:|---:|---:|---:|"]
    for name,v in summaries.items():
        a,b=v["future_mean_audio_mae_video_mse"]; c,d=v["persistence_mean_audio_mae_video_mse"]; lines.append(f"| {name} | {100*v['retrieval']['audio_to_video']['person_mean']:.1f}% | {100*v['retrieval']['audio_to_text']['person_mean']:.1f}% | {100*v['retrieval']['video_to_text']['person_mean']:.1f}% | {a:.4f} / {c:.4f} | {b:.4f} / {d:.4f} |")
    lines += ["","Chance varies with eligible windows and is saved per person. Training-mean forecasts and paired person intervals are also in representation_summary.json. Alignment/field readouts do not establish language understanding. The smaller256-width arms are a single-seed exploratory size comparison; this is not enough to infer a scaling law. The trained reset-memory arm also changes embedding pooling to an average of normalized chunk readouts, so its retrieval difference is not a pure memory effect; the same-weight long reset control isolates runtime memory removal more directly.","","## Beyond thirty seconds","","Every reserved AMI participant receives the same fixed80s start and40.96s interval. All main architecture/seed outputs and registered wrong-time, reset and prefix controls are in watch.html. Text prompts use training-estimated timing with no reference alignment. All audio is retained, including poor synthesis; external recognizer text is kept separate from actual model CTC text. field_content_summary.json reports actual CTC text from the model; long_condition_summary.json includes every control, source/oracle waveform recognition and no-reference insertion counts.","","| Run | Audio→video code MSE / provided static | Motion MSE / static motion | Video→audio WER |","|---|---:|---:|---:|"]
    for name,v in long_summary.items(): lines.append(f"| {name} | {v['audio_to_video_code_mse']:.4f} / {v['provided_static_code_mse']:.4f} | {v['audio_to_video_motion_mse']:.5f} / {v['static_motion_mse']:.5f} | {100*v['video_to_audio_mean_clip_wer']:.1f}% |")
    lines += ["","Long spontaneous meetings contain silent participants and crosstalk. All fixed cases remain, and normalized reference-word/edit counts distinguish no-reference intervals from ordinary WER. No general-scene visual decoder or pretrained language model is used in these fields. Native video is64×64 and appearance is supplied. The text edge represents CMU phones, not arbitrary English semantics.","","## Continuous decisions and research scope","",f"Broad speech gate: **{'PASS' if gates['speech_slwm_both_seeds_both_fresh_sets'] else 'FAIL'}**. Continuous SLWM sensing gate: **{'PASS' if gates['continuous_sensing_slwm_both_seeds'] else 'FAIL'}**. Field actor versus direct-sensor actor gate: **{'PASS' if gates['field_actor_beats_direct_sensor_actor_both_slwm_seeds'] else 'FAIL'}**. Full paired cost/coverage results are in the P1-S1 report. Each tick runs automatically from persistent memory, with frozen world weights; sensing actions change observations only.","","Both SLWM and Transformer share the same recurrent field/memory design. A shared-latent result is not automatically specific to the extra SLWM signal paths. Compare paired metrics, ablations, actual compute and both seeds before attributing a gain. Useful robot decisions additionally require action-conditioned physical dynamics, goals/reward, multi-step consequence tests and a faster control loop. The1.365s received chunk plus analysis support is unsuitable evidence for fast motor control. Waveform rendering additionally uses512ms of future predicted mel context and is performed after rollout; file-based stateful inference is not an end-to-end live microphone/camera transport."]
    (root/"report.md").write_text("\n".join(lines)+"\n"); gallery(root)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout="constrained")
    for r in cfg["runs"][:4]:
        c=json.loads((root/r["id"]/"curves.json").read_text()); axes[0].plot([x["step"] for x in c],[x["validation"]["selection_loss"] for x in c],label=r["variant"]+str(r["seed"]))
    axes[0].set(xlabel="Optimizer updates",ylabel="Held-person selection loss",title="Learning with fresh-person validation"); axes[0].legend(fontsize=8)
    x=np.arange(2); width=.18
    for j,name in enumerate(("silence","interpolated_mel","EXP-T2-801","EXP-T2-802")): axes[1].bar(x+(j-1.5)*width,[100*speech[s]["0.16"]["mean_clip_wer"][name] for s in speech],width,label=name)
    axes[1].set(xticks=x,xticklabels=list(speech),ylabel="Mean clip WER (%)",title="True160ms gaps; external recognizer"); axes[1].legend(fontsize=7)
    for i,name in enumerate(("EXP-T2-801","EXP-T2-802","EXP-T2-803","EXP-T2-804")):
        v=policy[name]; axes[2].bar(i-.15,v["mean_total_cost"]["learned"],.3,color="#6366f1",label="Learned" if i==0 else None); axes[2].bar(i+.15,v["mean_total_cost"][v["strongest_fixed"]],.3,color="#14b8a6",label="Best fixed" if i==0 else None)
    axes[2].set(xticks=range(4),xticklabels=["SLWM91","Tr91","SLWM92","Tr92"],ylabel="Future error + sensor cost",title="Continuous sensing; all ticks count"); axes[2].legend()
    fig.savefig(root/"results.png",dpi=170); plt.close(fig); print(json.dumps({"broad_report_complete":True,"gates":gates}),flush=True)


if __name__=="__main__": main()
