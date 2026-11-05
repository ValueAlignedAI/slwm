"""G2 gates, reproducible registries and the raw-versus-core decision."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from training.t2_real_av import write_json,stable_hash,cluster_interval
from training.t2_prepare_cremad import sha256


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text()); root=Path(cfg["artifacts"]); read=lambda p:json.loads(p.read_text())
    metrics={s:{r["id"]:read(root/r["id"]/(s+"_metrics.json")) for r in cfg["runs"]} for s in ("test","external")}
    complete={r["id"]:read(root/r["id"]/"training_complete.json") for r in cfg["runs"]}
    asr=read(root/"exploration/intelligibility.json"); results={}; comparisons={}; gates={}
    for split in metrics:
        results[split]={}; comparisons[split]={}
        for run in cfg["runs"]:
            key=run["id"]; x=metrics[split][key]; m=x["metrics"]; gain=1-m["learned_mel_mse"]/m["interpolation_mel_mse"]
            score=asr["summary"][split]; wer_gain=score["paired_changes"][key]["relative_wer_gain_vs_interpolation"]
            results[split][key]={"mel_mse":m["learned_mel_mse"],"temporal_mse":m["learned_temporal_mse"],"gain_vs_interpolation":gain,
                "mean_clip_wer":score["mean_clip_wer"][key],"wer_gain_vs_interpolation":wer_gain,
                "mean_per_clip_gain_person_cluster_95_interval":x["person_cluster_gain_95_interval"]}
        for field,raw in (("EXP-T2-422","EXP-T2-421"),("EXP-T2-424","EXP-T2-423")):
            a,b=metrics[split][field]["per_clip"],metrics[split][raw]["per_clip"]
            if [r["sample_id"] for r in a]!=[r["sample_id"] for r in b]: raise ValueError("Unpaired core comparison")
            gain=np.array([(rr["learned_mel_mse"]-ff["learned_mel_mse"])/rr["learned_mel_mse"] for ff,rr in zip(a,b)])
            comparisons[split][field+"_vs_"+raw]={"mean_per_clip_gain":float(gain.mean()),"person_cluster_95_interval":cluster_interval(gain,[r["actor"] for r in a])}
    for arm,ids in (("raw_residual",("EXP-T2-421","EXP-T2-423")),("frozen_core_residual",("EXP-T2-422","EXP-T2-424"))):
        gates[arm]=all(results["test"][i]["gain_vs_interpolation"]>=.05 and results["test"][i]["wer_gain_vs_interpolation"]>=.05
                       and results["external"][i]["gain_vs_interpolation"]>=-.02 for i in ids)
    core_gate=all(comparisons["test"][k]["mean_per_clip_gain"]>=.05 and comparisons["test"][k]["person_cluster_95_interval"][0]>0
                  and comparisons["external"][k]["mean_per_clip_gain"]>0 for k in comparisons["test"])
    summary={"sprint":"T2-G2","config_hash":stable_hash(cfg),"results":results,"paired_core_comparisons":comparisons,"gates":gates,"core_benefit_gate":bool(core_gate),
        "total_training_wall_seconds":sum(c["wall_seconds"] for c in complete.values()),"recognizer_diagnostic":asr["summary"],
        "decision":"Retain raw observation skip with small residual decoder; frozen core adds no reliable reconstruction value. Keep core for timing alignment only.",
        "external_status":"reused diagnostic, architecture informed by G1 failure; never used for fitting or checkpoint selection"}
    write_json(root/"summary.json",summary)
    lines=["# T2-G2 — preserve observed sound at the edge", "",
        "**Decision: retain the raw observation skip and small residual decoder. The frozen aligned core does not improve this reconstruction task.** This is a useful signal-preserving edge component, not evidence of general generation or world dynamics.", "",
        "G1 showed new-domain degradation when a normalized shared field replaced observed audio detail. G2 separately preregisters a fixed 25→100 Hz physical-grid skip, plus a learned correction bounded to ±0.25 normalized mel units and initialized at zero. Four 1,600-step runs compare raw versus frozen-core corrections in seeds 41 and 42, with identical 548,352-weight decoders, data, crops and training budget. Frozen-core arms additionally cost 4,741,089 weights. No external generator supplies sound.", "",
        "The same CREMA 5,061/210/240 split, 3.636 training hours, batch 32, 32-packet windows and 51,200 sampled windows per arm are retained. Validation selects checkpoints every 200 steps. All 64 RAVDESS clips are reused diagnostic data because the architecture responds to G1's external failure; no external clips fit parameters or normalization. The 48-case recognizer subset is unchanged, with only four external people and repeated prompts. Confidence intervals resample people, 2,000 draws.", "",
        "Only decoder/data/update budgets are matched: cached frozen fields include overlap-averaged context from the full observed clip, potentially beyond a decoder crop; raw corrections see the crop directly. Core arms also pay additional extraction compute and parameters. No claim of matched total compute or identical temporal receptive field is made.", "",
        f"Total decoder-training wall time: {summary['total_training_wall_seconds']:.1f} seconds on RTX 3090; preparation/scoring/rendering excluded. Raw skip has zero neural weights; codec/statistical values are counted separately (249,376 fitted values overall, 128 derived affine buffer values introduce no new fitting).", ""]
    for split in ("test","external"):
        score=asr["summary"][split]; control=metrics[split]["EXP-T2-421"]["metrics"]
        lines += [f"## {split}", "", "| Condition | 100 Hz mel MSE | Gain vs interpolation | Temporal MSE | Mean clip WER |", "|---|---:|---:|---:|---:|",
            f"| Deterministic interpolation | {control['interpolation_mel_mse']:.6f} | — | {control['interpolation_temporal_mse']:.6f} | {100*score['mean_clip_wer']['interpolation']:.1f}% |"]
        for run in cfg["runs"]:
            key=run["id"]; v=results[split][key]
            lines.append(f"| {key}: {'raw' if run['input']=='audio_coarse' else 'frozen core'} + skip, seed {run['seed']} | {v['mel_mse']:.6f} | {100*v['gain_vs_interpolation']:+.1f}% | {v['temporal_mse']:.6f} | {100*v['mean_clip_wer']:.1f}% |")
        lines += ["",f"Original / true 100 Hz codec / old audio head rendered at 100 Hz WER: {100*score['mean_clip_wer']['reference']:.1f}% / {100*score['mean_clip_wer']['codec100']:.1f}% / {100*score['mean_clip_wer']['old_audio_100renderer']:.1f}%. Matching the renderer does not by itself fix the old head.", "",
            "| Comparison | Mean per-clip mel gain | Person-cluster 95% interval |", "|---|---:|---:|"]
        for key,v in comparisons[split].items():
            ci=v["person_cluster_95_interval"]; lines.append(f"| {key} | {100*v['mean_per_clip_gain']:+.2f}% | [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]% |")
        lines += ["", "All listening results are evaluator measurements, not SLWM transcription. They use mean per-clip WER, can exceed 100%, and remain a fallible selected diagnostic. In particular, external WER improvement intervals include zero for these four people; the means are not strong confirmation of intelligibility generalization. Mel improvements use all clips, not just the listening subset.", ""]
    lines += ["## Frozen gates and accounting", "",
        f"Raw-residual pilot gate: **{'PASS' if gates['raw_residual'] else 'FAIL'}**. Frozen-core-residual pilot gate: **{'PASS' if gates['frozen_core_residual'] else 'FAIL'}**. Core-benefit gate: **{'PASS' if core_gate else 'FAIL'}**. The preregistered gate requires both seeds to improve test mel error ≥5% and recognizer WER ≥5%, with no >2% relative external error regression. A core claim additionally requires ≥5% benefit over matched raw correction and a positive cluster interval. This is an exploratory repair, not a fresh external confirmation.", "",
        "| ID | New trainable weights | Frozen core | Selected step | Train seconds | Checkpoint SHA256 |", "|---|---:|---:|---:|---:|---|"]
    for run in cfg["runs"]:
        key=run["id"]; c=complete[key]
        if sha256(root/key/"best.pt")!=c["checkpoint_sha256"]: raise ValueError("G2 checkpoint changed")
        registry={"experiment_id":key,"sprint":"T2-G2","config_hash":stable_hash(cfg),**c,"gates":gates,"status":"completed"}
        write_json(root/key/"registry.json",registry); p=c["provenance"]
        lines.append(f"| {key} | {p['decoder_trainable']:,} | {p['frozen_core']:,} | {c['selected_step']} | {c['wall_seconds']:.2f} | `{c['checkpoint_sha256']}` |")
    lines += ["", "## Retained artifacts and next step", "",
        "Every checkpoint includes optimizer/config/RNG state. Source snapshots, configuration and source hashes, complete per-clip mel/temporal metrics, wrong-correction controls and all actual WAV files are retained. G1 artifacts remain unchanged. The standalone `exploration.generation_infer` accepts G1 or G2 checkpoints and real audio/video inputs; G2 only reconstructs observed audio.", "",
        "The next core experiment should test retaining phonetic/spectral information under varied speakers, channels and corpora while preserving alignment, with explicit raw bypass and frozen-core probes. Do not spend more parameters on this same narrow recipe, infer identity from audio, or treat the sharp copied seed as learned motion. Video→speech and audio-driven visual dynamics remain failed gates."]
    (root/"report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"report":str(root/"report.md"),"gates":gates,"core_benefit_gate":bool(core_gate)}),flush=True)


if __name__=="__main__": main()
