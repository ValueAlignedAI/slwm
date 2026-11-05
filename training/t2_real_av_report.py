"""Summarize every frozen arm, including negative outcomes, into research artifacts."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from training.t2_real_av import cluster_interval, write_json


def report(config_path: str) -> Path:
    cfg = json.loads(Path(config_path).read_text())
    root = Path(cfg["artifact_dir"])
    runs = [json.loads((root/r["experiment_id"]/"metrics.json").read_text()) for r in cfg["runs"]]
    card = json.loads((Path(cfg["corpus_dir"])/"dataset_card.json").read_text())
    # These two completed source cards counted the unused ImageNet classifier
    # in the downloaded checkpoint. Preserve their immutable hashes and record
    # deployed feature-encoder accounting as an explicit erratum.
    checkpoint_params = card["feature_spec"].get("frozen_checkpoint_parameters",card["feature_spec"]["frozen_encoder_parameters"])
    encoder_params = checkpoint_params - (512*1000+1000)
    write_json(root/"parameter_accounting.json",{"model_parameter_counts_by_experiment":{r["experiment_id"]:r["parameter_counts"] for r in runs},
               "frozen_downloaded_checkpoint_parameters":checkpoint_params,"frozen_feature_encoder_parameters_used":encoder_params,
               "unused_classification_head_parameters":513000,
               "note":"Original source cards/registries counted the complete ImageNet checkpoint. The classifier is discarded; immutable source cards are retained with this correction."})
    comparisons = []
    for full in (r for r in runs if r["arm"] == "full"):
        baseline = next(r for r in runs if r["arm"] == "vanilla" and r["seed"] == full["seed"])
        values = {}
        for name in ("audio","visual"):
            values[name+"_relative_error_improvement"] = 1-full["test"]["errors"][name]/baseline["test"]["errors"][name]
            lookup = {r["sample_id"]:r for r in baseline["test"]["per_sample"]}
            sample_rows = full["test"]["per_sample"]
            improvements = np.array([lookup[r["sample_id"]][name]-r[name] for r in sample_rows])/baseline["test"]["errors"][name]
            values[name+"_person_cluster_95_interval"] = cluster_interval(improvements,[r["actor"] for r in sample_rows])
        def mean_r1(row):
            return .5*sum(row["test"]["retrieval"][d]["r1"] for d in ("audio_to_video","video_to_audio"))
        values["mean_retrieval_r1_absolute_improvement"] = mean_r1(full)-mean_r1(baseline)
        values["architecture_pass"] = (max(values[n+"_relative_error_improvement"] for n in ("audio","visual")) >= cfg["gates"]["architecture_improvement"]
                                        or values["mean_retrieval_r1_absolute_improvement"] >= cfg["gates"]["retrieval_absolute_improvement"])
        comparisons.append({"seed":full["seed"],"full":full["experiment_id"],"baseline":baseline["experiment_id"],**values})
    fulls = [r for r in runs if r["arm"] == "full"]
    prediction = all(any(v["prediction_pass"] for v in r["gates"].values()) for r in fulls)
    contribution = all(any(v["cross_modal_pass"] for v in r["gates"].values()) for r in fulls)
    architecture = all(c["architecture_pass"] for c in comparisons)
    decision = {"prediction_pass_both_seeds":prediction,"cross_modal_pass_both_seeds":contribution,
                "architecture_pass_both_seeds":architecture,"hybrid_scale_gate_pass":prediction and contribution and architecture,
                "comparisons":comparisons,"planned_runs_completed":len(runs),
                "train_seconds_total":sum(r["training_seconds"] for r in runs)}
    residual = cfg["model"].get("prediction_parameterization") == "residual_persistence"
    if cfg["gates"].get("require_visual_dynamics",False):
        dynamics = all(1-r["test"]["errors"]["visual"]/r["test"]["errors"]["visual_persistence"] >= cfg["gates"]["prediction_improvement"] for r in fulls)
        decision["visual_dynamics_pass_both_seeds"] = dynamics
        decision["hybrid_scale_gate_pass"] = decision["hybrid_scale_gate_pass"] and dynamics
    write_json(root/"decision.json",decision)
    fields = ["experiment_id","arm","seed","parameters","audio_mse","visual_mse","a2v_r1","v2a_r1","within_person_auc","training_seconds","best_step"]
    table = []
    for r in runs:
        t = r["test"]
        table.append({"experiment_id":r["experiment_id"],"arm":r["arm"],"seed":r["seed"],"parameters":r["parameter_counts"]["total"],
                      "audio_mse":t["errors"]["audio"],"visual_mse":t["errors"]["visual"],
                      "a2v_r1":t["retrieval"]["audio_to_video"]["r1"],"v2a_r1":t["retrieval"]["video_to_audio"]["r1"],
                      "within_person_auc":t["retrieval"]["within_person_auc"],"training_seconds":r["training_seconds"],"best_step":r["best_step"]})
    with (root/"comparison.csv").open("w") as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    lines = ["# T2 real audiovisual " + ("residual follow-up" if residual else "pilot") + " — 2026-11-04", "",
             "## Decision", "",
             ("The hybrid passes this narrow pilot gate; replication on a second domain is the next allowed scale check."
              if decision["hybrid_scale_gate_pass"] else
              "Do not scale the hybrid on this evidence. At least one preregistered gate failed; keep the checkpoints and benchmark as negative or limited evidence."),
             "",f"Prediction gate (both seeds): **{prediction}**. Cross-modal contribution gate: **{contribution}**. Architecture gate: **{architecture}**.","",
             "## Question and provenance", "",
             "Does a small shared signal processor predict real audio/video futures and benefit from the correctly paired other modality beyond simple baselines?",
             "",f"Protocol: `{cfg['protocol']}`. Cycle: `{cfg['cycle']}`. Base commit: `{cfg['source_commit']}`; exact changed-source hashes accompany every checkpoint.",
             "",f"Dataset: [CREMA-D]({card['source_url']}), snapshot `{card['source_revision']}`. ODbL 1.0 database / DbCL 1.0 contents. "
             f"{card['valid']} recorded clips; split counts `{card['split_counts']}`; person counts `{ {s:len(v) for s,v in card['actor_splits'].items()} }`. "
             f"Failures: {len(card['failures'])}. Audio/video hours per split: `{card['duration_hours']}`.",
             "", "All trainable adapters, embeddings, processor and heads are counted below. The same frozen ImageNet ResNet18 encoder "
             f"({encoder_params:,} deployed parameters) supplies frame features to all arms; PCA and normalization are fitted only on training people. "
             "Its external pretraining is not evidence for learning perception from scratch.",
             "",f"Accounting correction: original source cards/registries count the complete downloaded ImageNet checkpoint ({checkpoint_params:,} parameters), "
             "including its discarded 513,000-parameter classifier. `parameter_accounting.json` records the deployed encoder count; original hashed cards remain intact.",
             "", "## Frozen evaluation", "",
             "Observe the first 16 of 32 normalized-time steps; predict only the remaining 16. Audio FFT windows cannot cross that boundary. "
             "Last-observation persistence and the training mean are measured explicitly. Validation alone selects checkpoint step. "
             "Retrieval uses independent encodes of each stream and every test clip as a candidate. Within-person shuffled controls "
             "preserve identity while changing the paired event. Confidence intervals resample whole people (2,000 bootstrap samples). "
             "Test outcomes never change the frozen matrix or budgets.", "", "## All planned results", "",
             "| Run | Arm | Seed | Trainable params | Audio MSE ↓ | Video MSE ↓ | A→V R@1 | V→A R@1 | Within-person AUC | Seconds |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in table:
        lines.append(f"| {r['experiment_id']} | {r['arm']} | {r['seed']} | {r['parameters']:,} | {r['audio_mse']:.4f} | {r['visual_mse']:.4f} | {r['a2v_r1']:.1%} | {r['v2a_r1']:.1%} | {r['within_person_auc']:.3f} | {r['training_seconds']:.1f} |")
    lines += ["", "Full-split retrieval chance: " + f"{runs[0]['test']['retrieval']['chance_r1']:.2%} across {runs[0]['test']['retrieval']['candidate_count']} candidates.",
              "", "## Strong simple controls", ""]
    if residual:
        lines += ["Residual heads start exactly at last-observation persistence. The train/validation arrays and PCA/normalizer "
                  "are byte-identical to the discovery cycle; the seven test people were absent from every discovery split. "
                  f"The additional visual-dynamics gate (≥5% over persistence in both seeds) is **{decision.get('visual_dynamics_pass_both_seeds')}**.", ""]
    for name in ("audio","visual"):
        e = runs[0]["test"]["errors"]
        lines.append(f"- {name}: training-mean MSE {e[name+'_zero']:.4f}; last-observation persistence MSE {e[name+'_persistence']:.4f}.")
    lines += ["", "## Paired modality contribution and uncertainty", "",
              "Positive relative improvement means the actual paired other stream predicts better than a within-person wrong stream.", "",
              "| Full run | Target | Improvement | Person-cluster 95% interval | Gate |",
              "|---|---|---:|---|---|"]
    for r in fulls:
        for name,c in r["test"]["cross_modal_contribution"].items():
            lo,hi = c["person_cluster_95_interval"]
            lines.append(f"| {r['experiment_id']} | {name} | {c['relative_improvement']:.2%} | [{lo:.2%}, {hi:.2%}] | {r['gates'][name]['cross_modal_pass']} |")
    lines += ["", "## Full model versus vanilla, matched seed", ""]
    for c in comparisons:
        lines.append(f"- Seed {c['seed']}: audio error improvement {c['audio_relative_error_improvement']:.2%}; video {c['visual_relative_error_improvement']:.2%}; mean independent R@1 difference {c['mean_retrieval_r1_absolute_improvement']:+.2%}; architecture gate {c['architecture_pass']}.")
    lines += ["", "## Concrete failure cases", "",
              "The table lists the largest video-error excess over persistence for each full-model seed. "
              "Every other clip's errors remain in `metrics.json`; these examples are selected to expose failures.", "",
              "| Run | Held-out clip | Person | Model video MSE | Persistence MSE |",
              "|---|---|---|---:|---:|"]
    for r in fulls:
        worst = max(r["test"]["per_sample"],key=lambda s:s["visual"]-s["visual_persistence"])
        lines.append(f"| {r['experiment_id']} | {worst['sample_id']} | {worst['actor']} | {worst['visual']:.4f} | {worst['visual_persistence']:.4f} |")
    lines += ["", "`forecast_examples.png` shows the first full-model seed's lowest, median and highest joint-error test clips. "
              "The audio panels distinguish observed context from predicted future; video curves compare feature-error against persistence. "
              "These diagnostic feature plots use the saved checkpoint in FP32; aggregate tables use the registered BF16 evaluation."]
    lines += ["", "## Compute and artifacts", "",
              f"RTX 3090, BF16 autocast with FP32 weights/optimizer, AdamW, {cfg['training']['steps']} steps × {cfg['training']['batch_size']} clips per run. "
              f"Total measured train-loop time {decision['train_seconds_total']/60:.1f} minutes; final diagnostics and preprocessing are additional. "
              "Data and optimizer-step budgets are matched; wall time is reported rather than claimed equal. "
              "Temporal mixer ablations remove small numbers of parameters; exact differences are in the table.", "",
              "Each run directory contains `best.pt`, `config.json`, `registry.json`, `curves.json`, `metrics.json`; "
              "the best checkpoint includes optimizer and RNG state, validation-selected step, and source/config/dataset hashes. "
              "`comparison.csv` and `decision.json` are machine-readable. `learning_curves.png` and `controls.png` show the aggregate behavior.", "",
              "Inference: `python -m training.t2_real_av_infer --checkpoint <run>/best.pt --clip <source.flv> --output <prediction.npz>`. "
              "The outputs are uncommitted internal feature futures; the model has no trained raw-media or language decoder.", "",
              "## Failures and limits", "",
              "The corpus contains acted speech, repeated sentence prompts, fixed backgrounds and only seven held-out people. "
              "Cross-modal retrieval can exploit demographic or clip-duration cues even when people are disjoint. "
              "Within-person controls address some identity shortcuts; they do not establish general grounding. "
              "Frozen image features may discard subtle lip motion, and normalized time does not test physical timing. "
              "A failed hybrid gate rejects scaling this recipe, not every possible latent predictive model. "
              "No evidence here supports text/code capability, general world understanding, hallucination reduction, policy selection, or raw-media generation.", "",
              "## Next allowed step", "",
              ("Replicate the full and vanilla models on a second real audiovisual domain at this small scale before raising parameter count."
               if decision["hybrid_scale_gate_pass"] else
               "Inspect per-person and temporal errors, retain the stronger simple baseline, and register a separate targeted experiment before any joint text/code run or larger model."), ""]
    if not residual:
        lines += ["## Recorded execution limitation", "",
                  "The discovery single-modality arms used equal sample budgets but different sampled clip orders because "
                  "their dropout path consumed fewer RNG draws. The full, vanilla, spectral, long-convolution and shuffled "
                  "comparisons used the same sample/dropout schedule. The residual follow-up fixes this schedule for all arms.", ""]
    simple_path = root/"posthoc_simple_controls/metrics.json"
    if simple_path.exists():
        simple = json.loads(simple_path.read_text())
        lines += ["## Exploratory stronger controls", "",
                  "Added after discovery and some follow-up outputs were seen, these controls do not change the frozen gate or supply new confirmation. "
                  "All fits use training clips only; ridge regularization is fixed at 100, without test tuning. "
                  "The time-profile control predicts the training mean at each normalized time, rather than the global zero mean. "
                  "Linear models predict future residuals from flattened observed features and retain both streams' own persistence anchors. "
                  "The audio/visual labels identify residual-regressor inputs; these are not absent-modality tests.", "",
                  "| Control | Audio test MSE | Video test MSE |", "|---|---:|---:|"]
        for name,value in simple["controls"].items():
            lines.append(f"| {name} | {value['test']['audio_mse']:.4f} | {value['test']['visual_mse']:.4f} |")
        lines += ["", "Fitted linear predictors and all control settings are saved under `posthoc_simple_controls/`. "
                  "Their results narrow the value of gains against a vanilla neural baseline; they do not validate multimodal grounding.", ""]
    (root/"report.md").write_text("\n".join(lines))
    plot(root,runs,cfg)
    return root/"report.md"


def plot(root: Path, runs: list[dict], cfg: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig, axes = plt.subplots(1,2,figsize=(13,4.5),layout="constrained")
    for r in runs:
        curve = json.loads((root/r["experiment_id"]/"curves.json").read_text())["curve"]
        points = [p for p in curve if "validation" in p]
        for ax,name in zip(axes,("audio","visual")):
            ax.plot([p["step"] for p in points],[p["validation"]["errors"][name] for p in points],label=f"{r['arm']} · {r['seed']}",alpha=.8)
            ax.set(title=f"Held-out-person {name} future prediction",xlabel="Optimizer steps",ylabel="Validation feature MSE ↓")
    axes[1].legend(fontsize=8,ncol=2)
    fig.suptitle("Real audiovisual pilot · validation chooses checkpoints")
    fig.savefig(root/"learning_curves.png",dpi=170)
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    labels = [f"{r['arm']}\nseed {r['seed']}" for r in runs]
    for ax,name in zip(axes,("audio","visual")):
        ax.bar(np.arange(len(runs)),[r["test"]["errors"][name] for r in runs],color=["#315c9b" if r["arm"]=="full" else "#8ca5ae" for r in runs])
        e = runs[0]["test"]["errors"]
        ax.axhline(e[name+"_zero"],color="#b35c29",ls="--",label="Training mean")
        ax.axhline(e[name+"_persistence"],color="#27816a",ls=":",label="Last observed feature")
        ax.set(xticks=np.arange(len(runs)),xticklabels=labels,title=f"Test {name} futures",ylabel="Feature MSE ↓")
        ax.tick_params(axis="x",labelrotation=60,labelsize=8)
        ax.legend(fontsize=8)
    fig.suptitle("All frozen arms · simple controls included")
    fig.savefig(root/"controls.png",dpi=170)
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    simple_path = root/"posthoc_simple_controls/metrics.json"
    simple = json.loads(simple_path.read_text()) if simple_path.exists() else None
    for ax,name in zip(axes,("audio","visual")):
        error = runs[0]["test"]["errors"]
        full = [r["test"]["errors"][name] for r in runs if r["arm"]=="full"]
        vanilla = [r["test"]["errors"][name] for r in runs if r["arm"]=="vanilla"]
        values = [error[name+"_zero"],error[name+"_persistence"],np.mean(vanilla)]
        labels = ["Train\nmean","Repeat last\nobservation","Transformer\n2 seeds"]
        colors = ["#aeb8ba","#7d9789","#8ca5ae"]
        if simple:
            values.append(simple["controls"]["linear_audio_only"]["test"][name+"_mse"])
            labels.append("Linear audio\ncontrol*")
            colors.append("#b89863")
        values.append(np.mean(full)); labels.append("SLWM\n2 seeds"); colors.append("#315c9b")
        bars = ax.bar(labels,values,color=colors)
        for bar,value in zip(bars,values):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f"{value:.3f}",ha="center",va="bottom",fontsize=9)
        for x,series in ((2,vanilla),(len(values)-1,full)):
            ax.plot([x,x],[min(series),max(series)],color="#172a37",lw=2)
            ax.scatter([x]*len(series),series,color="#172a37",s=12,zorder=3)
        ax.set(title=f"{name.capitalize()} future features",ylabel="Test feature MSE · lower is better",ylim=(0,max(values)*1.18))
        ax.tick_params(axis="x",labelsize=9)
    fig.suptitle("Small SLWM on real audio/video · " + ("residual forecast on new test people" if cfg["model"].get("prediction_parameterization")=="residual_persistence" else "absolute forecast"))
    if simple:
        fig.supxlabel("* Exploratory linear residual control; both streams retain their own persistence anchors. Dots show the two neural training seeds.",fontsize=8)
    fig.savefig(root/"forecast_comparison.png",dpi=180)
    plt.close(fig)
    plot_examples(root,runs,cfg)


def plot_examples(root: Path, runs: list[dict], cfg: dict) -> None:
    import matplotlib.pyplot as plt
    import torch
    from training.t2_real_av import batch_from_arrays,model_config
    from training.t2_train_latents import T2AudioVisualSLWM
    row = next(r for r in runs if r["arm"] == "full")
    samples = row["test"]["per_sample"]
    order = sorted(range(len(samples)),key=lambda i:samples[i]["audio"]+samples[i]["visual"])
    picks = [order[0],order[len(order)//2],order[-1]]
    with np.load(Path(cfg["corpus_dir"])/"test.npz",allow_pickle=False) as archive:
        audio,visual = archive["audio"][picks],archive["visual"][picks]
    checkpoint = torch.load(row["checkpoint"],map_location="cpu",weights_only=False)
    model = T2AudioVisualSLWM(model_cfg=model_config(cfg,"full"),audio_feature_dim=audio.shape[-1],visual_feature_dim=visual.shape[-1]).eval()
    model.load_state_dict(checkpoint["model"])
    context = cfg["training"]["context_steps"]
    with torch.inference_mode():
        prediction = model(batch_from_arrays(torch.from_numpy(audio),torch.from_numpy(visual),context))
    ap = prediction["audio_prediction"].numpy()
    vp = prediction["visual_prediction"].numpy()
    fig,axes = plt.subplots(3,3,figsize=(12,8),layout="constrained")
    for i,(label,index) in enumerate(zip(("Lowest joint error","Median joint error","Highest joint error"),picks)):
        actual = audio[i]
        combined = np.concatenate([actual[:context],ap[i,context:]],axis=0)
        limit = max(2.,float(np.quantile(np.abs(actual),.98)))
        for ax,values,title in ((axes[i,0],actual,"Actual audio features"),(axes[i,1],combined,"Observed + predicted audio")):
            ax.imshow(values.T,aspect="auto",origin="lower",cmap="coolwarm",vmin=-limit,vmax=limit)
            ax.axvline(context-.5,color="black",lw=1,ls="--")
            ax.set(title=title if i==0 else "",xlabel="Normalized time step",ylabel="Mel feature bin")
        model_error = ((vp[i,context:]-visual[i,context:])**2).mean(-1)
        repeat_error = ((visual[i,context-1]-visual[i,context:])**2).mean(-1)
        axes[i,2].plot(np.arange(context,len(actual)),model_error,label="Model")
        axes[i,2].plot(np.arange(context,len(actual)),repeat_error,label="Repeat last",ls="--")
        axes[i,2].set(title="Video feature future error" if i==0 else "",xlabel="Future time step",ylabel="MSE ↓")
        axes[i,2].legend(fontsize=8)
        axes[i,0].set_title(f"{label} · {samples[index]['sample_id']}\nActual audio features",fontsize=10,pad=7)
    fig.suptitle("Saved-checkpoint diagnostics · observed prefix | predicted future")
    fig.savefig(root/"forecast_examples.png",dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    print(report(parser.parse_args().config))
