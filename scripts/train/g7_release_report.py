"""Write the human-readable continuation and artifact report from the completed matched panels."""
import json
from pathlib import Path


def read(path):return json.loads(path.read_bytes())


def make(folder):
    metrics=folder/"metrics";curves=read(metrics/"scale-curves.json")["models"];stats=read(metrics/"matched-contrasts.json")
    cost=read(metrics/"project-compute.json");inventory=read(metrics/"checkpoint-inventory.json")
    data=read(metrics/"data.json")["complete_exposure"]
    def name(size,core,seed,arm):
        return f"g6-v5e-{size}-{core}-{seed}" if arm=="parent" else f"g7-v5e-{size}-{core}-{seed}-{arm}"
    def seed_values(size,core,arm,source,measure):
        return [curves[name(size,core,s,arm)][-1][source][measure] for s in (511,512)]
    def pair(v,dec=2):return " / ".join(f"{n:.{dec}f}" for n in v)
    lines=["# G7 matched multimodal continuation",
        "",f"Sixteen continuations reached 16,384 updates. The 116.50M and 202.35M hybrids each have an exactly parameter-matched dense GPT-2-style control. Each fresh/replay pair resumes identical weights and optimizer state at 8,192 updates. Both original seeds and all {inventory['verified_rungs']} continuation milestones passed independent readback from two physical checkpoint drives.",
        "",f"The two named Google assignments used {cost['project_compute_units']:.2f} estimated Colab compute units in total ({cost['total_runtime_hours']:.2f} single-chip hours). The 100-unit allowance applies to these assignments. Data preparation, alignment, simulation and held-out evaluation ran on Fedora. [Project-only ledger](metrics/project-compute.json) · [Checkpoint hashes](metrics/checkpoint-inventory.json)",
        "","## Equal exposure on independent simulator starts","","Each number is seed 511 / 512. Lower endpoint error in pixels is better. The final held set has 512 fresh Push-T starts. The raw ridge fits only original training scenes.",
        "","| Parameters | Core | Parent at 8,192 | Replay at 16,384 | Fresh at 16,384 | Raw ridge |","|---|---|---:|---:|---:|---:|"]
    for size in ("anchor","larger"):
        for core in ("hybrid","gpt2"):
            values=[pair(seed_values(size,core,arm,"fresh_test","endpoint_px")) for arm in ("parent","replay","fresh")]
            ridge=seed_values(size,core,"fresh","fresh_test","raw_ridge_px")[0]
            lines.append(f"| {'116.50M' if size=='anchor' else '202.35M'} | {'Hybrid' if core=='hybrid' else 'Dense GPT-2'} | "+" | ".join(values)+f" | {ridge:.2f} |")
    lines+=["","## Event transfer to the held Tokyo site","","Average precision is multiplied by 1,000. Higher values are better. This site has two held rooms. The raw control uses absolute frame motion.",
        "","| Parameters | Core | Parent | Replay | Fresh | Raw motion |","|---|---|---:|---:|---:|---:|"]
    for size in ("anchor","larger"):
        for core in ("hybrid","gpt2"):
            values=[pair([1000*x for x in seed_values(size,core,arm,"fresh_test","event_ap")]) for arm in ("parent","replay","fresh")]
            ref=1000*seed_values(size,core,"fresh","fresh_test","raw_motion_ap")[0]
            lines.append(f"| {'116.50M' if size=='anchor' else '202.35M'} | {'Hybrid' if core=='hybrid' else 'Dense GPT-2'} | "+" | ".join(values)+f" | {ref:.2f} |")
    lines+=["","## Text and cross-modal prediction","","Held text is measured in conditional byte negative log-likelihood on 512 independent FineWeb documents. Lower values are better. Audio-to-text retains the same teacher-forced previous-byte bigram for the correct and sensor-erased controls. Speech and image losses use the same units for all arms.",
        "","| Parameters | Core | FineWeb parent | FineWeb replay | FineWeb fresh | Fresh CREMA video-to-audio MSE |","|---|---|---:|---:|---:|---:|"]
    for size in ("anchor","larger"):
        for core in ("hybrid","gpt2"):
            def signal(arm,cond,metric):
                return pair([curves[name(size,core,s,arm)][-1]["signals_test"][cond][metric] for s in (511,512)],3)
            v=[signal(arm,"text_text","byte_nll") for arm in ("parent","replay","fresh")]+[signal("fresh","crema_video","audio_mse")]
            lines.append(f"| {'116.50M' if size=='anchor' else '202.35M'} | {'Hybrid' if core=='hybrid' else 'Dense GPT-2'} | "+" | ".join(v)+" |")
    lines+=["","## Equal TPU time","","The following comparison holds the continuation update-compute time at the longest duration bracketed by both sizes, including first-block compilation. Each entry is larger minus anchor on the original development Push-T endpoint. Negative favors the larger model. The held tests above use measured checkpoints at equal exposure.",
        "","| Core | Arm | Time in minutes, seed 511 / 512 | Endpoint difference in pixels |","|---|---|---:|---:|"]
    for core in ("hybrid","gpt2"):
        for arm in ("replay","fresh"):
            rec=[stats["equal_tpu_time"][f"size-{core}-{s}-{arm}"]["matched_time"]["update_seconds"] for s in (511,512)]
            lines.append(f"| {'Hybrid' if core=='hybrid' else 'Dense GPT-2'} | {arm} | {pair([v['seconds']/60 for v in rec],1)} | {pair([v['first_minus_second_endpoint_px'] for v in rec])} |")
    lines+=["","The full tables include the size-by-data interaction, whole-scene/room paired intervals, alternative clocks including evaluation wall time, and interpolated step counts. The common-time estimates are linear interpolations between registered checkpoints. Per-arm time excludes off-device archiving. Actual paid time includes data staging and draining. [Paired contrasts](metrics/matched-contrasts.json) · [All curve points and negative controls](metrics/scale-curves.json)",
        "","![Fresh-domain learning curves for both sizes and backbones](scale-curves.png)",
        "","## Shared state, speech and decisions","",
        f"Each continuation saw {data['rows_per_arm']:,} additional examples. The fresh arm changed {data['actual_changed']:,} while preserving modality masks. Audio, video, numeric state and text all contributed. Four of the nine families remained replay anchors. Two replaced families introduced a domain shift as well as fresh examples. [Data and modality exposure](physical-time.md)",
        "","Four-way audiovisual alignment on 14 previously unseen speakers reached 43–44% through the hybrid and 40–43% through dense GPT-2. A matched head on the raw sensors reached 65–69%. Correct units still produced 85% acted-speech and 99% audiobook word error. The signed-time and waveform controls identify losses at different stages. [Physical-time confirmation](physical-time.md) · [Waveform oracle](alignment-and-control.md)",
        "","All four supervised policies retained useful goal history. None of the four RL policies met its predeclared improvement criterion. Two-step exact planning succeeded in 45–48% of episodes across two diagnostic cohorts. The learned two-step planner lost this gain even when supplied raw state and the retained cue. The earlier hybrid camera-substitution effect did not replicate on Reacher. [Transfer and planning](transfer-and-planning.md)",
        "","## Continuous inference and custody","","The same 116.50M seed and first held Finnish recording were designated for both architectures and both data arms. Each input drives 1,024 consecutive ticks (43.69 seconds) with carried state. The public previews contain generated tracks. Video is a 16×16 grayscale prediction, displayed at a larger size. Prompt-conditioned clips expose changes in the model's output. The waveform and contact evaluation establish what those changes convey.",
        "","| Source and generated output | Hybrid, fresh | Dense GPT-2, fresh |","|---|---|---|",
        "| Audio → video | [Watch](examples/hybrid/audio-to-video.webm) | [Watch](examples/gpt2/audio-to-video.webm) |",
        "| Video → audio | [Listen](examples/hybrid/video-to-audio.ogg) | [Listen](examples/gpt2/video-to-audio.ogg) |",
        "| Text → audio and video | [Watch and listen](examples/hybrid/text-to-audio-video.webm) | [Watch and listen](examples/gpt2/text-to-audio-video.webm) |",
        "","Artifacts for every fit are indexed under the Fedora paths `/mnt/slwm-checkpoints-a/SLWM/g7` and `/mnt/slwm-checkpoints-b/SLWM/g7`. Each `g7-v5e-*/step-016384` contains `inference.npz`, `resume.npz`, the configuration and verification receipt. The source data and preprocessing caches are on the separate SSD under `/mnt/slwm-data/SLWM/g7`. [Inference receipts](metrics/continuous-generation.json) · [Input recording source](https://doi.org/10.5281/zenodo.7880637) · [Run protocol](protocol.md)"]
    comparisons=stats["at_registered_endpoints"]
    def outcomes(labels):
        return [comparisons[label]["fresh_test"]["first_minus_second_endpoint_px"] for label in labels]
    larger_replay=outcomes([f"continuation-v5e-larger-hybrid-{seed}-replay" for seed in (511,512)])
    larger_dense=outcomes([f"continuation-v5e-larger-gpt2-{seed}-replay" for seed in (511,512)])
    larger_fresh=outcomes([f"data-v5e-larger-hybrid-{seed}" for seed in (511,512)])
    size_gap=outcomes([f"size-hybrid-{seed}-fresh" for seed in (511,512)])
    data_labels=[f"data-v5e-{size}-{core}-{seed}" for size in ("anchor","larger") for core in ("hybrid","gpt2") for seed in (511,512)]
    wins=sum(comparisons[k]["fresh_test"]["first_minus_second_endpoint_px"]<0 for k in data_labels)
    supported=sum(comparisons[k]["fresh_test"]["endpoint_ci95"][1]<0 for k in data_labels)
    byte_wins=sum(curves[name(sz,co,se,"fresh")][-1]["signals_test"]["text_text"]["byte_nll"]<curves[name(sz,co,se,"replay")][-1]["signals_test"]["text_text"]["byte_nll"]
                  for sz in ("anchor","larger") for co in ("hybrid","gpt2") for se in (511,512))
    action_learns="The 202.35M hybrid still had optimization headroom on action prediction at update 8,192." if all(x<0 for x in larger_replay) else "Continued optimization of the 202.35M hybrid produced a mixed result on held action prediction."
    time_gap=[stats["equal_tpu_time"][f"size-hybrid-{se}-fresh"]["matched_time"]["update_seconds"]["first_minus_second_endpoint_px"] for se in (511,512)]
    paragraphs=["## What the scale intervention changed","",
        action_learns+f" Replaying the original mixture changed held endpoint error by {pair(larger_replay)} pixels across the two seeds. The large dense control changed by {pair(larger_dense)} pixels under the same continuation. Negative changes mean lower error.",
        "",f"At a fixed exposure of 16,384 updates, fresh examples lowered endpoint error in {wins} of eight architecture–size–seed comparisons. {supported} paired 95% scene intervals excluded zero in that direction. For the large hybrid specifically, fresh minus replay was {pair(larger_fresh)} pixels. On independent FineWeb documents, fresh training lowered next-byte NLL in {byte_wins} of eight comparisons.",
        "",f"At equal exposure, the large fresh hybrid minus the smaller fresh hybrid was {pair(size_gap)} pixels on 512 held starts. At common continuation TPU time the corresponding original-development differences were {pair(time_gap)} pixels. These comparisons use different evaluation sets and are reported separately. Two sizes and two seeds bound the strength of any scaling-law claim. [Matched estimates and uncertainty](metrics/matched-contrasts.json)",
        ""]
    at=lines.index("## Equal exposure on independent simulator starts");lines[at:at]=paragraphs
    (folder/"report.md").write_text("\n".join(lines)+"\n")
