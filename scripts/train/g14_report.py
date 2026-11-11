"""Render the verified G14 public tables and README from measured, scene-paired results."""

import argparse
import json
from pathlib import Path

BASE = Path("experiments/multimodal/t14/spoken-closed-loop-20261111")


def main(repo):
    folder = repo / BASE
    load = lambda file: json.loads((folder / file).read_bytes())
    physical = load("metrics/physical-results.json")
    r = load("metrics/release.json")
    lock = load("final-control-lock.json")
    groups = physical["cohorts"]
    mission = load("metrics/missions.json")
    speed = load("metrics/throughput.json")
    routes = load("metrics/routing-audit.json")
    stage = lock["stage"]
    guard = lock["contact_only"]
    stand = physical["preregistered_positive_seeds"]
    val = lambda x: 100 * x["value"]

    def percent(x):
        return f"{val(x):.1f}%"

    def points(x):
        return f"{val(x):+.1f} [{100*x['lo']:+.1f}, {100*x['hi']:+.1f}]"

    def rate(x):
        return f"{val(x):.1f}% [{100*x['lo']:.1f}, {100*x['hi']:.1f}]"

    def table(row):
        return "| " + " | ".join(row) + " |"

    def val_for(role, model, metric="success"):
        return groups[role]["by_policy_family"][model][metric]

    def contrast(role, name):
        return groups[role]["paired"][name]["success"]

    def models(label, model):
        return table(
            [
                label,
                *[rate(val_for(k, model)) for k in ("held", "fresh")],
                *[
                    f"{val_for(k,model,'final_mm')['value']:.1f}"
                    for k in ("held", "fresh")
                ],
            ]
        )

    def diff(label, name):
        return table([label, *[points(contrast(k, name)) for k in ("held", "fresh")]])

    stagename = {
        "initial": "the original one-step data",
        "expert-isolated": "the geometry-visited continuation data",
        "visited-isolated": "the learner-visited continuation data",
    }[stage]
    policy = (
        "restricting learned corrections to contact"
        if guard
        else "allowing learned corrections throughout approach and contact"
    )
    passed = "met" if physical["shared_processor_superiority"] else "did not meet"
    report = [
        "# G14 · Spoken goals, measured counterfactuals and sustained control",
        "",
        f"Four recordings from the same speaker instructed four different goals from each of **320 independently seeded tabletop starts**. Every registered policy acted from a private copy of the same simulator process. Each episode executed 72 native physics/control ticks. Across the two cohorts, the paired spatial-minus-dense selective placement difference was **{points(contrast('held','transport_over_gpt2'))}** and **{points(contrast('fresh','transport_over_gpt2'))} percentage points**. The matched processor comparison {passed} its registered superiority criterion. [Frozen panel](final-control-lock.json) · [All paired results](metrics/physical-results.json)",
        "",
        "## What actually reached the goal?",
        "",
        "Selective placement requires the requested block to stay within 25 mm of its requested station for three consecutive observations, with all three other blocks remaining within 35 mm of their starts through that arrival. Terminal distance uses the full 7.2-second episode. Intervals resample complete paired worlds 4,000 times within speaker and contact condition. The published two-speaker intervals condition on those people. The matched neural rows average three fixed training seeds.",
        "",
        "| Policy | 256 worlds, 8 speakers | 64 worlds, 2 fresh speakers | Terminal mm, 8 speakers | Terminal mm, 2 speakers |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, k in (
        ("Explicit missing goal", "none"),
        ("Direct push", "straight"),
        ("Geometric feedback", "geometry"),
        ("Fast push at contact", "contact_speed"),
        ("Raw physical predictor", "raw"),
        ("Dense GPT-2 style", "gpt2"),
        ("Spatial SLWM", "transport"),
        ("Gated core, original recipe and one seed", "gated"),
        ("Dense spatial attention, original recipe and one seed", "dense_spatial"),
        ("Geometry with the transcript", "geometry_oracle_cue"),
        ("Geometry with transcript and simulator poses", "geometry_oracle_pose_cue"),
    ):
        report.append(models(label, k))
    report += [
        "",
        "![Whole-world paired intervals, matched processors and causal interventions](physical-comparison.png)",
        "",
        f"The held panel uses {stagename} and {policy}. The parameter-matched neural processors each have {next(iter(r['physical_model_parameters'].values())):,} trainable parameters, two blocks, width 80, a 128-coordinate emitted field and eight common candidate actions. Both share an 88.15M-parameter frozen audio encoder and a 182,535-parameter speech head. Dense uses causal multihead softmax attention. The spatial processor uses position-weighted, decaying positive attention. These runs address the small physical processor. [Training curves and data hashes](metrics/physical-training.json) · [Speech and transfer](speech-and-transfer.md)",
        "",
        "## Which inputs and computations changed physical success?",
        "",
        "Positive values indicate higher selective-placement success for the first condition. Each cell is a paired percentage-point difference and a 95% interval.",
        "",
        "| Intervention | 8 held speakers | 2 fresh speakers |",
        "|---|---:|---:|",
    ]
    for lab, name in (
        ("Spatial minus dense, three seeds", "transport_over_gpt2"),
        ("Spatial minus geometric feedback", "transport_over_geometry"),
        ("Spatial minus raw learned prior", "transport_over_raw"),
        ("Spatial minus simple fast contact", "transport_over_contact_speed"),
        (
            "Correct spoken object minus substituted object",
            "transport_over_wrong_color",
        ),
        (
            "Correct spoken address minus substituted address",
            "transport_over_wrong_address",
        ),
        (
            "Original learned field minus training mean, spatial seed 1411",
            "transport_full_over_mean_field",
        ),
        (
            "Original learned field minus training mean, dense seed 1411",
            "gpt2_full_over_mean_field",
        ),
        (
            "Retained goal minus reset after one action, spatial seed 1411",
            "retained_over_reset",
        ),
        (
            "Pretrained speech minus original acoustic features with geometry",
            "new_over_original_speech",
        ),
        (
            "Confidence threshold minus open confidence gate with geometry",
            "confidence_gate_over_open",
        ),
    ):
        report.append(diff(lab, name))
    report += [
        "",
        f"The spatial advantage had a positive paired lower bound for {stand['held']}/3 seeds on the eight-speaker cohort and {stand['fresh']}/3 seeds on the two new speakers. The protocol requires at least two in each cohort. The field intervention preserves the raw proposal and direct candidate features while replacing the 126 learned coordinates. The six exported models give identical fields for spoken and written versions of the same decoded goal on the published prefix. The explicit goal interface enforces that equivalence. The intervention above tests whether the contextual field adds physical value beyond that interface. [Field intervention](field-intervention.md) · [Portable cross-modal checks](metrics/field-replay.json)",
        "",
        "## How the model was trained and selected",
        "",
        f"Original training used 4,698 states from 320 worlds, with eight physically measured candidate effects per state. Each continuation arm added 2,048 states from the same 128 additional worlds, goals and clocks. Eight candidate controls per state were executed for 15 native ticks, including the common continuation. All 12 definitive continuation fits used 1,024 updates of 128 samples. The 6,746 available states correspond to 19.4 sampled observations per unique state per fit when averaged over the pool. Training was balanced by world. The value-development bank contained 512 states on 32 other worlds. Physical policy selection used another 32 starts. The earlier exploratory continuations remain archived. [Exact return data](isolated-training-lock.json) · [Training design](continuation.md)",
        "",
        "The snapshot-and-restore diagnostic recovered identical reported poses and joints but could produce up to 72.24 mm difference during a repeated trajectory. Giving each physical branch its own copy of the unadvanced solver reproduced all poses and camera images in eight full-duration checks. All 960 isolated candidate comparisons also reproduced exactly when the enumeration order was reversed. Later independent replay reproduced every native position digest and every selective-placement score across 48 held episodes. The original one-step labels had 0.048 mm mean difference under the audit. The longer labels were recollected after failing the registered tolerance. [Solver and training audit](solver-isolation.md) · [Held trace audit](metrics/trajectory-replay.json)",
        "",
        f"The original routing choice was checked again under isolation. The paired difference for direct minus obstacle-aware routing was {points(routes['isolated_direct_over_routed'])} percentage points across the registered three development policies. The common route was locked before this replication.",
        "",
        "## Ongoing operations and real records",
        "",
        f"Six original human instructions arrived during each of 16 continuous missions lasting {min(mission['durations']):.1f}–{max(mission['durations']):.1f} seconds. The controller continued executing its previous goal while the next recording played. Recorded speech posteriors were cached and released only after the complete waveform had elapsed on the simulation clock. An independent CPU/CUDA waveform replay checks the speech edge. The same table, contacts and actuator history persisted throughout each mission. Only the explicit memory controls reset or ignore the goal. Goal retention here is implemented by the controller. The learned core receives four recent frames. [Mission protocol](missions.md) · [Measured mission outcomes](metrics/missions.json) · [Watch and replay](examples/README.md)",
        "",
        "| Mission policy | Selective placement | Arrival | Final target error |",
        "|---|---:|---:|---:|",
    ]
    for name, label in (
        ("geometry-carry", "Geometry, retained goal"),
        ("dense-carry", "Dense, retained goal"),
        ("spatial-carry", "Spatial, retained goal"),
        ("spatial-reset", "Spatial, reset after one action"),
        ("spatial-stale", "Spatial, keep first accepted goal"),
        ("spatial-clear-rejected", "Spatial, clear on rejection"),
    ):
        x = mission["performance"][name]
        report.append(
            table(
                [
                    label,
                    rate(x["success"]),
                    percent(x["arrival"]),
                    f"{x['final_mm']['value']:.1f} mm",
                ]
            )
        )
    report += [
        "",
        f"On already satisfied goals the spatial controller emitted zero movement on {100*mission['already_at_goal_noop_fraction']['spatial']:.1f}% of the tested decision clocks. The explicit absence control did so on {100*mission['already_at_goal_noop_fraction']['absent']:.1f}%. Across 768 acoustic input/map challenges the fixed speech gate made zero unsupported commits, including actual recordings naming an unsupported white block. It also rejected 41 valid, correctly decoded commands on the two new speakers. [Acoustic abstention and distinct waveform counts](abstention.md)",
        "",
        "Two reuse cohorts of real robot recordings separately measured a four-step, 0.8-second actuator forecast. Four genuine training parents reduced simulation-initialized error from 34.09 to 5.55 mm and from 30.18 to 5.14 mm. Real-only fitting with the same four parents reached the same rounded errors. The counterfactual placement scores above come from the official simulator. The prerecorded real cohorts measure effector response under observed controls. [Real calibration curves](speech-and-transfer.md) · [Original Language-Table simulator](https://github.com/google-research/language-table)",
        "",
        "## Compute, scope and next experiment",
        "",
        f"The RTX 3090 completed {r['fits']} neural fits with {r['generations']} optimizer/weight generations verified on two independent disks. Two v6e allocation attempts yielded a timeout and service unavailability. This study used {r['owned_credits']:.2f} Colab units. Cumulative tracked use is {r['total_owned_credits']:.2f} of the authorized 600. At batch one, warm accelerator processing and eight outcome heads took {speed['cores']['gpt2']['cuda']['1']['milliseconds']:.2f} ms for dense and {speed['cores']['transport']['cuda']['1']['milliseconds']:.2f} ms for spatial. At batch 512 they processed {speed['cores']['gpt2']['cuda']['512']['scenes_per_second']:,.0f} and {speed['cores']['transport']['cuda']['512']['scenes_per_second']:,.0f} encoded scenes per second. Camera rendering, speech encoding and simulation are outside these timings. [Throughput](metrics/throughput.json) · [Checkpoint custody](metrics/custody.json) · [Compute accounting](metrics/compute.json)",
        "",
        "The physical processor receives camera-based geometry, robot telemetry, action history and an explicitly decoded color–address goal. Success here concerns four tabletop objects, a constrained speech grammar and a supplied station map. The newly acquired speakers were excluded from all SLWM fitting. Their recordings’ membership in Whisper pretraining is unknown. Transfer to new hardware, unconstrained language, new tasks and a learned representation that can generate interchangeable modalities all require separate evidence.",
        "",
        r["next"],
        "",
        "[Runnable models](inference/README.md) · [Underlying per-scene results for eight speakers](metrics/held-scene-results.npz) · [Per-scene results for two new speakers](metrics/fresh-scene-results.npz) · [Release receipt](metrics/release.json)",
    ]
    (folder / "report.md").write_text("\n".join(report) + "\n")
    examples = [
        "# Sustained spoken tabletop control",
        "",
        "Both examples run the locked spatial model at seed 1411. They were selected by fixed mission index before their outcomes were observed. Each starts from an independently seeded scene. Six genuine human utterances arrive during continuous contact simulation. The view is sampled once per 300 ms macro, the physics clock runs at 10 Hz, and the file carries the original audio on its simulation clock. The controller receives each cached decoder result only after that recording finishes. The overlaid named stations are part of the public map.",
        "",
    ]
    for name, v in r["movies"].items():
        examples.append(
            f"- [{name} · {v['seconds']:.1f} seconds]({name}.mp4) · [Native positions, applied controls and retained goals]({name}.npz)"
        )
    examples += [
        "",
        "[Every mission and cue outcome](mission-outcomes.json) · [48 full native held trajectories](native-traces.npz) · [Independent replay audit](../metrics/trajectory-replay.json)",
        "",
        "The human recordings are from Martin Cooke, Jon Barker, Stuart Cunningham and Xu Shao's *The Grid Audio-Visual Speech Corpus*, CC BY 4.0. Audio was aligned to the recording clock and placed at cue onset. The control scene is generated in Google's open Language-Table PyBullet environment. [Original speech corpus](https://doi.org/10.5281/zenodo.3625687) · [Speech license](https://creativecommons.org/licenses/by/4.0/) · [Simulation source](https://github.com/google-research/language-table)",
    ]
    (folder / "examples/README.md").write_text("\n".join(examples) + "\n")
    instructions = folder / "inference/README.md"
    text = (
        instructions.read_text().split("\n## Execute a physical decision")[0].rstrip()
    )
    text += """\n\n## Execute a physical decision

Run the 128-coordinate model on the published four-observation prefix. The original spoken example, two other recordings from the same speaker, their feature arrays, and all six physical model checkpoints are included. Supply a different correctly shaped scene archive to use your own chronological camera and state features. The speech edge was trained on short six-word GRID commands.

```sh
PYTHONPATH=. python scripts/infer/g14_live.py --core transport --inspect
PYTHONPATH=. python scripts/infer/g14_live.py --core gpt2 --seed 1412
PYTHONPATH=. python scripts/infer/g14_live.py --text "bin blue at i six please"
PYTHONPATH=. python scripts/infer/g14_live.py --absent
PYTHONPATH=. python scripts/infer/g14_live.py --features experiments/multimodal/t14/spoken-closed-loop-20261111/inference/other-address-features.npz
```

The ordinary output gives the accepted goal, chosen candidate, no-op or movement route and signed two-dimensional macro offset in meters. With `--inspect`, the learned physical model also exposes its 128 coordinates, its predicted effector and object displacement, and the fitted probability of a large one-step utility prediction error. One macro spans three 10 Hz native ticks. [Replay all published goal paths](../metrics/field-replay.json) · [Continuous simulator and audio examples](../examples/README.md)
"""
    instructions.write_text(text)
    readme = repo / "README.md"
    body = readme.read_text()
    i = body.index("### G14 ")
    j = body.index("### G13 ", i)
    lead = f"### G14 spoken goals and continuous tabletop control\n\nThe frozen policies executed {r['total_held_branches']:,} branches across {r['total_held_worlds']} held starts, with four real spoken alternatives per physical scene. On two new speakers, spatial SLWM achieved {percent(val_for('fresh','transport'))} selective placement, dense GPT-2 style {percent(val_for('fresh','gpt2'))}, and geometric feedback {percent(val_for('fresh','geometry'))}. The spatial processor {passed} the registered three-seed superiority criterion. Pretrained speech recovered all three fields on 96.3% of fresh-speaker utterances. G14 includes {r['fits']} fits, six portable physical models, two original-audio continuous replays and few-shot calibration on real robot recordings. [Full physical and multimodal results]({BASE}/report.md) · [Run the models]({BASE}/inference/README.md) · [Watch continuous missions]({BASE}/examples/README.md)\n\n"
    readme.write_text(body[:i] + lead + body[j:])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("repo", type=Path)
    main(p.parse_args().repo)
