"""Compose the scientific release after all registered weights and held scores are verified."""

import argparse
import json
from pathlib import Path

FOLDER = Path("experiments/multimodal/t10/context-alignment-20261111")
SEEDS = (1001, 1002, 1003)
LABEL = {"hybrid": "SLWM", "gpt2": "Dense GPT-2", "gated": "Gated recurrent"}


def fmt(fn, scale=100, digits=1):
    return " / ".join(f"{fn(s)*scale:.{digits}f}" for s in SEEDS)


def build(repo):
    d = repo / FOLDER
    m = d / "metrics"
    load = lambda f: json.loads((m / f).read_text())
    ctx = load("context.json")
    real = load("grounding.json")
    cmd = load("instructions.json")
    card = load("data.json")
    release = load("release.json")
    fore = load("forecasts.json")
    row = lambda c, t, s: real["rows"][f"align-{c}-{t}-s{s}"]

    def av(c, t, role):
        return fmt(
            lambda s: row(c, t, s)["panels"][role]["retrieval"]["audio_to_vision"][
                "direct"
            ]
        )

    def occ(c, reg):
        return fmt(
            lambda s: ctx["rows"][f"context-{c}-{reg}-s{s}/5120"]["full"]["occupancy"]
        )

    def gen(c, t, name):
        def value(seed):
            x = row(c, t, seed)["panels"]["novel"]["conditional"][name]
            return 1 - x["aligned"] / x["prior"]

        return fmt(value)

    lines = []
    for c in ("hybrid", "gpt2"):
        for t in ("reconstruct", "shifted", "paired", "rooms"):
            lines.append(
                f"| {LABEL[c]} | {t} | {av(c,t,'familiar')} | {av(c,t,'novel')} | {av(c,t,'sony_extra')} |"
            )
    tr = card["holo"]["train"]
    risk0 = row("hybrid", "paired", 1001)["risk"]
    rp = []
    for role in ("familiar", "novel"):
        for c in ("hybrid", "gpt2"):
            ap = fmt(
                lambda s: row(c, "paired", s)["risk"][role]["metrics"]["all"]["ap"][
                    "mean"
                ]
            )
            auc = fmt(
                lambda s: row(c, "paired", s)["risk"][role]["metrics"]["all"]["auroc"][
                    "mean"
                ],
                1,
                3,
            )
            rp.append(
                f"| {role} | {LABEL[c]} | {ap} | {auc} | {risk0[role]['metrics']['raw']['ap']['mean']*100:.1f} | {risk0[role]['prevalence']*100:.1f} |"
            )
    gate = {
        c: {
            role: sum(v[role] for v in real["gates"][c].values())
            for role in ("familiar", "novel", "sony_extra", "sony_test")
        }
        for c in ("hybrid", "gpt2")
    }
    supported = max(gate[c]["novel"] for c in gate) >= 2
    grounded = (
        "The prespecified novel-task alignment criterion was met."
        if supported
        else "Neither architecture met the prespecified novel-task alignment criterion."
    )
    example = fore["rows"]["align-hybrid-paired-s1003"]["novel"]
    quality = release["drive_at_release"]
    total_hours = sum(r["hours"] for r in card["holo"].values())
    text = f"""# G10 · shared state under longer histories and new physical tasks

Matched variable-context training changes the conclusion from G9. At equal training exposure, SLWM led dense GPT-2 in **one of three** seeds on uninterrupted held control. The simpler gated recurrent reference handled two actuator changes better than the hybrid in every seed. On public human task recordings, synchronous audio–video supervision yielded {av('hybrid','paired','familiar')}% audio-to-video accuracy for SLWM on familiar task families, and {av('hybrid','paired','novel')}% on held coffee/computer tasks. Chance was 12.5%. {grounded}

![Context training, command retention, real retrieval, and risk](g10-comparisons.png)

Points represent three independent training seeds. The dotted retrieval reference is chance among eight time windows in the same recording. All controller policies issue an action and receive its consequences for 512 consecutive ticks.

## What originally looked like an architecture win?

Within each training seed, every architecture saw the same 41,943,040 clock positions, 61,440 independently initialized simulated worlds, public room data, source masks, commands and action targets. Each core had 13,245,696 parameters and emitted 128 coordinates. The primary cohort contained 256 fresh 51.2-second missions with unpredictable goal intervals and held motor stalls. Time at the requested goal was {occ('gpt2','short')}% for dense GPT-2 trained in 32-step segments, {occ('gpt2','variable')}% with variable 32–512 histories, and {occ('gpt2','long')}% with only 512-step histories. The corresponding variable-context outcomes for SLWM and the gated reference were {occ('hybrid','variable')}% and {occ('gated','variable')}%. All intervals are paired within training seed. [Full factorial, actuator changes and calibration-delay controls](context.md)

Variable training changes how often the model begins a new context, how far gradients traverse and how many available past observations compete for attention. The experiment identifies the effect of the segmentation schedule as a whole. The dense control used the common periodic clock inherited from G9. A separately tuned Transformer with a distinct long-range position scheme is a further comparison.

## Does sustained control mean remembering what the user said?

Showing a goal for 0.4 seconds instead of continuously reduced hybrid occupancy by **{fmt(lambda s:-cmd['parent'][f'context-hybrid-variable-s{s}']['brief_minus_continuous']['mean'])} points**. The dense loss was **{fmt(lambda s:-cmd['parent'][f'context-gpt2-variable-s{s}']['brief_minus_continuous']['mean'])} points**. Twelve matched post-training fits did not repair this consistently. Holding the last actually received word record until superseded reproduced every action of continuous delivery across nine frozen models and 256 further missions each. This is an explicit, inspectable goal memory for the simulator's structured instructions. [Causal cache, fresh-cohort tuning and limits](commands.md)

## Do real signals meet in the same coordinates?

We used up to ten minutes per session, totaling {total_hours:.2f} hours from **416 public HoloAssist sessions** with video, pitch-shifted microphone audio, synchronized accelerometer and gyroscope readings, completed spoken utterances and human correctness labels. The optimization split contained {tr['recordings']} recordings, {tr['hours']:.2f} hours and {tr['transcribed_words']:,} transcribed words. Development had 32 separate recordings. The 64 familiar-task tests were drawn from official validation. The 64 novel-task recordings belonged to coffee-machine or computer tasks absent from optimization and development. Grouping avoids repeated source sessions. People may recur across recordings. [Original public dataset](https://holoassist.github.io/) · [Split, hashes and source exposure](metrics/data.json)

Each post-training update rehearsed 4,096 simulated clock positions and drew 2,048 real positions. In the mixed treatment half the real candidate groups came from STARSS23 and half from HoloAssist. Audio-only and camera-only views of a synchronously recorded window are additional views of the same information. Only {fmt(lambda s:card['matched_posttraining_exposure'][str(s)]['four_holo_sources_and_present_utterance_and_fresh_imu'])}% of sampled HoloAssist clock positions combined a four-source mask, a recently completed utterance and fresh IMU samples. The compressed-video release omitted the audio track. We verified the audio tracks in the separate original AV members before training.

The fixed-temperature contrastive loss reads the full emitted field. Eight candidate windows belong to the same recording, share the same relative clock, and start in disjoint intervals. The shifted treatment substitutes a nonzero random cyclic permutation of the targets. The reconstruction treatment has zero contrastive weight. The room-only control retains genuine pairing and the same optimizer exposure while sourcing all of its real groups from STARSS23. All sensor decoders continue to read the same world field.

| Core | Post-training | Familiar HoloAssist (%) | Held task families (%) | Sony rooms 21/22 (%) |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(lines)}

In familiar tasks, the strict lower-bound test against reconstruction, wrong pairing and chance passed **{gate['hybrid']['familiar']}/3** SLWM and **{gate['gpt2']['familiar']}/3** dense seeds. On novel tasks these counts were **{gate['hybrid']['novel']}/3** and **{gate['gpt2']['novel']}/3**. A raw cross-sensor activity trace reached 13.1% on familiar tasks and 12.5% on novel tasks. All twelve ordered source combinations, self-retrieval, recording-paired intervals and both Sony panels are included. [Grounding scores](metrics/grounding.json) · [Paired recording measurements](metrics/grounding-paired.npz)

The paired objective preserved simulated control. Relative to their variable-context parents, the SLWM continuations changed occupancy by {fmt(lambda s:real["paired_contrasts"][f"hybrid-s{s}"]["parent_control"]["mean"])} percentage points and the dense continuations by {fmt(lambda s:real["paired_contrasts"][f"gpt2-s{s}"]["parent_control"]["mean"])}. All six were inside the five-point forgetting threshold on the downside. This separates learning a useful association within a task family from transferring it to a new one.

For the paired treatment on held task families, RGB prediction skill from audio alone was {gen('hybrid','paired','audio/vision')}% for SLWM and {gen('gpt2','paired','audio/vision')}% for dense GPT-2. Audio prediction skill from vision alone was {gen('hybrid','paired','vision/audio')}% and {gen('gpt2','paired','vision/audio')}%. Skill is one minus the ratio of mean recording loss to the training-feature prior loss. The training mean had lower error for both directions in every reported seed. The generated 60-second examples preserve the model's native 16×16 output and render the learned spectral envelope. [Watch and listen](examples/README.md)

## Do frozen features anticipate a human error?

Fixed-penalty logistic readouts received only measured prefixes ending at least 200 ms before an annotated human action began. The training cap was 96 examples per recording, selected without using outcomes. The unseen task panel contained **{risk0['novel']['events']:,} actions, {risk0['novel']['mistakes']} errors**. Future action identity, post-hoc narration and correctness were excluded from the input. Average precision directly reflects performance on rare errors.

| Recording panel | Paired model | Average precision (%) | AUROC | Raw-sensor AP (%) | Prevalence (%) |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(rp)}

The recording-bootstrap interval for improvement over raw-sensor average precision included zero in all six novel-task comparisons. The human actions were observed offline. The release includes recording-bootstrap intervals, comparisons to raw sensors and task prevalence, and readouts with words removed or words alone. [Per-event scores](metrics/risk-predictions.npz) · [Readout and paired uncertainty](metrics/grounding.json)

A separate passive one-tick check compared the trained sensor heads with simply repeating the current measurement. For the seed-1003 paired hybrid on novel tasks, the next accelerometer RMSE was {example['mean']['all']['acceleration']:.3f} in released sensor units with all sources, versus {example['mean']['persistence']['acceleration']:.3f} for persistence. Repeating the current accelerometer reading beat all twelve tested paired and reconstruction parents at this 100 ms horizon. [All twelve forecast controls](metrics/forecasts.json)

## Release and next experiment

All **{release['total_fits']} backbone fits** completed. We verified {release['verified_generations']} inference and optimizer generations on both Fedora HDDs. We replayed public packages through the ordinary ongoing-control and conditional-signal interfaces. The RTX 3090 supplied local training and evaluation. G10 used **{release['owned_units']:.2f} tracked Colab units** on named v5e-1 assignments. Cumulative project usage was {release['cumulative_owned_units']:.2f} of 100. Google timed out on v6e. An earlier assignment disappeared before training-bank delivery. The two replacement lanes restored the exact parent hashes and the same verified data. All study-owned assignments were released. A shared Drive project quota delayed the private cloud mirror. It had {quality.get('verified_files',0)} verified G10 files at release. [Release ledger](metrics/release.json) · [Reproduction steps](reproduction.md) · [Three complete runnable models](inference/README.md)

For live controllers, preserve the active instruction explicitly and train the dense and gated baselines on the deployed history distribution. For a broader shared representation, test whether a spoken instruction predicts the *next labeled manipulation and its measured physical change* across held task families. Match examples with wrong instructions from the same situation and compare with raw sensors, persistence and the same dense backbone. A coarse microphone/camera synchrony score alone cannot establish that causal relation.
"""
    (d / "report.md").write_text(text)
    path = repo / "README.md"
    base = path.read_text()
    i = base.index("### G10 context training and public task recordings")
    j = base.index("### G9 concurrent sensing", i)
    section = f"""### G10 context training and public task recordings

On matched variable histories, SLWM led dense GPT-2 on uninterrupted simulated control in one of three seeds. A simpler gated recurrent control reached about 94% occupancy in all seeds and handled two actuator changes best. With 416 public HoloAssist recordings, paired SLWM audio-to-video retrieval reached {av('hybrid','paired','familiar')}% on familiar tasks and {av('hybrid','paired','novel')}% on unseen coffee/computer tasks, against 12.5% chance. Brief goal instructions exposed a separate memory failure that a causal last-command cache repaired in the simulator. [Verified G10 study](experiments/multimodal/t10/context-alignment-20261111/report.md) · [Runnable models](experiments/multimodal/t10/context-alignment-20261111/inference/README.md) · [60-second examples](experiments/multimodal/t10/context-alignment-20261111/examples/README.md)

"""
    path.write_text(base[:i] + section + base[j:])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("repo", type=Path)
    build(p.parse_args().repo)
