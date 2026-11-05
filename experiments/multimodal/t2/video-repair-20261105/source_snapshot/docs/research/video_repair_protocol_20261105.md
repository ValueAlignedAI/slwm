# T2-V1 — does video improve speech repair beyond sound alone?

The user explicitly requests the 3090, architecture improvement and useful trained
artifacts for this question. Earlier model/source/data artifacts remain intact.
This isolated experiment measures conditional repair, not text generation,
autonomous dynamics, or a policy-commitment claim.

## Hypotheses fixed before training

Prior sum fusion could damage sound. Freeze the content-preserving audio core;
first train an audio-only residual anchor on the new gap task. Then freeze that
entire expert and train a separate visual correction applied ONLY in true gaps.
Audio-only capacity controls add the same correction architecture with an
approximately parameter-matched audio frontend. Thus correct video must beat
both the frozen anchor and an additional trained audio-only branch.

Use 96×64 mouth pixels, clip contrast normalization and frame differences,
a small four-stage CNN, temporal processing and local cross-attention (±2
packets) from audio to visual features. Compare a PCA-video frontend, ungated
fusion, shuffled-pair training and no negative training. Confidence uses only
observed audio and video. Wrong/mistimed-video training penalizes residuals and
trains alignment confidence; no evaluation target may determine the gate.
Source inspiration: https://arxiv.org/abs/1702.01992 and
https://www.isca-archive.org/interspeech_2023/montesinos23_interspeech.html.
No external pretrained visual/audio generator supplies output.

## True-gap support and preservation

Primary gap: one contiguous 320ms span (8×40ms packets) within a 1.28s context.
Secondary 160/480ms gaps are declared in advance. Training samples all three.
Audio features are 64ms centered STFT windows sampled at 25Hz. Hide EVERY input
feature whose sample support intersects a true gap, including the first packet
after it. Recompute deltas after observed-only interpolation. This removes the
boundary-sample leakage allowed by previous feature-packet erasure. A test must
change original waveform samples inside gaps and prove input invariance.

Rendered speech uses constrained iterative phase reconstruction and preserves
the actually observed waveform outside gaps exactly. That passthrough is real
conditioning, explicitly distinct from generated gap samples. Hidden original
samples are erased before conditioning or phase reconstruction. No clean hidden
waveform sample, reference phase, ASR output or test label enters inference.
Spectral corrections cover windows intersecting the gap (including their STFT
halo); the final waveform correction is confined to the missing sample interval.
Compare silence, interpolation, audio expert, audio-capacity control and video
repair with identical observed samples and decoder settings. No-gap input is
returned unchanged. Mel losses remain signal proxies, separate from speech WER.

## Data and holdouts

Retain CREMA train 5,061 / validation 210 / reused test 240, disjoint people.
Decode the original fixed crop at 96×64, not enlarged old 48×32 thumbnails.
New paired training: RAVDESS actors 07,08,19,20, all 60 full-AV speech clips each
(240), both repetitions/intensities, no overlap with any final test person.
New validation: the earlier R1 RAVDESS actors 03,04,05,06,15,16,17,18 (128 clips).
They are now explicitly validation data, not a fresh test in this cycle.
New reserved test: actors 09,10,11,12,21,22,23,24, normal intensity/repetition 01,
both statements/all emotions (128). Same corpus/two phrases, new people, not
general novel-vocabulary confirmation. The old RAVDESS 01,02,13,14 remain unused.

Use the previously fixed crop geometry, no model-score-based framing changes.
Keep all audio/PCA codec statistics fixed from CREMA training. Raw individual
SHA256 and ZIP-member CRC checked; declared whole-ZIP MD5 retained. Official
source https://zenodo.org/records/1188976, Livingstone & Russo (2018),
CC BY-NC-SA 4.0, noncommercial research. No new normalization or test fitting.

## Training and accounting

Two anchors (seeds 61/62), 1,600 updates each, start from corresponding frozen
R2 models 521/522 and train only a new audio residual branch. Phase B adds a
new branch on the frozen anchor: nine runs, 2,400 updates (619: 4,800), batch32,
24 CREMA +8 RAV training windows. Same windows/gaps/channel draws within a seed.
Gain ±12dB and four-knot EQ ±6dB affect both target and input. LR .0004,
80-step warmup/cosine decay; evaluation every400 updates. The extra audio-only
run tests twice the update exposure; it is not labeled data-budget matched.

Every candidate selects minimum equal-corpus mean validation hole-MSE at320ms,
using only CREMA validation and the explicitly repurposed RAV validation people.
The 611/619 audio baseline for seed61 is chosen by validation alone; seed62 uses
612. Freeze all choices before final test scores. Record exact active/frozen
parameter counts, head sizes, historical pretraining and wall time. The raw-CNN
and audio frontends are approximately matched, not claimed identical FLOPs.
Additional visual negatives mean extra compute, declared separately.

## Frozen gates and controls

Primary evaluated windows have ≥6 packets on either side for valid real shifted
video; use length≥50 clips, middle32 packets, centered320ms gap for all models.
Report exclusions and secondary all-clip results when shift controls are absent.
Paired person-cluster bootstrap intervals use2,000 draws.

1. Both primary CNN seeds614/615 improve mean per-clip hole-MSE ≥5% over the
   frozen anchor AND validation-selected additional audio-only control, with
   positive95% cluster intervals on reused CREMA and fresh RAV test people.
2. Correct video beats same-person DIFFERENT-utterance video, ±240ms shifted
   video and a repeated static frame by≥2%, with positive intervals. Also report
   gap-only frame shuffle, mouth occlusion and no-video. Wrong sensitivity alone
   does not prove usefulness. Shuffling is a training-only intervention.
3. Observation preservation is exact at waveform samples; no-video correction is
   exactly zero by construction. A practical speech claim additionally requires
   independent recognizer WER improvement ≥5% relative to the same audio baseline
   and silence, with a positive paired-person interval. Otherwise report the
   model as a feature predictor only. No architecture superiority follows solely
   from different-sized baseline comparisons.

Listening subset: two evenly spaced eligible CREMA sample IDs per person (16),
the two neutral statements per new RAV actor (16). Fixed before scores. Render
one true320ms gap per complete clip, one globally erased waveform before any
window, all conditions share phase/observed constraints. Save original, corrupted,
interpolated, audio, video, wrong/shifted/static controls plus predictions and
exact masks. The pinned Whisper-tiny.en evaluator is fallible and evaluation-only;
its decoded text is never SLWM output. Save all models including failures. If
video fails this gate, retain the useful audio artifact and report the negative.
