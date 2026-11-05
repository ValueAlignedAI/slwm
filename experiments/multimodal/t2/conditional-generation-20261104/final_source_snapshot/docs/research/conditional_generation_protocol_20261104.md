# T2-G1 conditional generation and frozen-world decoding — 2026-11-04

The user authorizes continuing the research autonomously using the available
Fedora RTX 3090. The previous result supports narrow timing alignment, not useful
generation. This cycle tests the edge decoders and conditioning, rather than
assuming more core parameters fix missing information.

## Research basis and falsifiable questions

SadTalker (https://arxiv.org/abs/2211.12194) separates reference appearance from
audio-driven motion. EnCodec (https://arxiv.org/abs/2210.13438) motivates treating
signal compression/rendering quality as a separate bottleneck. We do not use
either system's pretrained generator or claim to reproduce its results.

1. Does four audio subframes per 25 Hz world packet (100 Hz mel targets) improve
   reconstruction/intelligibility relative to the old 25 Hz edge representation?
2. Given the same starting mouth frame, can an audio-conditioned residual/warp
   decoder predict subsequent frames better than persistence and a time-only
   decoder? Does the actual audio help over a wrong clip from the same person?
3. Does the frozen aligned core help a decoder beyond raw signal features under
   matched decoder/data budgets, including on a new dataset?

## Data and claim limits fixed before results

Retain the CREMA-D M1 split (5,061 train / 210 validation / 240 test) and fixed
25 Hz, 48×32 mouth crop. Fit 100 Hz mel means/scales on training people only.
All previous weights and corpora remain immutable. CREMA test people are still
excluded from weight fitting, but have been evaluated in prior cycles: findings
on this reused holdout are exploratory, not untouched confirmation.

Reserve an external RAVDESS falsification set: actors 01,02,13,14; full audiovisual
speech only, normal intensity, repetition 01, both sentences, all eight emotions
(64 clips if available). Public source https://zenodo.org/records/1188976,
CC BY-NC-SA 4.0, Livingstone & Russo (2018). No external clips fit parameters,
normalization, hyperparameters or checkpoint selection. Inspect framing only to
define a fixed mouth ROI, then freeze it before computing model scores. Record
file hashes, failures, ZIP CRC checks and source/license metadata. No claim of
unrestricted-commercial data licensing is made.

## Sequential experiment stages

Prepare immutable multirate targets and frozen world fields from the small
4,741,089-parameter M1 checkpoint EXP-T2-301. Audio fields never see video;
video fields never see audio. Preserve true 25 Hz video timestamps. A 100 Hz
mel packet is four 64-bin frames per video step, with deterministic inverse mel
and estimated Griffin-Lim phase for every model/control waveform.

Audio arms (same decoder, data and step budget): frozen audio field→100 Hz mel,
frozen video field→100 Hz mel, raw audio→100 Hz mel and raw video→100 Hz mel.
Add a fifth diagnostic: observed 100 Hz mel→100 Hz mel autoencoder. It has
strictly richer observations and is not a fair core comparison. It isolates
whether the low-rate input itself limits speech reconstruction.
Compare original audio, both codec rates, old decoder, mean and wrong-input
controls. Original waveforms are never inserted into generated outputs.

Visual arms: frozen audio field + one actual starting frame → warp/residual
prediction; raw audio + starting frame; starting frame + time only; frozen field
+ direct residual-pixel prediction; wrong-audio training; second-seed full arm.
Every head starts at persistence, and all losses exclude the provided seed frame.
An observed seed is real additional conditioning, not an inferred identity or
an audio-only video. No target frame after the seed may enter any encoder.

Use continuous pixel, edge and temporal losses rather than only latent MSE.
All decoder sizes are measured; frozen core, codec statistics and evaluation-only
recognizer are counted separately. Decoder training does not update the aligned
core. Validation selects checkpoints. Record all runs, including failures, with
source/config/data/checkpoint hashes, seed, curves, samples and compute time.

Before training or external scoring: fix 1,600 updates per arm, audio batch 32,
visual batch 16, 32 physical-time packets, AdamW learning rate 0.0004 with 80-step
warmup and cosine decay, checkpoint selection every 200 updates. Audio loss is
mel L1 + 0.5 temporal L1. Visual loss is full-frame L1 + 2 mouth-region L1 +
0.25 spatial-gradient L1 + 0.5 temporal-gradient L1. Use the same random crops
within each task/seed, including unused wrong-input draws for fair controls.
Inspecting the framing of actors 01 and 02 fixes external geometry at
`x=.35w y=.54h width=.30w height=.24h -> 48x32 gray`, fps 25. No scores informed it.
For the independent recognizer, select the first four sample IDs per test person
in each corpus (32 CREMA clips, 16 external clips), before inspecting scores.
Use original, 25/100 Hz codecs, old audio/video heads, all five new audio heads,
and null/wrong-video controls. This small repeated-prompt diagnostic is not a
human intelligibility benchmark. The gallery includes validation low/median/high
visual errors and fixed test/external examples; it never substitutes for all-clip
pixel/mel metrics. Visual scores concern 32-frame middle windows, not arbitrary
length generation from a seed; audio rendering uses overlap-averaged windows.

## Frozen evaluation gates

Visual: at least 5% lower mouth-region MSE than seed persistence AND the time-only
head, plus at least 2% benefit from correct versus same-person wrong audio with a
positive person-cluster 95% interval. Report whole-frame error, temporal error,
predicted/true motion-energy ratio, failures, both seeds and external set.
The mouth metric uses rows 8:24, columns 10:38 in 32×48 crops. Do not call a
static sharp copied seed successful generation. External gains must retain the
direction without tuning on external clips; four people limit uncertainty.

Audio: codec comparison and model reconstruction must be separate. Require at
least 20% relative reduction in an independent recognizer's word-error diagnostic
versus the old rendered model output, and improvement in mel/temporal errors.
For cross-modal audio, require improvement over mean/wrong-input controls; WER
and recognizer text are exploratory and fallible, never generated model text.
The existing small recognizer is evaluation-only; no external TTS/vocoder may
supply outputs. A codec or reconstruction win is not a video-to-speech win.

## Interpretation

Giving all audio in a clip conditions motion on that sound; it is not autonomous
causal future prediction. Unseen appearance without a visual seed is ambiguous.
Report that conditioning explicitly. Retain the cheapest useful representation;
do not scale again if a raw-feature or time-only baseline explains the gain.
