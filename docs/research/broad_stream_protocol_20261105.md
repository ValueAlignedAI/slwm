# T2-B1: broader speech and long shared-field decoding

Register before generator fitting. The question is whether phonetic/lexical
supervision, recurrent working memory and one audio/video/text field improve
decoded content on broad speech, and how that compares with a Transformer.
This is separate from T1-S4 text weights and subsequent operational policy work.

## Data and frozen confirmation

Use original LibriSpeech train-clean-100 (~100h available); dev-other selection
uses20 deterministic speakers/12 clips each. Keep all retained training durations,
including short clips with padding/masks. Plan12 test-clean and12 test-other
people not used in earlier SLWM train/selection/test records, four8–18s clips
per person, sorted by the registered B1 SHA256 ordering. Exclude prior people,
not inconvenient scores. Test recordings already stored but unused people are
fresh for the trained generator; the external vocoder has its own disclosed data.

Use the AMI participant groups in broad_stream_data.json:28 train,4 selection,
4 calibration,12 test. Official global identities are verified disjoint before
training. Use the original individual headset WAV and mapped closeup AVI;
all sessions are continuous and much longer than30s. Restrict only to the common
audio/video timeline, logging endpoint differences. No held-out crop tuning or
discarding failed cases. Classical face tracking is causal, checked on training
footage before holdout preparation, with an explicit full-frame fallback.

The fixed source releases/checksums and metadata plans precede weights. Fit mel
statistics,64D gray64×64 video PCA and a word-duration regression on training only.
Use the fixed CMU pronunciation dictionary as an external text codec. Actual MFA
word boundaries supervise LibriSpeech; AMI uses official words. Phones are spread
uniformly inside each word: these are approximate phonetic targets, not claimed
true phone timings. Free-text inference estimates durations from the training
regression and receives no reference timing. Dictionary/codec hashes are saved.

## Architecture, comparisons and exposure

From-scratch shared fields use512 width,6 layers,8 memory slots,32-packet chunks.
One packet is4×256/24,000s. Each received chunk allows up to1.365s lookahead plus
the21ms analysis half-window; this is bounded offline streaming, not sample-causal
robot control. The working state persists between chunks; no inference weight
updates occur. A future head forecasts8 packets beginning2 packets beyond the
chunk boundary, avoiding overlap with observed FFT support.

Paired Transformer and SLWM seeds91/92 share all common initial weights, batches,
heads, codec, effective batch128 and2000 updates. SLWM adds local/frequency-kernel/
long convolutions; report exact counts and timing, including auxiliary passes.
Single-seed late-fusion, shuffled-AV-pair, no-content-loss and reset-memory controls
share positive plans and exposure. Drop all received modalities in10% of chunks and
half the static appearance cues during training, preparing working memory for
intermittent sensing. All RNG draws remain architecture independent. Late fusion uses three2-layer modality stacks
at the same total core budget, with separately carried states and shared edge
heads; active depth differs and is disclosed. Small256-width SLWM/Transformer
arms are an exploratory single-seed scale comparison, not a scaling law.

Batch contains75% Libri/25% AMI. Mix masked repair, audio→video, video→audio and
explicitly provided-text decoding; all task/crop/negative RNG draws are independent
of model dropout/initialization. Train mel, video/motion, field phonetic CE/CTC,
blind cross-modal alignment, future-signal and activity/risk heads. The no-content
control removes added phonetic/CTC losses only; it still has text-conditioning
reconstruction. Wrong-video inputs/targets use a different time from the same
training person. Targets never enter practical audio/video inference.

Vocos revision0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21 is a shared frozen external
waveform renderer. Count its weights separately. No pretrained video generator
or language model trains this core. Speech claims concern the new core + declared
renderer, not unassisted synthesis. Include recorded, oracle-mel/vocoder, silence,
linear waveform, and interpolated-mel/vocoder controls. Known PCM is clamped
exactly in repair mode; full video/text-to-audio synthesis has no reference phase.

## Selection and gates

Validate every200 updates on fixed selection plans (Libri128, AMI32 windows).
Select the lowest registered aggregate loss: masked-mel MAE +.2 audio→video code
MSE +.2 video→audio MAE +.2 text→audio MAE +.05 phone CE. Stop after two checks
more than5% above the best; log any exposure mismatch. No recognizer/test score
selects weights or changes targets. Freeze all choices before final tests.

Primary speech gate: at least5% relative WER reduction over the stronger silence/
interpolated-mel control with a positive person-cluster95% interval, both seeds,
on both fresh Libri sets at160ms. Compare matched Transformer WER and spectral
metrics explicitly, without assuming SLWM wins. Report320ms separately. A WER
loss despite lower spectral error is a failed practical gate. Pinned external
Whisper-tiny.en scores all fixed cases; its outputs never train the generator.

Shared-field analysis: independently encoded audio/video/text, within-person
retrieval against same-person wrong windows, field-only phone outputs, future
prediction versus persistence and Transformer, and all registered ablations.
Report uncertainty and failures; alignment/probes do not establish understanding.

## Continuous generation and operational limits

For every reserved AMI participant, run the fixed40.96s interval at80s through
persistent chunks. Save audio→video and video→audio, model/Transformer/constant/
shuffled controls, native64×64 video, frame counts, all audio, state/timing logs
and the actual inputs. Audio→video uses a declared training appearance preset;
video→audio reads no source audio. Appearance comes from a provided frame, not
invented scene knowledge. Long text prompts are fixed in config; save every
text→audio/video output using estimated timing, including poor results. These
probes test continuity, intelligibility, correspondence and drift beyond30s;
they are not arbitrary-scene text-to-video or evidence of motor planning. Include
a prefix-only control: after eight chunks (10.92s), withhold all new observations
for the remaining30.04s and log the free latent rollout, including collapse.

Sources: [LibriSpeech](https://www.openslr.org/12),
[AMI](https://groups.inf.ed.ac.uk/ami/download/),
[AMI signal mapping](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml),
[MFA release](https://github.com/Tom-Brenner/librispeech-vctk-phone-alignments),
[Vocos](https://huggingface.co/charactr/vocos-mel-24khz).
Keep source permissions/attribution; AMI/LibriSpeech are CC BY4.0, separate from
repository code licensing. Subsequent learned sensor/action policy training must
have its own frozen world checkpoint, protocol and held-out evaluation.
