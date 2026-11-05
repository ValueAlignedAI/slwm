# T2-C1 — speech content and rejecting mistimed video

Registered before new model training. The user requests better content,
mistimed-video rejection, fresh speakers and responsible scaling. Keep this
small audiovisual study separate from T1-S2's GPT-2-scale English comparison.
No124M audiovisual pretraining is justified by the available acted-speech hours.

## Frozen data roles and overfitting controls

Use original GRID media with its embedded soundtrack, official word alignments
(25kHz sample clock), fixed training-example crop and per-stream PTS alignment.
No endpointed external waveform replaces the soundtrack. Every source ZIP has
an acquisition SHA256 and every used member its CRC; all arrays are hashed.
Train people01–16 on sentence-transcript SHA256 buckets0–7. Checkpoint-selection
people17–20 use bucket8; threshold-calibration people22–24 use bucket8. Final
people25–32 use bucket9. Thus primary test has both new people AND sentence
combinations absent from training/selection/calibration. Speaker21 has no video;
33/34 stay unopened. Secondary seen-combination and training-person composition
holdouts are diagnostic only. No model or threshold is chosen on final scores. Source clips below the original50-packet input criterion are recorded as exclusions; more than0.5% invalid sources stops preparation for review, before any model fitting. The exact90 short source videos from training actor08 pass that alarm only after [documented raw-CRC/frame-count review](grid_source_quality_review_20261105.md); input criteria and held-out people stay fixed.
Vocabulary comes only from training alignments. GRID's six-word grammar and
~51-word vocabulary limit all semantic/recognition claims.

Keep earlier CREMA normalization fixed; fit none on GRID validation/calibration/
test. Mix audio-base updates24GRID+4CREMA+4earlier Libri training windows, and
candidate updates24GRID+8CREMA. Unit losses ignore non-GRID labels. This retains
broader acoustic training while adding thousands of new sentence combinations.
Record actual unique clips/hours and repeat exposure. Select minimum GRID
validation320ms gap-MSE every400steps. Stop after three consecutive checks>5%
above the prior best, retaining all best/final weights and exposing any unequal
budgets. Calibration actors never fit neural weights. Nominal base4000updates,
candidate3000, batch32; two primary seeds. The fixed finite budgets, composition
holdout and speaker holdouts constrain overfitting; they do not guarantee it
cannot happen. Report training/validation curves and all negative results.

## Content architecture and controls

Warmstart the earlier R2 speech core, then adapt it on the expanded training
mix. A word-distribution head reads the processed audio field. A tied timed-word
embedding conditions the mel decoder; explicit supplied text is an optional
separate mode. Primary audio/video inference never receives transcription or
word timing. Every fourth update adds a timed-text reconstruction loss to train
that interface; all audio variants receive it. Compare unit CE weight0 versus.2
in two seeds, with otherwise identical initialization/data/budgets. Retain the
observed-sound field readout at.5. The ablation isolates added unit supervision,
not removal of all text supervision.

Freeze the selected content base702/704. Train equal-capacity additional audio
and video candidates, each changing unit logits, latent field and a bounded
fine-rate spectral residual. Video uses a spatially pooled CNN, clip-consistent
translation augmentation, temporal features and local cross-attention. Audio
uses a larger frontend to approximately match graph-connected trainable weights.
Both candidates get the same positive windows/unit objectives. The deployed video
model also stores the frozen additional audio candidate, and rejection returns
that STRONG comparison output exactly. Count all deployed weights, including
fallback; this is a candidate-capacity comparison, not equal total memory/FLOPs.
Also train a video no-unit-loss and shuffled-pair arm; evaluate unconditionally
accepted versus calibrated rejection on the same frozen weights.

## Observed-only synchronization and fixed rejection rule

Learn normalized per-frame video/audio matches and offset logits for−8..8.
The audio projection sees only actually observed audio; windows touching the
true gap are masked and deltas formed after observed-only interpolation.
Use real shifted crops at training magnitudes2,3,5,6,8packets, plus same-person
different-utterance negatives. Train offset CE and aligned/not-aligned BCE, not
target-dependent inference risk. Primary scores never use hidden target audio.
Quality is sigmoid(aligned logit)×P(offset0). Accept only when predicted offset
is zero and quality≥a threshold calibrated on people22–24.

Choose the lowest threshold among1001fixed values that admits≤5% of each polarity of±240ms
calibration shifts; it maximizes aligned coverage under that cap. Require≥50%
aligned coverage, otherwise mark calibration as failed even if rejection is
conservative. No final-test threshold adjustment. Wrong utterances are a
secondary diagnostic: content differences confined to erased audio can be
indistinguishable from a trustworthy source using observed context alone.

## Registered primary gates and diagnostics

True320ms gaps,1.28s center context, support-aware masking and exact observed
PCM preservation.160/480ms are secondary. All shifts use real adjacent frames,
never circular wrap. Fresh people/sentence scores are computed after all neural
checkpoint choices and thresholds are frozen. Use2000-draw person-cluster95%
intervals, with eight final people.

1. Content objective: both seeds improve missing-region non-silence word accuracy (mean paired per-clip non-silence accuracy)
   from a separately training-fitted frozen-field ridge readout by≥5percentage
   points with positive intervals, versus the no-unit-loss base. Gap-MSE≥3%
   better, and clean-MSE no more than5% worse. The ridge sees processed fields,
   never raw bypasses. Fit the first128 training clips/person, train-whitened fields, fixed ridge.01; no held-out tuning. Save a time-only baseline and letter/digit breakdowns.
2. Rejection: on fresh±240ms video, acceptance≤5% (bootstrap95% upper≤10%),
   aligned coverage≥50%, and mean per-clip spectral harm vs the strong audio
   candidate≤1% (95% upper≤3%). Also report unseen-training-magnitude±160ms,
   wrong/static/blank/missing video and raw ungated corrections. Report coverage
   alongside harm so rejecting every video cannot pass.
3. Useful video: calibrated prediction improves mean paired per-clip gap-MSE≥3%
   over the matched extra-audio candidate in BOTH seeds, with positive intervals.
   Missing video equals the strong audio candidate exactly. A practical speech
   claim also needs≥5% relative WER reduction and positive person-cluster
   intervals against audio and erased-gap silence in both seeds on the fixed
   recognizer subset. The external recognizer is evaluation-only and fallible. Normalize case/punctuation, separate adjacent letters and digits, and map single digits0–9 to English words for both reference and hypothesis; do not correct grammar or letter homophones. Retain raw WER too.

Fix four evenly spaced test IDs/person (32) before scores for real PCM output
and recognizer checks. Save reference, erased, interpolation, base controls,
audio candidates, raw and rejected video, shifted/wrong/static controls and a
clearly labeled true-spectrum decoder diagnostic. Timed-text and forced-word
readouts are separate diagnostics, never main evidence for audio/video repair.

This is offline repair with future observed context, not causal dynamics or a
general world model. T1-S2 separately measures causal text prediction. Evidence
from these studies must not be merged into an untested jointly trained model.
Source motivation: [GRID](https://spandh.dcs.shef.ac.uk/gridcorpus/),
[masked audiovisual unit prediction](https://arxiv.org/abs/2201.02184).
