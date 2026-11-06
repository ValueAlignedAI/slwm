# G1: reusable state across source families

This study responds to task specialization in B1. Speaker-disjoint speech is
useful but does not establish a general world model. Keep T1-S5's independent
500M-token language models separate from these joint weights. G1 is an isolated
exploratory extension authorized by the user, not promotion past failed B1 gates.

## Hypotheses and frozen comparison

H1: heterogeneous predictive training yields a reusable state that helps a new
source/task after the core is frozen. H2: a nonlinear, chronological signal path
helps beyond an otherwise identical latent Transformer. H3: action-conditioned
state predicts consequences better than persistence or action-free histories.
H4: real phone timing, fine audio visibility and broader speakers improve speech
content without accepting mistimed video. Failure, negative transfer and weak
decoder ceilings are informative outcomes; no capability is assumed.

The configuration specifies both main seeds, a smaller paired scale, shuffled
alignment, no-future-loss and three full-capacity specialist controls. The
specialists consume exactly the corresponding target-family calls from the
mixed schedule; report their smaller training time and the aggregate cost of
three separate models. Match data, initialization of common weights, optimizer,
precision and update schedules within paired processor comparisons. The tiny
extra temporal-path parameter count is measured explicitly. These are smaller
joint models, not strict124M or core124M comparisons. Count adapters, embeddings,
all heads and any frozen components separately.

## Observations, architecture and objectives

One model processes scene images/captions, environmental sound/captions, speech,
paired speech video, wearable channels and robot scenes/proprioception/actions.
There is no dataset/task ID at the core. Modality/channel/time metadata are
explicit. No source is falsely treated as a synchronized observation of another
dataset. A from-scratch visual CNN replaces face PCA as the learned visual edge;
classical face crops remain a counted fixed preprocessing step for AMI only.
GPT-2 BPE is an edge codec with randomly initialized embeddings, not imported LM
knowledge. Vocos feature extraction is fixed; pretrained waveform rendering and
Whisper scoring, when used, are separate external weights.

At each received time, cross-attention compresses typed observations into four
latent slots. Causal binding processes a32-tick chronological field. SLWM adds
a local nonlinear update followed by a Fourier-parameterized causal31-tap filter
and an input-dependent gate. It never convolves over fictitious memory times.
The paired Transformer uses exactly the same Perceiver-style bottleneck,
adapters/readouts and chronological mask without this branch. Removing the
branch is the primary signal-path ablation. A bounded rolling context is working
memory; inference never changes main weights.

Generic time/modality queries read the received field. Speech repair may use the
whole received32-tick window and must report that buffering delay. Forecasts
cannot read future observations. Robot consequence queries may receive an
explicitly supplied planned action sequence; actor readouts cannot receive their
target commands. Action proposals are typed body-frame SE(3) deltas plus gripper,
not listen/look labels. No physical actuator is attached.

Objectives combine masked/future signal errors, actual MFA frame-phone and CTC
content losses, real-pair contrastive alignment, cross-view prediction and
action-conditioned consequences. Keep separate loss/exposure ledgers. Multiple
captions of one record are positives, not negatives. Activity labels remain out
of shared pretraining and are used only for new frozen-core readouts. Fine FFT
support masks prevent hidden waveform contamination; preserve all observed mel
frames exactly at rendering. Calibration can choose a residual/commit threshold
only from its registered role, before final tests.

## Data and leakage controls

Use full COCO2017 train/val with image-level caption groups and byte/exact-pixel
overlap purging; do not claim near-duplicate removal. Clotho roles additionally
group Freesound uploaders; purge lower-priority cross-role overlaps before fit.
PAMAP2 is the training sensor source. MHEALTH is absent from shared pretraining;
people1–3 fit new readouts,4–5 select,6–10 are final. Convert gyro units, expose
missingness and report side/axis placement shift. Common acceleration/gyro
transfer is distinct from ECG/magnetic channels without matched training units.

All40 dev-clean people appeared in prior metadata, so that source cannot supply
fresh confirmation speakers. An availability revision selects256 unexposed
train-clean-360 people using official speaker metadata before fitting:192 train,
16 selection,16 calibration,32 final. Retain the earlier failed freshness audit
and original archives. Reuse100h training speech with its prior exposure marked.
Actual MFA intervals replace within-word pseudo-phone timing; unknown/ambiguous
intervals lose lexical supervision while acoustic records remain.

AMI roles use full person-connected components. EN metadata lacks participant
camera identifiers, so a pre-fit metadata revision uses camera-complete IS/TS/ES
components, with24 new final people. Each new track is limited to its first900s
by a rule fixed before content scoring. Retain quiet and overlapping-owner
conditions as explicit strata; owner transcripts do not label all audible speech.
Old B1 training observations may be reused only in the training role.

All100 DROID debugging episodes are fully CRC/JPEG/action-contract audited.
Initial roles are collector-day disjoint. Raw building/scene metadata, where
available, must establish stronger group roles before any unseen-environment
claim. Keep unavailable metadata and source failures. A100-episode preview is
not a robotics scale corpus. The records lack timestamps: forecast horizons are
ordinal steps until cadence is independently verified. Rotation errors use
SO(3) geodesics, not wrapped or raw Euler-coordinate MSE.

Before fitting, freeze a dataset card with actual roles/counts/hours, all source
and prepared-array hashes, normalizers fitted on training data only, and the
configuration/source hashes. Data preparation can diagnose availability and
integrity, not select architectures from held-out scores.

## Evaluation and success gates

Primary speech: a fixed160ms gap schedule on32 fresh speech people and active
nonoverlapping-owner clips from24 fresh AMI people. Compare the practical fine
observed-only interpolation, audio-only, correct video and within-person
wrong-time video. Score phone/CTC error separately from rendered-waveform WER
and source/oracle ceilings. Preserve quiet/overlap strata rather than post-score
exclusion. A speech gain requires at least0.5 WER percentage points and a
positive person-clustered bootstrap lower bound. Video must add that gain beyond
the same model's audio-only arm; shifted video may worsen audio-only WER by at
most0.25 points. Both main seeds must pass to confirm the processor claim.

Primary transfer: freeze the field, then fit equal-budget readouts for MHEALTH
activity/forecasting and DROID action/consequence tasks. Compare full-capacity
specialists, raw-history readouts, invariant magnitude/spectral sensor features,
previous-command imitation and persistence. A broad representation claim needs
positive group-bootstrap gains on both source/task families, at least10%
relative improvement on their registered predictive metric, and no more than
1 WER point of speech degradation. Report every seed and all per-task effects;
an average cannot hide a collapsed modality. With few groups, wide intervals
mean insufficient evidence.

Robot consequences report translation, SO(3) and gripper errors at1/4/8 steps;
action removal/shuffling must harm prediction if action grounding is claimed.
Frozen actors predict commands from observed history with current target action
absent. Offline imitation and action sensitivity are not closed-loop success.
Any later procedural control experiment is labelled simulated and holds out
dynamics parameters/families; it cannot substitute for real-world validation.

Alignment reports cross-modal retrieval, same-person timing discrimination and
shuffled-pair controls. A collection of functioning narrow heads is not evidence
of transfer. Source-hidden readouts and input/action interventions carry the
generality claim. Commitment/uncertainty diagnostics report usefulness alongside
coverage; always-abstain does not pass.

Budget: at most12000 schedule positions/model, with3000/6000/12000 milestones,
128-update GPU-lock slices, validation-only selection and early termination for
nonfinite loss or two consecutive validation regressions over10%. No final test
is opened before every compared arm is frozen. Endpoints are primary; selected
weights carry their actual exposure. Report active GPU time, queue/I/O overhead,
throughput/peak memory and cold versus warm runtime separately. Fit no scaling
law from two sizes. Refine the wall-clock ETA from preflight measurements before
launching this matrix alongside the500M text continuation.
