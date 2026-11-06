# G1 frozen transfer and evaluation plan

Register this plan and metadata-only case lists before any G1 optimizer step.
Final observations may be prepared and integrity-checked, but final embeddings,
predictions, transcripts and scores stay unopened until every compared core,
readout and calibration rule is frozen. All endpoints and failed arms remain.

Speech uses up to eight deterministic hash-ordered, whole8–18s utterances per
fresh person, with usable official word alignment. Calibration uses four per
person. Each160ms gap is centered at the utterance midpoint;64/320ms gaps are
secondary. Whole official transcripts are the references. No reference text or
phone intervals enter the primary repair observation. Audio-only, correct,
static and±8/24-packet video conditions use identical weights and audio masks.
Predicted fine-phone and direct-mel renderers are separate candidates; calibration
chooses their fixed mixing coefficient from0,.25,.5,.75,1 by WER on calibration
people only. Keep the known waveform samples exactly, with a declared boundary
crossfade inside the hidden interval. Oracle phones, source waveform and source
mel/Vocos roundtrip are ceilings, not model predictions. Report full-pipeline
buffering, phone error, CTC WER and independent Whisper WER separately.

For each AMI person, freeze up to three8.192s windows per stratum: own active
nonoverlap, own active with overlap, no own annotations. Candidate centers are
every4.096s, at least4.096s from the recording edges; hash ordering chooses cases.
Own activity means word intervals occupy>=25% of the window; nonoverlap means
other-speaker intervals occupy<10%. Windows with little own activity remain in
the last stratum; no owner annotation does not imply acoustic silence. Cluster
primary intervals by person-connected meeting component, with person intervals
as sensitivity diagnostics. Report missing strata/counts, all raw transcripts
and reference-word-weighted errors. Negative timing tests receive no offset ID.

New readouts use frozen weights and identical examples/order/optimization.
Each condition supplies either the last shared state, edge-only temporal
summaries, raw history, or fixed sensor features. Fit training-only centering
and a maximum128-component SVD projection, zero-padding smaller feature sets;
count these fitted coefficients. A128→128→target SiLU readout, AdamW1e-3,
weight decay.01, batch256, at most2000 updates, validation every100 and three
non-improvements patience is common to all conditions. Seeds201/202; selected
validation state only; no readout backpropagation into the core. Normalization
and activity class weights use readout-training people/groups only. Report head
parameters, actual updates, inference and training time separately.

Raw sensor history includes all common-channel values and missingness. Fixed
sensor features contain per-site acceleration/gyro magnitudes, mean/std/range,
last values and temporal spectral energy. Raw robot images are area-reduced to
8×8 per view/time, retaining the whole32-step history alongside values/masks.
Edge-only controls pool learned edge outputs by temporal mean/std/last/slope;
they do not invoke shared processor blocks. Robot heads additionally concatenate
the same17 last proprioception/previous-command channels after the128-component
projection, giving145 input dimensions in every actor condition. Goal tokens
use the same trained text edge in edge/raw controls; no pretrained text model is
introduced. Feature transformations and target scales fit readout training only.

MHEALTH people1–3 supply up to10000 training windows,4–5 up to3000 selection,
6–10 up to2000/person final; starts are deterministic hash-selected without
replacement. Context32 and future8 at50Hz. Primary input uses common named
acceleration/gyro channels only, with real missingness; ECG is secondary.
Primary task is new activity classification: nonzero majority label over the
received context, ties by smaller label; report macro balanced accuracy and
macro F1 over all12 activity labels, plus per-person scores. Windows with no
activity label remain in forecasting. Secondary forecasting predicts future
1/4/8-step physical sensor residuals relative to the last observation and
compares persistence, last-velocity, raw-history and magnitude/spectral features.
The broad gate uses10% relative classification-error reduction over the strongest
selection-chosen baseline and a positive person-bootstrap lower bound.

DROID readouts use only verified connected building/collector groups with their
existing train/selection/calibration/test roles. Hash-select at most64 starts per
episode without replacement, context32 and future8 ordinal steps. The actor
predicts the current body-frame6D SE(3) command plus gripper from received state;
the current target command and future plan are absent from actor features.
Every condition also receives the same last observed proprioception and previous
issued command. Goal-present and goal-missing cases remain separate strata.
Controls remove/shuffle goals, retain only the last frame, and use previous
command/no-op/train-mean predictions. A raw-pixel/history and an edge-only
readout prevent attributing trivial sensor decoding to shared processing.
The primary actor score is absolute error normalized per dimension by the
training-target standard deviation (floor.01); report physical translation,
SO(3) and gripper errors separately. The gate requires10% improvement over the
strongest selection-chosen baseline and positive connected-group bootstrap
effect. A copied commanded pose is a consequence baseline, not an actor input.

Robot consequences separately predict actual1/4/8-step observed poses and
16×16 RGB changes from the same history, with supplied plans. Compare no-action,
within-batch shuffled-action, persistence, last-velocity, commanded-pose-as-state,
last-frame and zero-motion controls. SO(3) geodesics evaluate reconstructed
rotation matrices. Report image motion errors separately from static backgrounds.
Action sensitivity without lower held-environment error is not grounding.

Alignment uses independently encoded modalities, same-person distractors for
speech video and record-level negatives for scene/sound captions. Unseen
audio→image combination uses the previously fixed ten categories, official COCO
object labels and at least2/5 literal/synonym caption consensus in Clotho. Keep
all eligible multi-label cases and absent classes. Compare direct shared-space
cosine similarity against specialist→fixed text prototype→image similarity and
the shuffled-alignment arm; no paired sound/image training or physical-event
claim. Source counts and all category aliases are frozen in the case manifest.

All paired effect intervals use10000 group-bootstrap draws, seed7391. Report
both main seeds; do not pool the five sensor test people or five robot groups
into false per-frame independence. Speech degradation may be at most1 WER point
for a broad-transfer claim. Two model sizes provide a scaling comparison, not a
scaling law. The operational test replays over40s with repeated frozen-state
updates/proposals, deadline/staleness accounting and a weight-hash invariance
check. It is offline replay, not physical closed-loop control or online learning.
