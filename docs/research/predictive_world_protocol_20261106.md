# G2: mandatory predictive state and causal action history

Registered before primary optimizer updates. User-authorized follow-on to G1's
task-specialization audit, not a modification of G1 or the500M language study.

## Questions and architecture

Can one compact predictive state preserve information useful to tasks absent
from core training? Do accumulated action effects improve recorded consequence
prediction over independent-command queries? Does a signal branch help relative
to a parameter/data-matched Transformer once both use the same state/dynamics?

Typed image/audio/numeric/supplied-text edges feed a causal four-slot field.
A GRU posterior compresses that field into512 scalars at every received tick.
Every reconstruction, phone/character, future and alignment head receives this
same state. There is no raw-observation persistence/interpolation residual in
the G2 loss. Time/type/view query metadata are allowed; raw content is not.

Future states roll forward recurrently from received memory, consuming the full
preceding command sequence and measured elapsed intervals. Missing plans have
an explicit presence flag;10% of robot training positions omit the plan. A
missing plan is distinct from a supplied mean-normalized command sequence. Later commands cannot
change earlier priors. A point control restarts every query from received memory
and consumes only its own command. A bypass control adds full-field attention to
the shared query using existing attention weights. These controls have exactly
the same parameters as their mandatory-state partner, but different computation.

Streaming consumes only the newest encoded packet of a causal received window;
it carries GRU memory between ticks without replaying the window through memory.
Tests require equality to a batched causal prefix pass. No inference weight
updates, physical actuation or autonomous reward-learning claims are made.

Motivation: recurrent latent dynamics in [DreamerV3](https://arxiv.org/abs/2301.04104)
and action-conditioned prediction in [V-JEPA2](https://arxiv.org/abs/2506.09985).
This deterministic, from-scratch mixed-source study is not a reproduction of
their data scale, stochastic state model, policy training or robot deployment.

## Scale and comparisons

`configs/t3/predictive_world.json` fixes12 runs:

| IDs | Configuration | Seeds |
|---|---|---|
| G2-201–204 |800×12 SLWM/Transformer,512-state |201/202 |
| G2-205–208 |512×10 SLWM/Transformer, same512-state |201/202 |
| G2-209/210 |800×12 with field bypass |201 |
| G2-211/212 |800×12 with point dynamics |201 |

Total learned inference parameters are approximately124.5M/124.1M at the anchor
and55.8M/55.6M at the smaller size; exact measured counts are saved per run.
The matched differences are<0.4%. All edges, posterior, dynamics and trained
heads count. No pretrained vision/language generator is hidden outside these
totals. The Fourier-coordinate31-tap branch is a nonlinear-path comparison,
not evidence that Fourier coordinates outperform direct-time FIR coefficients.

Each run receives12,000 global updates with the same family/example positions
for its paired seed, two accumulated microbatches/update, BF16, AdamW,
512-update warmup to2e-4 then cosine2e-5, weight decay.1, norm clipping1,
EMA.996. Microbatches128 for images and32 for temporal data give effective
batches256/64 at32 packets. One quarter of training positions use64 packets
with half the temporal microbatch, preserving temporal packet budget while
training longer memory. Validation/readout context stays32. Contrastive negatives and covariance estimates remain microbatch
sized; they are not falsely described as global-batch negatives. Validation is
fixed every512 positions and at3k/6k/12k milestones. Two successive macro-loss
rises>10% above selection best stop a run; incomplete exposure is reported and
cannot support a fixed-endpoint superiority claim. Preserve endpoint/best and
resumable states, both source snapshots and actual cost/exposure logs.

Parameters/examples are matched; active compute is measured, not equalized.
Time-matched validation curves are descriptive; no fresh compute-matched test
is inferred from interpolation. Two sizes/two seeds are a bounded scaling
comparison, not a fitted scaling law or compute-optimal foundation training.

## Real data and contamination controls

The fixed call mix is30% COCO,35% Libri speech,10% meeting streams,2% Clotho,
8% PAMAP and15% DROID. Existing source identity roles, byte deduplication and
training-only nonrobot normalization are retained. Speech has176.1h/443 Libri
training speakers; COCO has118,272 train images. No independent-event hours are
manufactured by adding synchronized cameras/participants. Clotho uploader
purges and its small unique duration remain explicit. Captions/goals are text
edges; the independent500M-token language weights are not joint G2 knowledge.

Robot data expands to the first complete records of1536 fixed hash-selected
official DROID1.0.1 shards. This is not uniform episode sampling. Verify pinned
generation, HTTP range, complete-record CRC and payload SHA; preserve source
receipts, metadata failures and decoded-cache hashes. Only newly downloaded,
reconstructible transport copies may be removed after durable cache receipts.
Unchanged G1 caches can be linked, never edited. Source building/collector
components define roles before fitting; require at least400 verified training
episodes and five final components. Missing language remains missing.

Accept physical robot time only where raw HDF5/RLDS poses match one-to-one within
2e-6, native seconds+nanos are finite and strictly increasing. Accept an exact raw
prefix with at most one omitted raw tail frame; every RLDS pose must match.
Reject unresolved duplicate collector metadata without selecting a candidate. Integer timestamp
subtraction precedes float conversion. Keep ordinal flags otherwise. Body-frame
SE(3) commands use the last received reference pose, never a future observed pose.
Half of training starts may draw from motion windows; the other half are uniform.
Motion means eighth-step translation>=1cm OR rotation>=.1rad OR gripper>=.1.
This stratum is fixed before fitting, with unconditional results also retained.
Robot moments are recomputed from G2 training components only: old G1 robot
moments must not leak across the newly constructed G2 roles.

The new [UCI HAR source](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
is absent from core training. Preserve official21/9-person train/test identity
split; hash-select16/5 of the21 for readout training/selection. Use total accel
and gyro, convert g to m/s², anti-alias downsample128 to32 packets over the full
window, record12.5Hz. The waist phone deliberately changes placement; six scalar
channels do not imply that it was a wrist sensor. Published50% overlapping
windows stay in the same person role; bootstrap people, not windows. All normal
and missing-sensor/raw controls see the same signals. Preserve source CC BY4.0.

## Learning and falsification

Direct fixed-target reconstruction and future losses must improve through the
state; activity and current-command inference are not supervised core targets.
Command streams are explicitly conditioning inputs to dynamics pretraining. EMA future targets encode
received past plus actual future in a causal teacher-only continuation, not an
independently reset future window. Variance/covariance penalties operate in
FP32; off-diagonal covariance is summed/divided by dimension, not diluted by
dimension squared. Monitor sample rank and retrieval with duplicate-aware chance.
G1/A1 failures motivate this design but do not select G2 final cases or outcomes.

Hardware preflight increased microbatches only after all four processors fit
within19GiB peak allocations, preserving effective batches and improving
negative/covariance sample counts. First run real-source gradient/optimizer/memory/checkpoint preflight at both
sizes. The original preflight's missing environmental-audio phone-key failure
and source remain saved; the fixed revision is separate. Primary training waits
for complete data/source freeze. A512-update screen of the four main models may
share the GPU queue first; the remaining fixed run waits for previous study
weights to freeze. No test outcome changes its recipe.
No G1 final scores are used to tune this registered recipe. Then all G2 cores,
readouts and rules freeze before the G2 final lock opens observations for scoring.

Failure includes state collapse, raw/random features explaining transfer,
motion forecasts no better than persistence/point controls, lost speech content,
or gains that vanish against the matched Transformer or independent-group
intervals. Keep failed artifacts and distinguish a better general bottleneck
from a better signal branch. See the separately fixed evaluation protocol.

Initial ETA is48–72h including the existing3090 queue. Replace estimates with
measured per-family preflight costs; do not call a short smoke checkpoint a
completed scale experiment.

## Execution and artifact validation

`python -m training.predictive_pipeline` runs complete-data gradients, the fixed
512-update screen, the queue-aware12-run fit, frozen readouts, the final lock,
independent task/dynamics/alignment/content evaluation and41s ongoing replay.
Every phase records logs/status and resumes only the identical source recipe.
Actual exported endpoints and their state-only actors are reloaded for fixed-case
state/action/future inference, checked for finite outputs and repeated-inference
equality. These engineering checks cannot substitute for final transfer scores.

The Mac delivery worker hashes every inference/result file against Fedora, keeps
resumable optimizer/EMA states on Fedora and publishes curated findings. It may
remove immutable GPU-host milestone copies only after identical local copies
are verified. Earlier151M-token parents remain until the500M study final summary.
Sources and all failed preflights remain preserved. No main G2 optimizer update
occurs until the data and complete source/evaluation recipe are frozen.
