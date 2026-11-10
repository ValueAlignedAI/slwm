# P1-S1: ongoing decisions from a frozen shared world representation

Registered before final world/policy test scores. This is a recorded-data
active-sensing environment: the agent continuously chooses which sensor to read,
updates working memory, forecasts and chooses whether to commit a proposal.
Actions change subsequent observations. They do not change the recorded physical
world; this is not humanoid/drone motor-control evidence. A robot would additionally
need action-conditioned physical transitions, embodiment data and faster control.

Freeze T2-B1 best checkpoints801/802/803/804 using their existing selection only.
Never update world weights during policy fitting or operational inference. Use
AMI train participants for fitting, independent calibration participants for
thresholds, and reserved12 test participants for final episodes. All choices and
costs below precede test inspection. The world-selection participants do not fit
the policy or calibrate its commitment threshold.

Each1.365s tick may listen, look, read both, or wait. The next chunk exposes only
the requested original sensor values; text labels remain unavailable. One shared
bootstrap chunk initializes every compared agent identically, with equal cost.
World memory persists; no new user request triggers the subsequent30 ticks.
The goal (forecast audio, video or both) is an explicit task input, not learned
free-language understanding. Each active modality costs0.05 per tick.

Fit a128-hidden risk regressor from current processed-field mean, memory mean,
current model risk and previous sensor bits. It predicts both audio and video
forecast error for each of four next sensing actions. Training targets enumerate
the four counterfactual observation choices, run copies of the same frozen state,
and score the model's future head against training-only future recordings. The
target is eight packets starting two packets beyond the newly received chunk;
audio error is normalized-mel MAE and video error is normalized-code MSE. Goal
cost is audio error,0.2×video error, or their sum, respectively. Fit log errors
with squared loss; standardize actor features using training means/deviations
(deviation floor0.05). Labels
and future signals never enter the decision input. The actor minimizes predicted
goal-weighted future error + sensor cost. Fit at most1500 AdamW steps, batch256,
lr.001, weight decay.01, dropout.1, seed391. Hold out six of the28 world-training
people from all risk-head fitting, ordered by SHA256("P1-RISK-"+global_identity).
Fit feature statistics only on the other22 people. Select the lowest held-person
log-risk MSE every100 steps and stop after two checks >5% above the best. Retain
best/last optimizer and RNG states. This added pre-fit regularization/selection
rule is registered before any policy weights or final test scores; it addresses
the small number of independent risk-fitting people. The four genuinely new
calibration people remain reserved for commitment thresholds, not head selection.

Use five fixed training starts per person (120,170,220,270,320s). Each episode has
one bootstrap chunk and30 automatic decisions, requiring42.325s of observations
plus0.427s future-target support. Pacing the30 decisions takes40.96s after the
bootstrap. Reject an episode that lacks the registered common signal support;
never silently pad or truncate it. This clock clarification precedes policy fits
and test scores. Training trajectories use a fixed seeded uniform action stream;
all four counterfactual actions are labeled at each encountered state. Use
independent calibration starts120,170,220s to set per-goal70% predicted-error
commitment coverage. No threshold tuning on test. Test uses the same fixed80s
start for every reserved person. Compare learned sensing, always-listen,
always-look, always-both, always-wait, deterministic alternating and periodic
half-rate sensing, with all sensor costs reported. Retain all episodes.

Primary utility counts future prediction error on EVERY tick, whether or not
an output is committed, plus sensing cost. Abstention cannot hide errors. Require
lower person-averaged total cost than the strongest fixed baseline, with a
positive person-cluster95% paired interval, both seeds, and test commitment
coverage at least50%. Also report raw errors, sensing fractions, conditional
committed error and coverage. The Transformer policy receives identical features,
head capacity, examples and selection rules. A failed gate stays failed.

Aggregate the three registered goals equally for the primary per-person utility;
also retain each goal separately. Each goal has the same episode/tick count.
Policy registry IDs101–104 map to world runs801–804 as recorded in config. These
are separate small fitted decision heads, not additional world-model pretraining.
Additionally report active/quiet future-target intervals using official own-speaker
word/phone support. That label is evaluation-only and never enters observations,
features, action choice or commitment. The primary utility retains every tick.
The equal bootstrap costs0.1 per episode, reported separately from subsequent
per-tick utility; adding that shared constant cannot change paired effects.

Before policy fitting, register an additional direct-sensor actor111–114 for
each world, with the same1027 inputs,128 hidden units, initial head seed, fitting
people, labels, dropout/selection and separate70% commitment calibration. Its
state contains observed-only audio/video feature means (400+64),32 pooled audio
temporal deviations and16 pooled video deviations, plus a512D exponential
running memory (0.8 previous+0.2 current), current audio-statistic energy and
previous sensor bits. Absent sensor values are erased before every statistic.
It receives no processed field, learned risk, labels, transcript or future signal.
Reuse exactly the field actor's training trajectories and counterfactual labels.
Raw and field heads both control the same frozen world forecaster at evaluation.
The original fixed-baseline utility gate is unchanged. Separately require a
positive person-cluster field-versus-direct-sensor utility interval, both seeds,
and field commitment coverage≥50% before claiming added decision value from the
processed state. This is one simple direct-sensor control, not proof that all
possible raw-history policies are weaker.

Run one40.96s paced CPU episode while the3090 continues other work when feasible.
Record every automatic tick, sensor action, source timestamps, output commitment,
working-state norm, wall latency and missed deadlines; model-weight hashes must
match before/after. Other episodes run accelerated replay with the same state
machine. This measures an operational loop over files, not live camera capture
or safe flight. Long audio/video/text rendering remains separately labeled T2-B1
diagnostics; output pixels alone do not establish useful decisions.

This isolates one step toward embodied operation: value of additional sensing
from a shared state. Next physical-action work requires a separately registered
simulator/action dataset and multi-step consequence/reward tests. The general
motivation is consistent with [Dreamer](https://arxiv.org/abs/2301.04104), whose
action-conditioned imagined control is a stronger capability than this probe.

Pre-fit pacing refinement: choose the next sensor from the current state, wait
until that chunk's recorded arrival deadline, and only then process its values.
Log decision, observation-availability, observation-read and proposal wall times
separately. Compute budgets exclude the intentional wait. Sleeping only after
processing a preloaded future chunk would not test arrival ordering. The same
refinement applies to the label-free file CLI, with a fake-clock causality test.
The final paced episode spans at least40.96s after bootstrap plus its last compute
time. Features remain precomputed; this is a causally paced file replay, not a
live capture transport. Utility, fitting examples and all selection rules stay
the same, and this change precedes all primary policy fits and final scores.
