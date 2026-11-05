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
and score the model's future head against training-only future recordings. Labels
and future signals never enter the decision input. The actor minimizes predicted
goal-weighted future error + sensor cost. Fit1500 Adam steps, batch256, lr.001,
seed391, retaining the risk weights and training sources.

Use five fixed40.96s training starts per person (120,170,220,270,320s), stopping
an episode at the common signal endpoint with the valid length recorded. Use
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
