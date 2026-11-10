# Short compiled learning pilot, registered before fitting

The disposable compilation probe measures 1.458x/1.524x steady-step throughput
for SLWM/Transformer. First gradients differ 0.271%/0.248% relative, with cosine
above .999996. SLWM's initial loss difference .001196 misses its registered .001
absolute tolerance; that gate remains failed. The Transformer local gate passes.
Neither result establishes preserved learning quality. Keep the frozen B1 matrix
unchanged, retain the original benchmark and do not reinterpret its threshold.

This separately identified exploratory pilot tests validation learning directly,
using only the original train and selection people. It does not read new test
people or become an additional primary B1/policy candidate. Run two paired
randomly initialized full-size models, seeds91, for exactly400 updates and select
using the same200-update validation as B1. Everything matches B1 except compiling
`StreamWorld.process` in default mode and the declared400-update cosine horizon.
The shortened LR horizon is shared with an eager reference inside this pilot;
comparing against B1's2000-horizon first400 steps alone would confound execution
with the learning-rate schedule. Therefore retain four pilot arms: eager and
compiled SLWM, eager and compiled Transformer, with identical400-step schedules.

Compute both arms with the same sampled window order, global/micro128, losses,
initialization, AdamW and precision. Local source/config snapshots must freeze
before any update. Use the shared GPU lock and release it every200 updates.
Bound this pilot to those400 updates; no validation-based budget extensions.

Primary pilot endpoint is the recorded validation loss at update400 (not the
best of different endpoints). A candidate clears the exploratory learning check
only if its selection loss is no more than1% above its paired eager arm's loss.
Also retain all component losses, selected state, full optimizer/RNG, wall time
including cold compilation, and steady-step benchmark data. No test metric is
used for this check.400 updates/single seed cannot establish converged or
fresh-speaker equivalence. Original B1 curves at400 remain contextual only.

The compiled/eager implementation preserves parameter names and mathematical
architecture; numerical kernel fusion changes BF16 arithmetic. Portable pilot
states can load through the original eager model. These are auxiliary runtime
research artifacts, not an extra unified multimodal/robot-control claim.
