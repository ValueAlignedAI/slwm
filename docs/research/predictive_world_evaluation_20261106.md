# G2 frozen transfer and dynamics evaluation

Prospective implementation and test roles freeze before primary core fitting.
Every compared core endpoint must freeze before fitting downstream readouts;
all readout weights/projections freeze before opening final observations.
No final metric selects a checkpoint, source role, head or calibration rule.

Use the mandatory512D state as the primary feature. The128D alignment projection
is secondary. Compare all12 trained models, four deterministic untrained
same-architecture controls (large/small SLWM/Transformer seed201), trained edge
features, raw history, physical invariants and fixed rules. Random weights are
reconstructed from the same frozen initialization/source, with zero core updates.

Fit training-only centering/scaling and128D SVD, then an identical128-hidden SiLU
MLP. Primary actor inputs use only the compact state. Raw/edge controls reserve
17 of their128 input dimensions for explicit received proprioception/previous
commands and use111 SVD components for remaining features. State/random probes
use128 SVD components and no raw bypass. Thus every learned actor has identical
128→128→7 head capacity, while raw controls keep a strong physical shortcut. Two readout seeds301/302; AdamW1e-3, batch256,
at most2000 updates, validation every100, patience3. Choose strongest baseline
on readout selection, never final data. Activity class weights and regression
target moments are readout-training-only. Preserve heads, projections and costs.

Primary withheld tasks are six-way UCI HAR activity and the current DROID
body-frame command/gripper inferred from received history. Neither task is a supervised core target; command streams do condition
dynamics pretraining. At readout inference, action plans/current target commands are absent from
actor features. These are supervised frozen-state transfer, not zero-shot
decision making. MHealth twelve-way activity is a secondary cross-source check.
All source-person windows remain grouped. Actor errors are absolute per-axis
errors divided by training-target std with.01 floor; also report meters, SO(3)
geodesic radians and gripper. Same-head missing/shuffled-goal, last-image and
no-image controls test input dependence without refitting. Keep missing-goal
cases and all failures.

For all paired intervals use10,000 independent-group bootstrap draws, seed7391.
HAR primary units are its nine official final people; robot units are connected
building/collector components. For classification resample confusion matrices,
report macro recall/F1 over all six (or12 MHealth) classes. Regression uses equal
independent-group means. Never treat overlapping windows as independent trials.

Report separate preregistered decisions:

- Useful transfer: both large SLWM seeds improve at least10% in relative primary
  error over the strongest selection-chosen baseline with positive lower effect
  intervals, on both HAR and actor tasks. Failure is retained; aligned-projection
  success cannot replace a failed mandatory-state gate.
- Retention: separately report whether HAR error is within2 absolute points of
  raw history, and actor error within5% of raw history, including paired intervals.
  Noninferiority alone does not become a superiority/general-world claim.
- Architecture: compare mandatory/bypass and history/point at matching seed201,
  size and exposure for both processors. These one-seed mechanism controls need
  replication before a universal mechanism claim. Main SLWM/Transformer and
  size comparisons retain both seeds and active cost.

Recorded consequence tests use actual future commands to predict1/4/8-step
physical poses/gripper and native16×16 RGB changes. The primary moving-pose
metric is eighth-step mean squared error over XYZ, continuous6D rotation and
gripper, normalized by G2 training-channel scales; physical component errors
are always reported beside it. Up to32 uniform and32 moving starts/episode
are fixed before training, with duplicates merged and stratum membership kept. Compare persistence,
last-velocity, commanded-pose-as-state, no-plan, shuffled plan and prefix-shuffle
with the final command fixed. Score unconditional and the fixed moving stratum;
report physically timed versus ordinal subsets, SO(3) and invalid-rotation
penalties, plus image-motion errors separately from background MSE. Changing
plans does not supply counterfactual physical outcomes. Improvement requires
positive group intervals and5% relative reduction of eighth-step moving-pose
error over the point control; do not reward prefix sensitivity alone.

Keep independent-modal semantic retrieval as a diagnostic: image-caption and
sound-caption retrieval, and the original fixed ten-category unpaired
sound→image combination. Record repeated-ID/multilabel chance and per-category
counts; compare matched Transformer and untrained/edge controls. No matched
physical-event interpretation follows from category agreement.

The live check processes more than40s of consecutive observations with persistent
GRU memory, trained actor proposals and no weight updates. Log packet times,
missed deadlines, stale rejection, memory-reset control and weight hashes.
Only the newest encoded packet updates memory. Native robot cadence is used
only where verified; chosen replay pacing is labeled. No proposal changes the
recorded future inputs, so this is not physical closed-loop control.

Export endpoint state/dynamics weights, config/source/data hashes and trained
readout bundles with a load/infer example. All12 fixed-endpoint results, selected
positions, actual exposure and failed checks remain visible. Checkpoint/source
roundtrips and implementation tests are engineering evidence, distinct from
final held-task results. Two sizes are not a scaling law. Future G3 changes must
use new confirmation data after these final outcomes are opened.
