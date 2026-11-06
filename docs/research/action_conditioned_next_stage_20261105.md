# From shared observations to action-conditioned worlds

The next physical-action study should test controllable transitions, not simply
increase the speech model's width. The current P1 actor chooses recorded sensors;
the B1 future head predicts only eight packets beyond an85ms guard. A40.96s
conditionally generated file is not evidence of a40.96s planning horizon.

## Verified next-data interface

The official DROID debugging corpus is100 episodes/2.193GB in31 TFRecord shards.
Only the first27.49MB shard was downloaded for this preparation check. Its two
episodes contain166/238 steps, three320x180 RGB cameras, proprioception, commands
and three language fields. All1,212 JPEG frames decode. Source generation IDs,
GCS MD5, local SHA256 and TFRecord CRCs are retained. No model was fitted and no
robot command was sent. Reproduce with `python -m training.action_data_probe`
and `python -m training.action_data_review`.

In these two debug1.0.0 records, the default seven-dimensional `action` equals
commanded Cartesian pose6 plus commanded gripper position1 exactly. It does not
match the velocity vectors suggested by the schema comment. This is a measured
sample-specific contract, not an assumption about every DROID version. The
authors document XYZ Euler orientations in their
[conventions discussion](https://github.com/droid-dataset/droid/issues/9).
Raw orientation-coordinate discrepancies exceed6.2, while wrapped-coordinate
discrepancies are below0.151. Future metrics need a proper rotation representation;
raw Euler L2 would manufacture large errors at the wrap. Wrapped coordinates in
this probe are a diagnostic, not an SO(3) geodesic measurement.

These TFRecords have no timestamp feature. Obtain verified collection cadence
or raw robot/camera timestamps before assigning physical seconds to transitions.
Keep observation, commanded action and resulting observation in explicit order.
The two episodes validate parsing only; they cannot support scene-generalization
or scale claims. The [official guide](https://droid-dataset.github.io/droid/the-droid-dataset)
offers the full1.7TB RLDS corpus. Use a pinned, bounded streamed subset with raw
environment/session metadata rather than assuming the full corpus fits locally.

## Minimal next controlled model

Build a separate action-conditioned predictor from three scene encoders,
proprioception, a typed action edge and a goal edge into persistent shared state.
The current face/PCA decoder is inappropriate for robot scenes. Count any new
vision encoder/codec, language teacher or renderer, including frozen weights.
Compare the same edges and data through SLWM and Transformer processors, plus a
simple direct-observation behavior-cloning policy. Retain a small model before
committing to GPT-2-scale multimodal capacity.

Freeze episode/environment/collector-session groups and training-only
normalization before fitting. Frame-random train/test splitting would measure
near-duplicate leakage. Measure one-step and increasing-horizon action-conditioned
forecasts, persistence and no-action predictors, shuffled-action controls,
previous-command imitation and goal sensitivity. A useful state must preserve
information about consequences that simple current observations cannot supply.
Offline imitation error is not closed-loop task success.

Test an actor on the frozen state against the same-capacity direct-history actor,
then compare short imagined rollouts and receding-horizon feedback in a bounded
simulator. Goals outside the trained/predicted horizon require explicit nearby
subgoals or a demonstrated longer rollout. Keep fast control separate from a
slower semantic/planning cycle; timestamped capture, bounded queues and stale
sensor rejection are part of that experiment, not established by file replay.

## Research connections and limits

[V-JEPA2](https://arxiv.org/abs/2506.09985) separates large video pretraining from
action-conditioned post-training on DROID. Its internet-video pretraining budget
is vastly larger than ours, so its robot results do not predict our small-model
performance. [V-JEPA2.1](https://arxiv.org/abs/2603.14482) studies dense temporal
features, deep self-supervision and joint image/video tokenization. These motivate
a comparison of temporally structured learned scene edges with pooled PCA, not a
claim that a particular extra signal branch is necessary.

[The Planning Limits of Latent World Models](https://arxiv.org/abs/2609.39235)
reports that action ranking degrades when goals lie beyond the imagined horizon,
even with much larger predictors in its tested settings. Treat that as a
hypothesis to test here: report horizon-dependent usefulness and feedback/subgoal
controls, not just next-frame loss or the duration of a rendered video. None of
these external results was replicated in the present speech/sensing batch.
