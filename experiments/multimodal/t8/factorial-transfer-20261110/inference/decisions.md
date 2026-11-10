# Using a trained world checkpoint to choose an action

The pushing model predicts a physical outcome for each candidate control sequence. The selection rule executes the sequence predicted to bring the object closest to a supplied goal. Given the same state and different goals it can choose different actions. This interface assumes the planar simulator's observation and control convention.

```bash
python -m training.g8_decide \
  --checkpoint /mnt/slwm-checkpoints-a/SLWM/g8/g8-v5e-anchor-hybrid-511-other/step-016384 \
  --input experiments/multimodal/t8/factorial-transfer-20261110/examples/puck_barrier-input.npz
```

The command verifies `inference.npz` against `complete.json`, prints the chosen zero-based plan index, the forecast displacement and goal distance for every proposal, and the selected 16 waypoints in simulator pixels. A copied checkpoint directory works with the same command. Both Fedora checkpoint disks store the full model and optimizer.

| Input NPZ key | Shape | Units and meaning |
|---|---|---|
| `n` | 16 × 6 | Observed pusher x/y, object x/y, sine and cosine of object angle. Each position is scaled as coordinate / 256 − 1. |
| `v` | 16 × 256 | Past grayscale camera, uint8, 16×16 pixels flattened per tick. |
| `plan` | candidates × 16 × 2 | Proposed pusher x/y waypoints at future ticks, in the same normalized position units. |
| `goal_displacement_px` | 2 | Desired object displacement from its last observed position, in simulator pixels. |

Use `--source state` to evaluate proprioception without camera input, or `--source video` to use the camera alone. The unselected sensor array can be omitted. Both sources are required in the default joint mode.

The input file contains observations, proposed controls and the goal. The measured future is used only afterward to check the choice. The first reserved start in each task is selected in advance. Compare the two new non-simulator-only checkpoints, their issued controls and measured regret in the [decision example receipt](../metrics/decision-examples.json).
