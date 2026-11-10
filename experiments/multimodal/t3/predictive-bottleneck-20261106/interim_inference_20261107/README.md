# Interim shared predictive-state inference

Four124M-scale checkpoints, each trained for512 optimizer updates. These are early research artifacts, not completed12k runs or validated general-world/decision models. The fixed example is the first selection case; final test observations remain unopened. Predictions condition on supplied recorded future commands. No trained action head exists yet. The preview retains poor outputs.

`manifest.json` records exact SHA256, parameter counts and real inference checks. Each folder contains `world.pt`, numeric/state outputs and three-view future RGB arrays. `observations.npz` and `future.npz` are the normalized inference inputs; targets are separate in `reference_not_input.npz`.

From `source_snapshot/`, with recorded Torch/NumPy/SciPy/tiktoken dependencies:

```bash
python -m exploration.predictive_infer --checkpoint ../G2-201/world.pt --observations ../observations.npz --future ../future.npz --output /tmp/shared_prediction.npz --device cpu
```

The core returns512D state and normalized numeric forecasts. Image forecasts have native16x16 RGB resolution. This is not free-running video generation. All four runs use the same example; no cases were selected for output quality.

Source attribution: DROID (Khazatsky et al.,2024), official public DROID1.0.1 RLDS release at https://droid-dataset.github.io/ and the pinned gresearch/robotics/droid/1.0.1 source manifest. The displayed reference images are low-resolution crops from the fixed public selection episode; prediction columns are model outputs. No private user document or image is included.
