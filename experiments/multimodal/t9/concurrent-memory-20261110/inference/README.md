# Run the two trained G9 models

These are complete, portable copies of the 13.25M, four-layer, 128-coordinate models trained for 4,096 updates with seed 911. Both use GELU and were selected by configuration. Per-output-channel four-bit packing applies to large matrices. Smaller tensors use float16. The runtime reconstructs float32 parameters and runs the original JAX inference path. This changes storage size and can change decisions.

| Model | Package bytes | Original goal occupancy | Portable goal occupancy |
| --- | ---: | ---: | ---: |
| SLWM | 5,483,128 | 94.54% | 93.93% |
| Dense GPT-2 | 5,480,699 | 43.54% | 43.88% |

The comparison uses 128 held, 51.2-second missions on native TPU. The packaged model and command-line entry point also passed sustained CPU inference on Fedora. All 128 per-mission scores and both weight hashes are in the receipt. [Portable SLWM](exposure-hybrid-z128-gelu-s911-fresh.npz) · [Portable dense GPT-2](exposure-gpt2-z128-gelu-s911-fresh.npz) · [Verification receipt](portable.json)

From a repository checkout with JAX and NumPy, run a live mission. The issued control changes the next simulator state and sensor observations. The output records 512 successive decisions for each requested world.

```bash
study=experiments/multimodal/t9/concurrent-memory-20261110
weights="$study/inference/exposure-hybrid-z128-gelu-s911-fresh.npz"
PYTHONPATH=. python scripts/infer/g9_world.py "$weights" --mission --worlds 4 --output missions.json
```

Run cross-modal inference on the supplied 51.2-second synchronized sensor stream. The example below exposes audio and goal words. The output includes emitted states, RGB forecasts of the hidden camera input, physical estimates, event logits, action logits and candidate-conditioned next audio and physical measurements.

```bash
PYTHONPATH=. python scripts/infer/g9_world.py "$weights" \
  --input "$study/examples/sensor-stream.npz" --sources audio words --output predictions.npz
```

At 10 Hz, `u` holds the previously issued primitive at each clock tick, with 0 for hold, 1–4 for directional controls and 5–6 for the two tap polarities. Optional `plan` contains the candidate control for one-step forecasting and defaults to hold. Supply selected source arrays on the same clock. Audio `a` has 64 quantized band values, `v` has 768 quantized 16×16 RGB values, `n` has 12 physical channels and `w` has eight small-vocabulary word slots. Only named sources enter the encoder. Text consists of task commands and event labels. [Sensor conventions](../../../../../models/g9_tabletop.py) · [Real signal preparation](../../../../../scripts/data/g9_recordings.py)

The unquantized training stages on Fedora use `/mnt/slwm-checkpoints-a/SLWM/g9/runs/<tag>/step-04096` and the same path on the second checkpoint disk. Pass the stage directory in place of the portable file. Add `--head <path-to-parent-head.npz>` or `--head <path-to-random-head.npz>` for the four corresponding large frozen-parent experiments. The loader checks the head and parent identities. [Adapted heads](../metrics/followups/heads)
