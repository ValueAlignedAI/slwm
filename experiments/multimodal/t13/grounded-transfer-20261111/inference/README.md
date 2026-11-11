# Run a trained G13 model

The public package has ten robot fields, sixteen open-vocabulary component models, the portable four-word acoustic encoder, and the fitted physical, action and voice readouts. Parameters are compressed losslessly. Inference verifies file hashes. The robot fields share the same 700,301 trainable parameters per core. Four physical context clocks and five slots feed the processor. Internal proposals, uncertainty and policy routing all use the shared state. [Manifest and thresholds](manifest.json) · [Verified fields](../metrics/world-parity.json)

From the repository root use Python 3.12, NumPy and JAX. Prepared examples include the frozen causal signals. These three commands respectively route an ordinary forecast, propose the directly trained net action and rank hypothetical controls for a desired displacement in metres.

```bash
g13=experiments/multimodal/t13/grounded-transfer-20261111
python -m scripts.infer.g13_world "$g13/examples/robot-stream-1.npz" --event 10
python -m scripts.infer.g13_world "$g13/examples/robot-stream-1.npz" --event 10 --direct-action
python -m scripts.infer.g13_world "$g13/examples/robot-stream-1.npz" --event 10 --target 0.055 0
```

Omit `--event` to process every clock. The ordinary forecast emits accepted object/physical outputs or `internal_only`. Plan selection can return `no_op`. The numerical direct-action diagnostic reports the learned four-control sum and its raw reference. To supply a candidate control sequence, use `--plan '[[0.01,0],[0.01,0],[0.01,0],[0.01,0]]'`. The passive action and moved-object proposals are invariant to that substitution. Use `--model gpt2-linear` for the matched dense model or `--model transport-linear` for the spatial model. The physical examples are recorded history; the commands print forecasts and never control hardware.

To hear an actual held speaker and predict a recorded physical direction, the following example replaces the cardinal words in the written context with a real microphone waveform. The default target is effector direction. Use `--direction-target object` for a block-motion readout.

```bash
python -m scripts.infer.g13_world "$g13/examples/voice-left.npz" --event 0 \
  --model transport-anchored --instruction "point your arm parallel to the blue triangle" \
  --wav "$g13/examples/voice-left-true.wav"
```

The natural HoloAssist experiment has a separate open-vocabulary processor. The public noun inventory has 165 entries. The default command decodes accepted outputs. Use `--inspect` to see diagnostic top-five proposals and `--mode wrong` or `none` to compare the same observed scene.

```bash
python -m scripts.infer.g13_component "$g13/examples/component-A.npz" --fold A --inspect
```

For an original waveform override, install PyTorch, Transformers and FFmpeg and provide local copies of Whisper-small.en at `e8727524f962ee844a7319d92be39ac1bd25655a` and all-MiniLM-L6-v2 at `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The native path computes actual acoustics and recognized words, then sends the latter through the frozen semantic encoder.

```bash
python -m scripts.infer.g13_component "$g13/examples/component-A.npz" --fold A \
  --wav "$g13/examples/component-A-true.wav" --whisper "$WHISPER_SMALL_EN" \
  --minilm "$MINILM_L6" --inspect
```

The packed component models reproduced all 48 prepared top-proposal checks. The four original-waveform substitutions reproduced the cached gate route in all four cases and the same top diagnostic noun in three. The single-clip recognition path can produce a slightly different transcript. [Native-waveform evidence](../metrics/natural-waveforms.json) · [Recorded fixtures](../examples/README.md)
