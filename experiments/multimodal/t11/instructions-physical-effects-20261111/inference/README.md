# Replay the spoken-instruction intervention

The release contains complete four-bit copies of the three matched 13.25M shared-state parents and six compact G11 adapters. The small public packages use seed 1001 and the paired training condition. Select the adapter for the device fold. Fold A holds out coffee machines and computers, and fold B holds out printers and cameras. The receipt lists the hash of each original training generation, stored package sizes, and exact-versus-compact scores on every eligible held decision. [Verification receipt](portable.json)

From a repository checkout with JAX and NumPy, replay the supplied held case with the actual and counterfactual recorded instructions. The same 3.2-second sensor prefix is used in both commands.

```bash
study=experiments/multimodal/t11/instructions-physical-effects-20261111
PYTHONPATH=. python scripts/infer/g11_next.py "$study/inference" "$study/examples/coffee.npz" \
  --fold A --core hybrid --instruction correct --output correct.json
PYTHONPATH=. python scripts/infer/g11_next.py "$study/inference" "$study/examples/coffee.npz" \
  --fold A --core hybrid --instruction wrong --output wrong.json
```

Each output ranks next verbs, returns a target noun from the training vocabulary and a semantic noun candidate, and predicts the fractions of a local surface moving nearer and farther plus its mean absolute range change. The underlying physical measurement and attribution limits are documented in the study. `none` removes instruction audio and words. `acoustic` keeps frozen acoustic features, `asr_text` keeps only automatic words, `small` substitutes the stronger ASR transcript, and `human` uses the human-transcript diagnostic.

To exercise the microphone front end, supply a WAV of one completed English instruction and a local copy of Whisper tiny.en at the revision in the configuration. This optional route also uses PyTorch, Transformers, SciPy and soundfile. A human-transcript condition on a new WAV requires the matching `--transcript`.

```bash
PYTHONPATH=. python scripts/infer/g11_next.py "$study/inference" "$study/examples/coffee.npz" \
  --fold A --core hybrid --speech-wav "$study/examples/coffee-correct.wav" \
  --asr-dir "$whisper_tiny_en" --output waveform.json
```

The feature examples supply measured audio, image and IMU prefixes, the explicitly available prefix geometry, and separately packed true and earlier instruction features. Future correctness, verb and surface targets are stored outside the input archive. The 117M source and adapter artifacts are indexed in the full two-disk inventory. [Audible examples and recorded outcomes](../examples/README.md)
