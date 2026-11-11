# G10 inference

These complete portable models use training seed 1003, the Fedora seed. The two paired models have 128-coordinate shared outputs, real natural-utterance and visual-motion inputs, and conditional decoders. The gated control is the matched variable-context model before post-training. Each package stores large matrices in four bits per value with per-output scaling. The inference runtime restores their parameters and executes the research model in JAX.

| Artifact | File size | Original / portable continuously instructed occupancy | Portable brief-command occupancy |
| --- | ---: | ---: | ---: |
| [Paired SLWM](align-hybrid-paired-s1003.npz) | 5,589,326 bytes | 94.3% / 94.1% | 34.5% |
| [Paired dense GPT-2](align-gpt2-paired-s1003.npz) | 5,574,621 bytes | 95.3% / 94.5% | 29.6% |
| [Gated recurrent control](context-gated-variable-s1003.npz) | 5,465,970 bytes | 94.2% / 94.1% | 34.9% |

The export check uses the first 128 missions from the independent command-confirmation cohort. The machine-readable receipt includes per-mission occupancies, model hashes and public-recording retrieval before and after packing. The complete study reports all training seeds. [Export receipt](portable.json)

Run from the repository root in an environment with JAX and NumPy. The live simulator issues actions and feeds their consequences back on a 10 Hz clock. Each run lasts 51.2 seconds.

```bash
study=experiments/multimodal/t10/context-alignment-20261111
weights="$study/inference/align-hybrid-paired-s1003.npz"
PYTHONPATH=. python scripts/infer/g10_world.py "$weights" \
  --mission --worlds 4 --changes 2 --instruction brief --output missions.json
```

The `latched` instruction option stores the last received word record where a request remains valid until superseded. The `brief` condition relies on the model's own state. `continuous` supplies the benchmark's persistent display. The discrete control vocabulary comprises hold, four directions and two tap polarities.

Condition on completed spoken-instruction features in the supplied public 60-second human task recording, then select the desired outputs.

```bash
PYTHONPATH=. python scripts/infer/g10_world.py "$weights" \
  --input "$study/examples/held-public-input.npz" --sources words \
  --emit vision audio state --output predictions.npz
```

Inputs share a 10 Hz clock. Audio contains 64 quantized spectral values, video has 16×16 RGB samples, physical input has 12 channels, and `x` holds 128 signed natural-language features. Optional `w` contains the original eight task/event word slots. Passing `--utterances file.json` accepts a list of objects with numeric `end` times in seconds and `text`; features become available after an utterance completes. Use `--sources audio`, `vision`, `physics` or `words`, individually or jointly. The real recording has no issued simulator control, so `u` defaults to hold. Candidate-conditioned predictions use the optional `plan` array. [Data source and output examples](../examples/README.md)
