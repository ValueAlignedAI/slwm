# Run the spoken-goal decoder

From the repository root, use the verified speech features and the public station map packaged with the original recording.

```sh
PYTHONPATH=. python scripts/infer/g14_live.py --speech-only
```

The original recording contains “bin blue at i six please.” The decoded object is blue, its letter and digit are I and 6, and its station is index 0 in this particular map. The prepared route needs NumPy and JAX. To encode the waveform yourself, install PyTorch and Transformers and supply the local pinned Whisper-small.en snapshot.

```sh
PYTHONPATH=. python scripts/infer/g14_live.py --speech-only \
  --wav experiments/multimodal/t14/spoken-closed-loop-20261111/inference/spoken-goal.wav \
  --encoder /path/to/whisper-small.en --device cpu
```

The source example is speaker 33, clip `bbai6p`. Martin Cooke, Jon Barker, Stuart Cunningham and Xu Shao released *The Grid Audio-Visual Speech Corpus* under CC BY 4.0. This file contains mono 16 kHz audio aligned to the start of their recorded video. The inference manifest gives SHA-256 for every model and example. Model revision `e8727524f962ee844a7319d92be39ac1bd25655a` supplies the shared pretrained Whisper edge. [Corpus](https://doi.org/10.5281/zenodo.3625687) · [License](https://creativecommons.org/licenses/by/4.0/) · [Encoder](https://huggingface.co/openai/whisper-small.en/tree/e8727524f962ee844a7319d92be39ac1bd25655a) · [CPU and CUDA replay](../metrics/speech-replay.json)

## Execute a physical decision

The package contains three matched dense and three spatial checkpoints, the calibrated raw physical predictor, a four-observation camera/telemetry prefix and two alternative recordings from the same person. The inputs fit on a small machine. Prepared inference needs no download of the external speech encoder.

```sh
PYTHONPATH=. python scripts/infer/g14_live.py --core transport --inspect
PYTHONPATH=. python scripts/infer/g14_live.py --core gpt2 --seed 1412
PYTHONPATH=. python scripts/infer/g14_live.py --text "bin blue at i six please"
PYTHONPATH=. python scripts/infer/g14_live.py --absent
PYTHONPATH=. python scripts/infer/g14_live.py --features experiments/multimodal/t14/spoken-closed-loop-20261111/inference/other-address-features.npz
```

On this fixed camera prefix, the source recording requests blue at I6 and yields a 43.4 mm by 28.6 mm simulated command offset. The same speaker requesting blue at Y7 changes that command by 37.4 mm. The supplied written I6 command reproduces the original field and action. The frozen confidence gate declines the other-block example. All six physical weights produce the same verified decisions from the public package and the training archive. The optional inspection exposes the physical predictions and the 128-coordinate field even when the geometric approach controls that tick. [Model and signal checks](../metrics/field-replay.json)
