# Run G12 inference

The package contains the lossless float32 weights of 24 trained models. All four processors are available for each held-device split in three configurations. Width 128 and width 256 use the primary training schedule. The early-joint release chooses between full training inputs and modality dropout using the seed-1201 development criterion. Every output head reads the shared 128-dimensional field. A 40%-development-coverage threshold is calibrated separately for action and device-state proposals. [Model specifications, source checkpoint hashes and thresholds](manifest.json) · [Numerical parity under true, wrong and absent speech](../metrics/inference-checks.json)

From the repository root with Python 3.12, NumPy and JAX installed, the following command processes the camera stream one observed clock at a time.

```bash
g12=experiments/multimodal/t12/instruction-sequences-20261111
python -m scripts.infer.g12_live "$g12/examples/camera-stream.npz" --fold B --core associative
```

Use `--core gpt2`, `--core gated`, or `--core oscillator` for a matched comparison. Use `--width 256` for the wider processor or `--recipe early` for the early-joint model. The camera and printer streams use fold B. Computer and coffee use fold A. Compare `--mode correct`, `--mode wrong` and `--mode none` on the same input. The emitted JSON has a chronological source time and a route. Only accepted outputs are decoded. A no-op appears as `internal_only`. The default batch size is one. Larger batches can replay precomputed causal windows faster. [Recorded examples](../examples/README.md)

To replace a completed instruction with a waveform, install PyTorch and Transformers, make FFmpeg available, and provide the local Whisper-small.en snapshot at revision `e8727524f962ee844a7319d92be39ac1bd25655a`. Set `WHISPER_SMALL_EN` to its directory.

```bash
python -m scripts.infer.g12_live "$g12/examples/camera-stream.npz" --fold B --core associative \
  --event 142 --wav "$g12/examples/camera-foil.wav" --whisper "$WHISPER_SMALL_EN"
```

The audio override recomputes acoustics and recognized words. The normal stream supplies causal visual, acoustic and IMU features and completed speech. This interface outputs forecasts for offline operational replay. Uncertainty, coverage and the tested spans are reported with the results. [Waveform checks](../metrics/raw-audio.json) · [Study report](../report.md)
