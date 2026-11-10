# Released acoustic weights

The 651,696-byte [allocated acoustic codebook](allocated-acoustic.npz) contains the train-fitted PCA, scalar quantization centers and fixed bit allocations. Its SHA-256 is `e4e3f423d10bb5e099da05d042a20ee80cc4e58bcb91cab128113291b6096cb0`. Encoding mixes input channels to mono and converts to 24 kHz. Decoding produces mono PCM at 24 kHz and preserves the encoded sample count. The model works with Python 3.11+ and the repository's audio research dependencies.

From the repository root, encode a recording and decode its generated bitstream using the separately released Vocos waveform renderer.

```bash
python -m training.g8_audio_codec encode input.wav recording.slwma --bits-per-pair 48
python -m training.g8_audio_codec decode recording.slwma reconstructed.wav --vocoder /path/to/vocos-mel-24khz
```

The allowed bit counts per frame pair are 6, 24, 48, 96 and 192. Multiply by 46.875 to obtain bits per second. Encoding uses the local trained artifact by default. An explicit `--weights` path works for both commands. The decoder verifies the stream's declared model and payload hashes.

The pinned external Vocos revision is `0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21`. The model directory contains `config.yaml` and `pytorch_model.bin`. The released acoustic codebook models the input spectrum; the externally trained Vocos model supplies waveform synthesis. [Pinned renderer](https://huggingface.co/charactr/vocos-mel-24khz/tree/0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21) · [Measured quality and example bitstreams](../speech.md)
