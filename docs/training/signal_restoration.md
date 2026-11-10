# Shared-field content conservation and packet restoration — T2-R1/R2

Read the [findings](../research/findings.md),
[R1 protocol](../research/signal_restoration_protocol_20261105.md), and
[R2 protocol](../research/content_conservation_protocol_20261105.md).
These are simultaneous signal experiments and researcher-selected diagnostic
outputs. Policy, free text and autonomous future dynamics are not trained.

`RestorationSLWM` removes the old unused reconstruction heads, reuses the M1
adapters/processor and preserves observed sound with an explicit interpolation
path. Missing four-packet spans are filled only from observed neighbors, then
audio differences are computed. No hidden target reaches any input. A learned
correction is bounded to ±.25 normalized units in observed packets, ±3 in hidden
ones. Core and raw baselines share the same 32-packet learned context. No wider
frozen field cache is used.

`ContentRestorationSLWM` adds a training-only 256→64 linear readout, 16,448
weights. Its observed-packet MSE supplies a conservation gradient even when the
raw bypass already protects the output. The primary .5 objective passes both
seeds' preregistered representation/output/timing gates on the harder new set.
Useful positive contribution from video remains failed; this is not solved
speech generation. Independent linear probes see only audio-only world fields,
not the raw bypass. They measure decodability of observed sound, not semantics.

## Data and reproduction

Use the unchanged 5,061/210/240 CREMA split, plus 1,278 LibriSpeech dev-clean
training clips from 32 people and 64 validation clips from eight other people.
The training mixture is 24 CREMA + 8 Libri windows, with half the CREMA video
inputs dropped. All encoding/PCA normalization stays CREMA-training-only.
R1 first reserves 48 test-clean clips/12 people and 128 new RAVDESS clips/eight
people. R2 responds to those results, explicitly reuses them diagnostically,
and reserves 48 test-other clips/12 different people as a fresh harder check.
No external fitting or checkpoint selection occurs.

LibriSpeech official archives are verified by MD5; individual files have SHA256.
CC BY 4.0, Panayotov et al. (2015), https://www.openslr.org/12. RAVDESS selected
ZIP members have CRC32 and SHA256, with declared whole-archive MD5 recorded
rather than recomputed. CC BY-NC-SA 4.0, Livingstone & Russo (2018),
https://zenodo.org/records/1188976, held-out noncommercial evaluation only.

The authorized Fedora runtime is `/home/db/venvs/vllm/bin/python`; PyTorch,
torchaudio, NumPy, Pillow, Transformers and ffmpeg are installed. The isolated
`.research-venv` site-packages provide pytest/matplotlib. No system packages were
changed. Full source/config/data/checkpoint hashes and runtime versions are in
the final audit.

```bash
cd /home/db/dev/slwm
bash scripts/train/run_signal_restoration_fedora.sh
bash scripts/train/run_content_conservation_fedora.sh
python -m evals.restoration_silence_control --config configs/t2/signal_restoration.json
python -m training.restoration_report \
  --r1 configs/t2/signal_restoration.json --r2 configs/t2/content_conservation.json
```

Use the runtime's Python, or `SLWM_PYTHON` for the shell runners. Existing source
snapshots/corpus hashes are immutable; change recipes under a new version.
Raw media and source archives remain on Fedora. Compact feature arrays, all
checkpoints, codec statistics, reports and actual output media are mirrored locally.

## Ordinary-file inference

Run from the repository root. CPU is default; `--device cuda` uses the GPU.
Audio must be at least 1.28s; at most 5.12s are observed. Missing intervals are
declared explicitly, or supplied as a boolean `.npy` flag per 25Hz packet.

```bash
python -m exploration.restoration_infer \
  --checkpoint experiments/multimodal/t2/content-conservation-20261105/EXP-T2-521/best.pt \
  --audio /absolute/path/to/recording.wav --gaps 0.4:0.56,1.2:1.36 \
  --output /absolute/path/to/result
```

Add `--video /absolute/path/to/video.mp4 --video-domain ravdess` for synchronized
mouth context, or `--video-domain cremad`. These are fixed framing assumptions,
not a general face tracker. Some sound must be observed; this is not a trained
video-only speech generator. `--visibility /absolute/path/to/mask.npy` replays a
saved corruption. With no gaps or mask, the command reconstructs observed sound.

Outputs: actual `restored_audio.wav`, `predicted_mel.npy`, `visible_mask.npy` and
input/checkpoint hashes in `inference.json`. No source waveform samples are
inserted, no pretrained generator supplies sound, and inference never updates
weights. A 64ms centered mel window can straddle packet boundaries: this is not
sample-exact waveform inpainting. Do not infer practical intelligibility from
feature MSE; compare original/codec, interpolation and observation-matched silence.
Whisper-tiny.en is a pinned, fallible evaluator only; its text is not SLWM text.

## Retained evidence

`signal-restoration-20261105` retains nine runs; `content-conservation-20261105`
retains four. Every arm has selected optimizer-bearing weights, last weights,
validation curves, complete held-out errors and person-cluster intervals, exact
counts, initial/final source snapshots and hashes. Field-only ridge fits use
training data only. The zero-auxiliary control reproduces the prior primary
model. Runtime tests include actual CPU audio/video files, hidden-packet
invariance and pixel-identical videos with different soundtracks.
