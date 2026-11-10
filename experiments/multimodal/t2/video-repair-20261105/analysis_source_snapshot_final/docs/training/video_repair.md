# T2-V1: controlled video contribution to speech repair

This cycle asks whether recorded lip motion improves a true 320ms speech gap
beyond additional audio-only capacity. It is offline supervised repair, with
future observed context. It is not free speech generation, novel-word synthesis,
a policy, or a demonstration of general world modeling.

- Protocol: [frozen experiment design](../research/video_repair_protocol_20261105.md).
- Decoder ceiling: [evaluation-only addendum](../research/video_repair_decoder_diagnostic_20261105.md).
- Config: `configs/t2/video_repair.json`.
- Results, best/last weights, optimizers, source snapshots, actual WAVs:
  `experiments/multimodal/t2/video-repair-20261105/`.
- Fedora data: `artifacts/t2_audio_visual/video_repair_v1/`.

## Architecture and accounting

The existing R2 content-preserving speech expert has 5,100,289 frozen weights.
An audio residual anchor learns the support-aware gap task. Stage B freezes both
and adds one correction branch. The primary full model has 7,990,547 weights;
1,423,025 belong to the new trainable CNN-video branch. It uses original 96×64
grayscale mouth frames plus differences, a four-stage CNN, temporal layers,
audio-query local cross-attention and an observed-context confidence gate.
Corrections touch only spectral windows intersecting the missing waveform.

The audio control replaces the CNN with an audio-field frontend. Common layers
and decoders have paired initialization, and graph-connected trainable counts
are within about 1%. It never reads video. Inactive gate/alignment weights are
reported separately rather than counted as a capacity match. PCA video is a
smaller ablation. Its CREMA features reuse the historical48×32 codec inputs;
new RAV features downsample the96×64 crop before applying that fixed PCA. This
resize chain makes it an approximate feature baseline. The doubled-update
audio run is explicitly a stronger
exposure control. Video negatives add training compute. This is neither a 124M
strict comparison nor a claim of architecture superiority at matched FLOPs.

Missing video in a visual checkpoint returns its frozen audio anchor exactly.
It does not automatically invoke the separately trained extra audio branch.
There is no learned policy selecting output modalities in this experiment.

## Genuine erasure and waveform preservation

At 16kHz, the centered 1024-sample Fourier window can see hidden samples even
when its center is outside a gap. Hide every overlapping coarse window,
including the first after the nominal gap. Interpolate only from observed
windows, then compute deltas. The primary score covers fine-window centers
inside the true gap; a separately reported halo loss covers overlapping edges.

Ordinary-file inference erases every missing sample before any transform,
computes one global feature sequence and uses a 1.28s context around each gap.
The phase decoder starts from observed-only phase and projects known waveform
samples back into every iteration. PCM16 output reproduces known samples
bit-for-bit. Hidden originals, reference phase and transcripts never enter any
model or practical decoder. The separately labeled oracle-spectrum diagnostic
uses reference spectra only to measure a decoder ceiling.

## Reproduce on the authorized Fedora machine

The default script expects `/home/db/dev/slwm`, the existing R2 checkpoints,
fixed CREMA codecs/data and `/home/db/venvs/vllm/bin/python`. Set `SLWM_ROOT`
and `SLWM_PYTHON` to use another installation. Torch, torchaudio, NumPy,
ffmpeg, Pillow, matplotlib, pytest and transformers are required. The pinned
Whisper-tiny.en evaluator is loaded from the existing local model cache.

```bash
scripts/train/run_video_repair_fedora.sh data
scripts/train/run_video_repair_fedora.sh train
scripts/train/run_video_repair_fedora.sh evaluate
```

Prepared manifests skip downloading. Source downloads use the official Zenodo
record with spaced byte-range requests, source rate-limit backoff, exact range
checks, member CRCs and raw SHA256. Whole-ZIP MD5 values are declared metadata,
not claimed full-archive verifications. CREMA and RAV train/validation/test
people are disjoint. Old R1 RAV test actors are explicitly validation here;
only the new eight final actors are fresh test people. RAV has two phrases and
CREMA a small fixed vocabulary. No codec is fitted on validation or test.
RAVDESS source data is CC BY-NC-SA 4.0; these are research artifacts.

The script freezes validation-only audio choices in `selection.json` before
reading final test scores, fixes32 listening IDs, snapshots evaluation sources,
scores320ms plus the declared160/480ms gaps, renders all conditions, runs a
fallible external recognizer, audits trained weights/inputs/PCM and writes
`report.md`, `summary.json` and `video_vs_audio.png`. Changing frozen training
sources requires a new registered experiment directory. Both best and final
weights are kept; best checkpoints also contain optimizer/RNG state.

## Use a checkpoint on ordinary files

From the repository root, with an audio recording and a synchronized video:

```bash
python -m exploration.video_repair_infer \
  --checkpoint experiments/multimodal/t2/video-repair-20261105/EXP-T2-614/best.pt \
  --codec-dir experiments/multimodal/t2/video-repair-20261105/bundle \
  --audio recording.wav --video recording.mp4 --video-domain cremad \
  --gaps 1.20:1.52 --device cpu --output /tmp/slwm-repair
```

`repaired.wav`, the predicted spectrum, missing-sample mask, hashes and
preservation audit are written to the destination. `--gaps` identifies absent
evidence even if the input file still contains its original samples. Multiple
short gaps may be comma-separated; each must be at most 480ms. An empty gap
string preserves the recording. Clips need at least 1.28s and are limited
to 5.12s; internal gaps and the fixed dataset crop geometry define the validated
scope. Use `ravdess` for that corpus's established crop. Arbitrary camera views,
edge gaps, noise, music and real-world packet-loss patterns remain unvalidated.

For an audio-only model, use the checkpoint listed under `audio` in
`selection.json` and omit `--video`. For an audio fallback inside a visual
checkpoint, omit `--video`; the frozen anchor is then used. No recognizer,
transcript or corpus target is needed for ordinary-file inference.

The experiment is deliberately small. [AV-HuBERT](https://arxiv.org/abs/2201.02184)
learns audio/visual hidden units through masked multimodal prediction;
[Montesinos et al.](https://www.isca-archive.org/interspeech_2023/montesinos23_interspeech.html)
use video and audio context for speech inpainting and investigate AV-HuBERT
visual features. This cycle imports neither model nor pretrained visual features.
Its result therefore tests this small correction architecture and data budget,
not the general limits of audiovisual speech repair.

## Retained execution and output revision2

The original renderer stopped on the first clip's short audio tail. Its log,
partial files and evaluation snapshot are retained. The separately hashed
revision2 repeats at most three terminal subframes from the already erased
recording, matching the existing codec edge rule. Training, feature scores,
selected IDs and gates are unchanged. For this retained artifact directory,
use `scripts/train/resume_video_repair_render_v2.sh` to reproduce corrected
rendering and reports. The original runner intentionally rejects changed
evaluation sources. Both snapshots and the correction record are preserved.

[Post-hoc crop inspection](../research/video_repair_crop_inspection_20261105.md)
limits interpretation of the central-rectangle occlusion diagnostic; it does
not consistently cover the entire lip region. This diagnostic is outside the
primary gate. Recorded video motion, rather than lip-specific causality, is the
capability supported by the CREMA controls.
