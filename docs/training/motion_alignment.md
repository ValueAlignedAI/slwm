# Motion-aware alignment and diagnostic conditional outputs

This is the T2-M1 / X0-M1 follow-up to the 4.47M appearance-feature pilots.
The user authorized architecture changes, modest scaling, matching modalities,
and exploring generated outputs. Results live in
`experiments/multimodal/t2/motion-alignment-20261104/`.

## Architecture

Both versions train their neural components from scratch:

- Small: width 256, four signal blocks, **4,741,089** trainable parameters.
- Larger: width 384, six signal blocks, **15,232,225** trainable parameters.

Audio and visual temporal adapters project physical-time feature sequences into
one aligned `Z[B,T,D]` field. Local, spectral, long-convolution, attention and
gated feed-forward blocks process the field. Dedicated heads predict audio
log-mel/change features and mouth appearance/motion features. A comparison
baseline retains the older concatenated `Z[B,2T,D]` packing. Vanilla and
no-spectral arms use the same data and codecs, with measured parameter differences.

Independent audio/video alignment encodes the other stream as entirely absent
and omits absolute positions. Clip alignment excludes false negatives among
repeated sentence prompts; this limited use of labels is disclosed. The local
loss compares actual paired windows to shifted windows from the same recording.
There is no access to identity, transcript or emotion as model input. Whole
modalities are dropped during prediction training.

## Physical-time, decodable data

The source is pinned CREMA-D at `1658cd342dff90010aa843eaeebd53610a08b1dc`,
with original README/license retained. Selection contains 5,520 clips; 5,511
usable clips remain: 5,061 training, 210 validation, 240 test. Splits contain
62/7/8 disjoint people. All prior test people remain excluded from training;
the eight new confirmation people were unused in all earlier cycles.

Features preserve 25 Hz timestamps. The audio frontend computes 64-bin log-power
mel plus its first difference. The visual frontend crops a fixed lower-face ROI
and resizes to 48×32 grayscale; training-only PCA provides 96 appearance and
64 motion dimensions. PCA/statistical values are separately counted; there are
no frozen pretrained neural parameters. These features support **simultaneous
missing-modality translation**, not a causal future-prediction claim.

Fixed face crops restrict the supported framing. A generic arbitrary video
may have no mouth in this region. The small visual codec and estimated audio
phase impose quality limits independent of the world processor.

## Reproduction

The authorized GPU machine is `db@192.168.1.200`, directory `/home/db/dev/slwm`.
Its existing runtime is `/home/db/venvs/vllm/bin/python`; the separate research
environment supplies pytest/matplotlib. `ffmpeg` and `ffprobe` must be available.
The Fedora build uses VP9/WebM rather than unavailable libx264. Animated GIF
previews are also retained.

```bash
python -m training.t2_prepare_motion --config configs/t2/cremad_motion_prepare.json
python -m training.t2_motion_train --config configs/t2/motion_alignment_matrix.json
python -m evals.motion_causal_checks --config configs/t2/motion_alignment_matrix.json
python -m exploration.motion_outputs --config configs/t2/motion_alignment_matrix.json
python -m training.t2_motion_report --config configs/t2/motion_alignment_matrix.json
```

On Fedora, `bash scripts/train/run_motion_alignment_fedora.sh` runs the matrix
and subsequent analysis/probes/rendering after data preparation. Completed
matching configurations are reused. New experiments require new IDs/corpus
versions. Exact frozen training source, configs, optimizer/RNG state, hashes,
per-clip metrics and validation-selected checkpoints are saved. The X0 stage
records its own source and checkpoint hashes.

The registered ten-arm matrix compares small full/vanilla, legacy packing,
no local alignment, shuffled pairs, and larger full/vanilla in two seeds plus
a larger no-spectral arm. All receive 1,600 × 32 sampled clip windows of 32
frames (1.28 seconds). Validation alone selects checkpoint step. The eight
new test people do not tune architecture, budgets or thresholds.

## Give the trained model an input

Select a saved checkpoint and provide one observed modality. The default
`--head cross` exports only the other modality as an uncommitted diagnostic.

```bash
python -m exploration.motion_infer \
  --checkpoint experiments/multimodal/t2/motion-alignment-20261104/EXP-T2-306/best.pt \
  --audio /path/to/speech.wav \
  --output /path/to/audio_conditioned_output

python -m exploration.motion_infer \
  --checkpoint experiments/multimodal/t2/motion-alignment-20261104/EXP-T2-306/best.pt \
  --video /path/to/video.flv \
  --output /path/to/video_conditioned_output
```

Inputs must contain at least 1.28 seconds; the first 5.12 seconds are used.
`--device cpu` supports non-CUDA inference. Video conditioning reads video
frames only; any audio track is excluded. Audio conditioning reads audio only.
`--head audio`, `--head video` or explicit `--head all` provide diagnostic
reconstruction/head selection. This is not a learned policy/commitment claim.

Generated audio inverts predicted mel using newly estimated Griffin-Lim phase;
no original waveform or pretrained vocoder supplies generated speech. Mouth
appearance inverts the predicted PCA coefficients. A separate motion rendering
integrates predicted deltas on a training-mean face, explicitly a prior rather
than an inferred identity. Each output includes a provenance JSON and predicted
feature NPZ so the renderer can be inspected independently.

## Exploration and controls

The gallery includes audio→mouth, audio→motion, video→audio, same-modality
reconstruction, original media, codec round trips and no-input predictions.
Best/median/worst validation prediction cases and fixed test-person clips are
selected automatically. The larger model seed is selected by validation.

Closed-set linear probes are fit after freezing the core: 12 sentence prompts
and six emotion labels, separately for audio/video. Probe weights, class priors,
raw-feature and shuffled-label controls, validation-only temperature calibration,
test accuracy and ECE are saved. A recognized prompt is not free text generation
or general transcription. Probe parameters are counted separately.

Timing checks include same-recording shifted candidates, an untrained audio
energy/motion control, wrong clips from the same person, zeroed and reversed
audio, and an envelope-only audio corruption. Strong timing alignment can still
reflect speech energy/onsets rather than semantic understanding. Report the
alignment, generation and recognition results separately.

## Frozen-head diagnosis and speech quality

`exploration.motion_linear_decoders` is a separate post-hoc test: it freezes the
core and fits linear edge decoders on training frames with ridge=10. Original
heads and every registered metric remain unchanged. It adds 148,032/221,760
fitted coefficients for the small/large models and saves median-example outputs
with the `*_linear_*` prefix. It improves some same-modality reconstruction but
does not solve audio-conditioned motion.

The optional external Whisper-tiny.en recognizer only scores already-rendered
original, codec, model-reconstruction, generated and null waveforms. Its 37.76M
parameters are evaluation-only; they do not supply any generated audio or the
closed-set readout. The selected-gallery WER diagnostic is fallible, can exceed
100% through insertions, and does not establish human intelligibility.

All synthesized audio is peak-normalized for listening. Compare the supplied
original/codec/null controls; waveform loudness is not a calibrated prediction.
