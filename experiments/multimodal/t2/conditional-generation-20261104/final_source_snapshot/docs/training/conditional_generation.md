# Conditional generation research — T2-G1/G2

These completed, isolated experiments train diagnostic output heads, not policy
commitment or autonomous dynamics. See [findings](../research/findings.md),
[G1 protocol](../research/conditional_generation_protocol_20261104.md) and
[G2 protocol](../research/conditional_residual_followup_20261104.md).

G1 freezes the 4,741,089-weight EXP-T2-301 core with independently encoded audio
and video. Five 548,352-weight audio heads predict four 100 Hz mel frames per
25 Hz packet. Six 213,699-weight visual heads predict 32-frame grayscale mouth
sequences from sound and ONE actual seed frame. No later image is an input.

G2 adds a fixed interpolation of observed audio plus a zero-initialized correction
bounded to ±0.25 normalized mel units. Raw and frozen-core corrections have
identical 548,352 trained weights; core arms add 4,741,089 frozen weights.
Deterministic interpolation has zero neural weights. Raw residual heads
EXP-T2-421/423 pass the two-seed pilot gate; core corrections fail externally.
This preserves observed sound, not novel speech from video.
Cached fields overlap 32-frame encodes across each observed recording, and can
include context outside a decoder crop; raw corrections see the crop directly.
Head/data/update budgets match, but core arms have extra context and compute.

## Data and reproduction

Retain CREMA-D's 5,061/210/240 clips with disjoint people. The test is reused.
Fit all new 100 Hz normalization only on training people. Raw media and old
codecs remain unchanged. The G1 external check is 64 RAVDESS full-AV speech files
from actors 01,02,13,14, normal intensity, repetition 01, both statements and all
emotions. It never fits statistics or weights or selects checkpoints. G2 explicitly
reuses it after observing G1's failure. The fixed y=.54h crop was selected from
framing before scores. Raw SHA256, ZIP CRC, source metadata and CC BY-NC-SA 4.0
license (Livingstone & Russo, 2018) are retained; noncommercial evaluation only.

The tested Fedora environment is `/home/db/venvs/vllm/bin/python`, with torch,
torchaudio, NumPy, Pillow, Transformers and ffmpeg with VP9/WebM. Matplotlib and
pytest use the isolated `.research-venv` site-packages. No system packages changed.
Generated sound uses inverse mel and newly estimated Griffin-Lim phase; no
pretrained generator or recorded phase supplies output.

```bash
cd /home/db/dev/slwm
bash scripts/train/run_conditional_generation_fedora.sh
python -m training.t2_generation_residual --config configs/t2/conditional_generation_residual.json
python -m training.t2_generation_residual --config configs/t2/conditional_generation_residual.json --evaluate
python -m exploration.generation_residual_outputs --config configs/t2/conditional_generation_residual.json
python -m training.t2_generation_residual_report --config configs/t2/conditional_generation_residual.json
```

Use the environment's Python (`SLWM_PYTHON` overrides the shell runner). Source
and corpus hashes are immutable; change a recipe under a new run/corpus version.
Do not overwrite checkpoints after inspecting test scores.

## Standalone inference

From the repo root, CPU is the default; `--device cuda` uses the GPU. All new
checkpoints, feature arrays and codecs are mirrored locally. Raw media stay on
Fedora. Supply ordinary audio/video files; no training metadata is required.

```bash
# Useful small edge: reconstruct observed sound.
python -m exploration.generation_infer \
  --checkpoint experiments/multimodal/t2/conditional-residual-20261104/EXP-T2-421/best.pt \
  --audio /absolute/path/to/recording.wav --output /absolute/path/to/audio_result

# Diagnostic video->audio: speech remains unreliable.
python -m exploration.generation_infer \
  --checkpoint experiments/multimodal/t2/conditional-generation-20261104/EXP-T2-402/best.pt \
  --video /absolute/path/to/clip.mp4 --video-domain ravdess \
  --output /absolute/path/to/video_result

# Diagnostic sound + ONE actual mouth image -> 1.28 seconds of frames.
python -m exploration.generation_infer \
  --checkpoint experiments/multimodal/t2/conditional-generation-20261104/EXP-T2-411/best.pt \
  --audio /absolute/path/to/recording.wav --seed-image /absolute/path/to/mouth.png \
  --start-frame 0 --output /absolute/path/to/motion_result
```

Inputs must contain at least 1.28 seconds; at most 5.12 seconds are observed.
The seed is a pre-cropped mouth image, resized to 48×32, corresponding to the
requested sound start time. Crop choices assume fixed CREMA/RAVDESS framing,
not general face tracking. Outputs are WAV, predicted mel arrays, or GIF/WebM
and frame arrays, plus input/checkpoint/seed hashes in `inference.json`.
Inference never updates weights. Remuxing containers can change video timestamps;
isolation checks compare identical decoded frames/timestamps with different sound.

Each cycle's `experiments/multimodal/t2/conditional-{generation,residual}-20261104`
directory retains reports, source/config/data/checkpoint hashes, every checkpoint,
curves, registries and complete per-clip metrics. Listening diagnostics use 32
fixed CREMA and 16 fixed external cases with original/codec/model/interpolation/
wrong/null controls. Whisper-tiny.en (37,760,256 pinned evaluator-only weights)
is fallible; its text is not SLWM output. All 104 tests pass, plus actual-file CPU
checks for reconstruction, video-only inputs and audio-plus-seed motion.

An explicit EXP-T2-415 readout correction disables the shuffled-training flag at
inference, without retraining or reselection. Use `corrected_*_metrics.json` and
`EXP-T2-415_correct_audio.gif` for that arm. Original metrics/media and derived
report are preserved for audit. The shell runner applies this correction.
