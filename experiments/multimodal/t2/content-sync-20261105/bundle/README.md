# Content and synchronization research bundle

These are small research models, validated on reserved GRID people and sentence
combinations. Read [the corrected results](../report_v2.md) and [all fixed audio comparisons](../listen.html)
before interpreting capability. The vocabulary is restricted; this is not a
general world model or production speech system.

- `audio.pt`: matched additional-audio candidate711.
- `video.pt`: video candidate713 plus its exact strong-audio fallback and frozen calibration.
- `content_base.pt`: content-supervised base702, with an explicit timed-word text interface.
- The two fixed NPZ codecs and training-only word vocabulary are included.

From the SLWM repository root:

```bash
python -m exploration.content_sync_infer --checkpoint BUNDLE/video.pt \
  --codec-dir BUNDLE --calibration BUNDLE/calibration.json \
  --media recorded_grid.mpg --gaps 1.32:1.64 --device cpu --output /tmp/slwm-repair
```

Replace `BUNDLE` with this directory. Use `audio.pt` for audio-only prediction;
`--audio recording.wav` instead of `--media` supplies an already synchronized
waveform. Original GRID media uses the recorded PTS alignment and fixed crop.
Observed PCM is preserved on that aligned timeline; initial trim/pad is logged.
With the base checkpoint, `--timed-text` explicitly supplies a JSON list of
`{start,end,word}` records. Practical audio/video inference receives no labels.
`--probe-words` requests a separate closed-vocabulary diagnostic. No recognizer
is needed for inference. Other camera views, open vocabulary and causal physical
dynamics remain unvalidated. Keep the source research-use permissions.
