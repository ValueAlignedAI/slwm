# SLWM true-gap repair research bundle

Video fails the registered contribution gate; retain trained audio controls and report the negative or mixed result.

- [Audio control](audio_seed61.pt): EXP-T2-619, validation-selected.
- [Video correction model](video_seed61.pt): EXP-T2-614, fixed primary seed61.
- [Results](../report.md), [failure analysis](../failure_analysis.md), [all32 listening cases](../listen.html).
- [Hashes and checkpoint identities](manifest.json).

Both files include all frozen and trained neural weights and their configuration.
The two NPZ codecs are the unchanged CREMA-training statistics/PCA. In the SLWM
repository, ordinary-file CPU inference needs no dataset or recognizer:

```bash
python -m exploration.video_repair_infer --checkpoint experiments/multimodal/t2/video-repair-20261105/bundle/video_seed61.pt \
  --codec-dir experiments/multimodal/t2/video-repair-20261105/bundle --audio recording.wav --video recording.mp4 \
  --gaps 1.20:1.52 --device cpu --output /tmp/slwm-repair
```

For audio only, choose `audio_seed61.pt` and omit `--video`. Omitting video from
the visual checkpoint invokes its frozen anchor, not the stronger audio control.
Outputs preserve observed PCM samples exactly and predict only short erased gaps.
Use the fixed CREMA/RAV crop geometry; arbitrary camera views are unvalidated.
Clips are limited to5.12s, need1.28s context and gaps at most480ms. These remain
research artifacts; RAVDESS source data is CC BY-NC-SA4.0.
