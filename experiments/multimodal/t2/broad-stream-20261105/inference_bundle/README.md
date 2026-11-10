# Portable B1 world fields and sensing actors

These are inference exports of the validation-selected weights. Every neural tensor is checked against the full retained training state. Full optimizers/RNG, final states, curves and sources remain one directory above. manifest.json binds each actor to its exact exported world and records all codec/renderer hashes.

| World | Variant / seed | Trained neural weights | Selected update |
|---|---|---:|---:|
| EXP-T2-801 | slwm / 91 | 23,980,313 | 1800 |
| EXP-T2-802 | transformer / 91 | 23,805,209 | 1800 |
| EXP-T2-803 | slwm / 92 | 23,980,313 | 2000 |
| EXP-T2-804 | transformer / 92 | 23,805,209 | 1400 |
| EXP-T2-805 | late_fusion / 91 | 23,805,209 | 1800 |
| EXP-T2-806 | slwm / 91 | 23,980,313 | 2000 |
| EXP-T2-807 | slwm / 91 | 23,980,313 | 2000 |
| EXP-T2-808 | slwm / 91 | 23,980,313 | 2000 |
| EXP-T2-809 | slwm / 91 | 6,683,929 | 1800 |
| EXP-T2-810 | transformer / 91 | 6,596,377 | 1600 |

Worlds801/802/803/804 pair SLWM/Transformer at seeds91/92. Actors101–104 use their shared fields;111–114 use matched simple sensor statistics. The other worlds are registered controls, not test-selected winners. The world text edge is CMU pronunciation with train-estimated timing, not the separate125M general language model.

Run from the repository root with research-stream dependencies, matching PyTorch/Torchaudio and FFmpeg/FFprobe. Replace INPUT.wav/INPUT.mp4 with ordinary files. Defaults use CPU; add --device cuda for the recorded GPU path. Every output retains proposals, actual field text, state traces and media.

```bash
python -m exploration.broad_stream_infer --checkpoint experiments/multimodal/t2/broad-stream-20261105/inference_bundle/EXP-T2-801.pt --codec-dir experiments/multimodal/t2/broad-stream-20261105/inference_bundle --mode audio-to-video --audio INPUT.wav --seconds 40.96 --output /tmp/slwm_audio_to_video
python -m exploration.broad_stream_infer --checkpoint experiments/multimodal/t2/broad-stream-20261105/inference_bundle/EXP-T2-801.pt --codec-dir experiments/multimodal/t2/broad-stream-20261105/inference_bundle --mode video-to-audio --video INPUT.mp4 --seconds 40.96 --output /tmp/slwm_video_to_audio
python -m exploration.broad_stream_infer --checkpoint experiments/multimodal/t2/broad-stream-20261105/inference_bundle/EXP-T2-801.pt --codec-dir experiments/multimodal/t2/broad-stream-20261105/inference_bundle --mode text-to-av --text 'This passage supplies the words and estimated timing for the phonetic edge.' --min-text-seconds 40.96 --output /tmp/slwm_text_to_av
python -m exploration.stream_sensing_infer --world experiments/multimodal/t2/broad-stream-20261105/inference_bundle/EXP-T2-801.pt --actor experiments/multimodal/t2/broad-stream-20261105/inference_bundle/EXP-P1-101.pt --codec-dir experiments/multimodal/t2/broad-stream-20261105/inference_bundle --audio INPUT.wav --video INPUT.mp4 --goal both --pace --output /tmp/slwm_continuous_sensing
```

Audio-to-video keeps the supplied soundtrack and predicts64x64 face motion over a supplied training appearance. Video-to-audio generates sound with separately frozen13,531,650-weight Vocos; its CLI never decodes source sound. Text generation uses the provided words, with no reference timing. Stretching a short prompt to40.96s is an interface example, not evidence of naturally fluent long speech; the registered long passages and all failures are in ../watch.html.

The actor consumes at least42.325s of files for one bootstrap plus30 decisions. Paced replay waits for recorded chunk availability before processing it, makes decisions continuously without another user request and checks weight invariance. Features are prepared ahead of replay. These sensing actions do not control physical motion. Read ../report.md and the P1 report before interpreting outputs.

The model waits for1.365s chunks plus FFT support. Offline Vocos rendering additionally uses512ms future predicted mel context. This bundle does not establish live microphone transport, fast motor control, general scene video, factual language understanding or a converged joint125M multimodal model.
