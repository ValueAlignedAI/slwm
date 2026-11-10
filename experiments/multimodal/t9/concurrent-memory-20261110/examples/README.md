# G9 inference examples

The action replay runs for 51.2 seconds on the held intermittent-stall dynamics. It shows the first registered mission for the 116.70M, 128-coordinate seed-912 pair. A black marker moves under the issued actions. The physical reference recovers calibration from the same observable tap response. [Action replay](held-live-control.mp4) · [Trace](live-trace.json) · [Full synchronized simulated sensor stream](sensor-stream.npz)

The real recordings below use the same held Sony recording, the first 40 seconds, the same word or sensor stream and the two matched large 512-coordinate seed-911 models. Every output is decoded from the emitted shared field. Audio consists of reconstructed band envelopes rendered with deterministic carriers. The native visual output is 16×16, enlarged for viewing. Speech intelligibility and scene-level realism are outside the demonstrated capabilities.

| Conditioning and output | SLWM | Dense GPT-2 |
| --- | --- | --- |
| Observed audio → predicted RGB | [Play 40 seconds](confirm-hybrid-z512-gelu-s911-fresh-audio_only.mp4) | [Play 40 seconds](confirm-gpt2-z512-gelu-s911-fresh-audio_only.mp4) |
| Observed video → predicted audio envelope | [Listen](confirm-hybrid-z512-gelu-s911-fresh-video_only.wav) | [Listen](confirm-gpt2-z512-gelu-s911-fresh-video_only.wav) |
| Event words → predicted RGB and audio envelope | [Play 40 seconds](confirm-hybrid-z512-gelu-s911-fresh-text_only.mp4) | [Play 40 seconds](confirm-gpt2-z512-gelu-s911-fresh-text_only.mp4) |

The accompanying per-model metadata identifies recording, duration, input mask and parent SHA-256. The room-level scores compare these conditions with a global training prior and wrong-time evidence within the same recording. [STARSS23 source](https://zenodo.org/records/7880637)
