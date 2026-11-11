# Sixty seconds of a recorded physical task

Both paired seed-1003 models run continuously on the same HoloAssist espresso-machine session from a held task family. Each decoder receives only the declared source through the shared emitted field. The native camera predictions have 16×16 RGB samples and are enlarged for viewing. The audio renderer assigns deterministic carriers to predicted spectral envelopes. These are signal predictions with no demonstrated spoken-word intelligibility.

| Source and selected output | SLWM | Dense GPT-2 |
| --- | --- | --- |
| Audio → RGB | [Play 60 seconds](align-hybrid-paired-s1003-audio.mp4) | [Play 60 seconds](align-gpt2-paired-s1003-audio.mp4) |
| Camera → audio envelope | [Listen](align-hybrid-paired-s1003-vision.wav) | [Listen](align-gpt2-paired-s1003-vision.wav) |
| Completed speech text → RGB and audio envelope | [Play 60 seconds](align-hybrid-paired-s1003-words.mp4) | [Play 60 seconds](align-gpt2-paired-s1003-words.mp4) |

The session was selected by a deterministic ID hash among held sessions whose first minute contained completed speech. The input is a derived feature excerpt from `z031-june-22-22-espresso`. The archive stores the actual 10 Hz camera, acoustic, motion and language inputs, ready for the public inference command. It excludes raw waveforms and verbatim conversations. [Input features](held-public-input.npz) · [Example provenance](source.json) · [Inference commands](../inference/README.md)

Source data are from HoloAssist by Wang and colleagues, ICCV 2023, distributed under CDLA-Permissive-2.0. The agreement accompanies this derived input. [HoloAssist](https://holoassist.github.io/) · [Data agreement](HoloAssist-LICENSE.txt)
