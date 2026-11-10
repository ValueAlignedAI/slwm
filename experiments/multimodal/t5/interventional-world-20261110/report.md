# G5 joint world-state research

## What the shared state predicts

On new simulator scenes, the 25.73M-parameter hybrid predicted the consequences of changing an action more accurately than the equally sized dense GPT-2-style Transformer. On real recordings, dense attention gave the higher contact-timing score at both seeds. The study's joint architecture-win criterion is still unmet. These are development results. The terminal weight lock governs the separate test scenes and recording dates. [Protocol](protocol.md) · [Development results and paired intervals](metrics/medium-residual-dev.json)

| Development metric | Physical control | Dense, seeds 511 / 512 | Hybrid, seeds 511 / 512 |
| --- | ---: | ---: | ---: |
| Final block-center error after 16 issued actions, pixels ↓ | Fitted ridge 21.42 | 15.31 / 15.02 | 9.83 / 10.07 |
| Contact timing from video, average precision ↑ | Raw frame motion 0.0581 | 0.0635 / 0.0629 | 0.0597 / 0.0599 |

For the 256 simulator scenes, the paired 95% intervals for the hybrid's reduction in endpoint error relative to dense were 4.86–6.10 and 4.37–5.52 pixels. Each scene replayed identical history before executing three different plans. The real-video comparison covered 668 windows and 1,247 positive ticks on three independent recording dates. Both date-bootstrap intervals for the architecture difference span zero. Withholding future controls and altering hidden inputs changed earlier predictions by zero in the explicit leakage check. [Per-seed metrics](metrics/medium-residual-dev.json) · [Numerical integrity checks](metrics/integrity.json)

The first 1.04M-parameter models missed the zero-displacement baseline after 192 updates. Increasing to 2,048 updates left the best action error above 26 pixels. The successful larger run also changed the physical inputs and the objective. Both architectures began from the same train-only contact odds and causal action ridge, then learned a residual through a shared 128-dimensional state. This experiment attributes the hybrid–dense difference within the matched pair. Separating the effects of scale, the causal motion inputs and the explicit priors requires further ablation. [Initial screen](metrics/pilot-dev.json) · [Longer screen](metrics/exposure-dev.json) · [Residual registration](residual.md)

## Training exposure

The data pool contains 323,010 base windows or examples, 10,297 additional real impact windows from 832 training recordings, and 4,096 simulated histories with three branches each. The 323,010 base examples include 131,072 text windows, 118,272 COCO images, 50,193 LibriSpeech windows, 12,704 GRID utterances, 7,305 DROID windows, 2,024 AMI windows and 1,440 Clotho recordings. The LibriSpeech 360-hour download is outside these counts. COCO provides still images. Each training batch contains all nine families. [Sampler and registered mixture](../../../../configs/t5/joint_world.json)

Each medium model received 524,288 sampled examples, or 33,554,432 clock positions. Sampling allocated 21.875% each to real impacts and simulated physics, 9.375% each to GRID, LibriSpeech, COCO, DROID and text-only windows, 6.25% to Clotho and 3.125% to AMI. Audio, video, numeric state, and text can co-occur in one example. The family percentages therefore do not represent mutually exclusive shares of measured sensor channels. The byte-text stream uses a causal next-byte loss. [Batch implementation](../../../../training/g5_batches.py)

All decoders use the same learned state. The two explicitly calibrated physical priors bypass it, so beating those priors matters. G5 evaluates a domain-specific control improvement and real cross-modal timing. A claim of general modality-independent knowledge still requires transfer to unrelated domains, correct swap and missing-input controls, and decoded semantic quality.

## Sustained inference and scale

Six development artifacts each contain 1,024 ticks (43.69 seconds). They exercise video-to-audio, audio-to-video and text-to-both for both medium architectures at seed 511. Each layer carries its state across sixteen consecutive windows. Generated frames are 16×16 grayscale. The generated waveform uses a fixed Vocos decoder. A change under the written prompts measures sensitivity to text; the prompts have no ground-truth movie or sound. [Hybrid inference measurements](metrics/live-hybrid-511.json) · [Dense inference measurements](metrics/live-gpt2-511.json) · [Live protocol](live.md)

The medium development gate advanced both architectures to 116,502,025 parameters, 8,192 updates, batch 256 and two initialization seeds. That is 134,217,728 multimodal sequence positions per arm and seed. A position can be a video frame, signal packet or byte and cannot be counted as a language token. Full optimizer and inference states from all four medium runs were hash-verified on two independent Fedora HDDs. The trained evidence reported above ran on the RTX 3090. [Promotion decision and checkpoint hashes](metrics/medium-residual-decision.json)
