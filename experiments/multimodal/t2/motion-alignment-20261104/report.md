# SLWM motion alignment, modest scaling and conditional outputs — 2026-11-04

## Research decision

Physical-time alignment passes both larger-model seeds and the input-corruption checks. Cross-modal generation remains incomplete; do not treat playable output as a useful general world model.

Completed **10** registered RTX 3090 runs, following the user's authorization to redesign the previous failed recipe and explore audio/video-conditioned outputs. The small model has 4,741,089 trainable parameters; the larger model 15,232,225. No pretrained neural encoder or external media generator supplies the results.

## What changed

A fixed lower-face codec retains appearance and frame-to-frame motion; 25 Hz physical timestamps replace normalized clip time. Audio and visual temporal adapters enter a shared field at corresponding times. Independent alignment removes the other modality and omits absolute positions. A same-recording shifted-window objective explicitly trains synchronization. Missing-modality prediction trains the two output directions. The legacy packing and no-local-alignment arms isolate these changes, while vanilla and no-spectral arms test whether signal blocks are necessary.

Research references: [AV-HuBERT](https://arxiv.org/abs/2201.02184) motivates masked multimodal representation learning; [SyncNet](https://github.com/joonson/syncnet_python) motivates same-recording timing negatives. This is an independently implemented small pilot, not a reproduction of their published results.

## Data and accounting

[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D), source `1658cd342dff90010aa843eaeebd53610a08b1dc`, ODbL database / DbCL contents. Usable split counts: `{'train': 5061, 'validation': 210, 'test': 240}`. People: `{'train': 62, 'validation': 7, 'test': 8}`. Selected 5520 recordings; excluded 9 failed/short clips. Retained hours: `{'train': 3.6361444444444446, 'validation': 0.14745555555555556, 'test': 0.16771111111111112}`. All old test people remain excluded from training and validation; all eight new test people were untouched in previous cycles.

The codec's 249,248 PCA/statistical values fit only training people and are counted separately. Feature dimensions are audio128 (log-mel and changes), visual160 (appearance PCA96 and motion PCA64). The core predicts simultaneous missing modalities; these metrics do not claim causal future prediction. Sentence metadata only excludes false negatives among repeated prompts in contrastive training; it is never an encoder input. This is weakly informed alignment, not a claim of fully label-free learning.

Each run uses 1600 optimizer steps × 32 independent clips, 32-frame random physical-time windows, BF16 autocast with FP32 weights/AdamW. Parameter, data and step budgets are explicit; wall time/FLOPs are not claimed equal. Measured train-loop time totals 20.8 minutes; preparation, probes and rendering are additional.

## Frozen confirmation results

| Run | Architecture | Seed | Parameters | Timing correct | Energy control | Video→audio MSE ↓ | Audio→motion MSE ↓ | Seconds | Best step |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP-T2-301 | signal_small | 41 | 4,741,089 | 95.5% | 36.1% | 0.6563 | 1.6362 | 118.6 | 1400 |
| EXP-T2-302 | vanilla_small | 41 | 4,684,769 | 90.6% | 36.1% | 0.6892 | 1.6369 | 81.0 | 1200 |
| EXP-T2-303 | legacy_concat | 41 | 4,741,089 | 94.6% | 36.1% | 0.6876 | 1.6375 | 120.6 | 1400 |
| EXP-T2-304 | no_local_alignment | 41 | 4,741,089 | 34.2% | 36.1% | 0.6536 | 1.6380 | 105.2 | 1600 |
| EXP-T2-305 | shuffled_pairs | 41 | 4,741,089 | 7.9% | 36.1% | 1.0558 | 1.6395 | 120.3 | 0 |
| EXP-T2-306 | signal_large | 41 | 15,232,225 | 93.6% | 36.1% | 0.6295 | 1.6374 | 168.5 | 1600 |
| EXP-T2-307 | vanilla_large | 41 | 15,105,505 | 87.6% | 36.1% | 0.7016 | 1.6373 | 111.1 | 1200 |
| EXP-T2-308 | signal_large | 42 | 15,232,225 | 94.1% | 36.1% | 0.6320 | 1.6383 | 168.7 | 1400 |
| EXP-T2-309 | vanilla_large | 42 | 15,105,505 | 90.1% | 36.1% | 0.7232 | 1.6400 | 111.5 | 1000 |
| EXP-T2-310 | large_no_spectral | 41 | 15,193,057 | 90.1% | 36.1% | 0.6534 | 1.6376 | 144.6 | 1400 |

Timing selection chooses one of seven independently encoded valid equal-length windows at shifts −9, −6, −3, 0, 3, 6, 9 frames. Chance is 14.3%. Only clips long enough for every candidate enter this metric; counts are saved per run. Identical padding/duration cannot reveal the answer. Retrieval uses independent encodes and all test clips as candidates; true-recording and sentence-level matches are separately reported in metrics. No fused retrieval is used.

## Gates and input dependence

Synchronization passes both large seeds: **True**. Video→audio decoding: **True**. Audio→motion decoding: **False**. Audio corruption checks: **True**.

Registered decoding gates require ≥5% improvement over the training mean plus ≥2% gain from the correct input over a within-person wrong clip, with a positive 95% person-cluster interval. Confidence intervals resample people, not frames. See each run's `gates` and contribution intervals.

| Model | Original audio | Zeroed | Wrong clip, same person | Reversed time | Envelope only |
|---|---:|---:|---:|---:|---:|
| EXP-T2-301 | 95.5% | 3.5% | 17.3% | 18.3% | 63.9% |
| EXP-T2-305 | 7.9% | 12.4% | 8.9% | 9.4% | 8.4% |
| EXP-T2-306 | 93.6% | 5.4% | 16.8% | 14.9% | 67.8% |
| EXP-T2-307 | 87.6% | 1.0% | 16.3% | 13.4% | 65.3% |
| EXP-T2-308 | 94.1% | 5.4% | 18.3% | 17.3% | 72.8% |
| EXP-T2-309 | 90.1% | 3.0% | 17.8% | 16.8% | 70.3% |

These are frozen input-corruption diagnostics. They qualify the timing claim without tuning any model or threshold. Even a genuine synchronization gain may reflect speech energy/onset cues rather than semantic understanding.

## Does the additional capacity help?

Seed 41 larger-versus-small timing gain: -2.0%, person-cluster 95% interval [-5.5%, +1.4%]. Video→audio error improvement: +4.1%. Train time ratio: 1.42×. The small model has one seed; this is a controlled pilot, not a scaling law. Larger full/vanilla models have two seeds. Do not attribute the gain to parameter count when stronger features or the local objective explain it.

## What the model generates

The saved model directly predicts log-mel and lower-face features. Audio input produces a coarse mouth reconstruction and a separate rendering of predicted motion on a training-mean face. Video input produces predicted mel features rendered as a waveform using newly estimated Griffin-Lim phase. PCA reconstruction and same-modality reconstruction are codec/model controls. The original audio or video is never substituted into a generated output. The results are internal diagnostic proposals, not policy-committed behavior, realistic people, a neural vocoder, or open language generation.

Models used in the gallery are selected by validation: small full model, best-validation larger full seed, and its matched larger vanilla baseline. Examples include lowest/median/highest validation prediction error and fixed clips from three new test people. No test cherry-picking drives model choice.

Example `1048_IWW_HAP_XX` from `EXP-T2-308`:

- [Audio→mouth video](exploration/EXP-T2-308/validation_1048_IWW_HAP_XX/audio_to_mouth.gif) and [integrated predicted motion](exploration/EXP-T2-308/validation_1048_IWW_HAP_XX/audio_to_motion.gif).
- [Video→generated audio](exploration/EXP-T2-308/validation_1048_IWW_HAP_XX/video_to_audio.wav).
- [Original audio](exploration/EXP-T2-308/validation_1048_IWW_HAP_XX/reference_audio.wav), [codec audio](exploration/EXP-T2-308/validation_1048_IWW_HAP_XX/codec_audio.wav), [audio reconstruction](exploration/EXP-T2-308/validation_1048_IWW_HAP_XX/audio_reconstruction.wav), [no-input audio](exploration/EXP-T2-308/validation_1048_IWW_HAP_XX/null_audio.wav).
- [Original mouth](exploration/EXP-T2-308/validation_1048_IWW_HAP_XX/reference_mouth.gif), [codec mouth](exploration/EXP-T2-308/validation_1048_IWW_HAP_XX/codec_mouth.gif), [video reconstruction](exploration/EXP-T2-308/validation_1048_IWW_HAP_XX/video_reconstruction.gif), [no-input mouth](exploration/EXP-T2-308/validation_1048_IWW_HAP_XX/null_mouth.gif).

[Complete output gallery](exploration/gallery.md). The machine-readable gallery index `exploration/gallery.json` and each `example.json` contain exact provenance, conditioning, expected prompt and frozen-probe outputs. Standalone inference accepts either `--audio` or `--video`; its default output head selects the other modality.

## Frozen recognition probes

The text readout is a separately fitted, frozen-core classifier over 12 known sentence prompts and six emotion categories. It is closed-set recognition, not transcription or free text generation. Ridge probes and validation-only temperature calibration are saved separately. Raw-feature, shuffled-training-label and class-prior controls limit interpretation.

| Model | Input | Task | Learned test accuracy | Raw feature control | Shuffled labels | Class prior | ECE |
|---|---|---|---:|---:|---:|---:|---:|
| EXP-T2-301 | audio | emotion | 49.2% | 47.1% | 14.2% | 16.7% | 0.138 |
| EXP-T2-301 | audio | sentence | 82.1% | 84.6% | 15.0% | 17.1% | 0.185 |
| EXP-T2-301 | video | emotion | 35.4% | 31.2% | 15.8% | 16.7% | 0.123 |
| EXP-T2-301 | video | sentence | 35.8% | 14.6% | 16.2% | 17.1% | 0.137 |
| EXP-T2-308 | audio | emotion | 49.6% | 47.1% | 13.3% | 16.7% | 0.127 |
| EXP-T2-308 | audio | sentence | 86.2% | 84.6% | 16.7% | 17.1% | 0.029 |
| EXP-T2-308 | video | emotion | 39.6% | 31.2% | 12.5% | 16.7% | 0.076 |
| EXP-T2-308 | video | sentence | 41.7% | 14.6% | 17.1% | 17.1% | 0.166 |
| EXP-T2-309 | audio | emotion | 44.6% | 47.1% | 10.8% | 16.7% | 0.074 |
| EXP-T2-309 | audio | sentence | 67.5% | 84.6% | 17.1% | 17.1% | 0.080 |
| EXP-T2-309 | video | emotion | 34.2% | 31.2% | 15.0% | 16.7% | 0.114 |
| EXP-T2-309 | video | sentence | 24.2% | 14.6% | 16.7% | 17.1% | 0.071 |

## Failure cases, artifacts and limits

All per-clip errors and confidence intervals remain in run metrics. Coarse PCA mouth output cannot establish an unseen person's appearance. Integrating predicted motion can drift; MSE often favors a generic face or nearly zero motion. Griffin-Lim audio can be distorted even when its mel error improves. Listen to the codec ceiling alongside generated sound. Timing correspondence does not establish intelligible speech, grounded captioning, physical prediction, general world modeling, or hallucination reduction. Fixed face crops and acted speech constrain external validity.

Each run retains exact config, source/config/data hashes, optimizer/RNG state, validation-selected checkpoint, curves and registry. The source snapshot preserves the frozen training code. X0 records its own source, checkpoint and codec hashes. `comparison.csv`, `decision.json`, `causal_checks.json`, recognition probe weights, media and gallery are all retained.

## Next research decision

Retain the alignment module and validate on a second, more diverse physical-time audiovisual domain at the smallest effective size. Prioritize intelligibility and causal motion prediction before another parameter increase.

## Exploratory output-head diagnosis

After seeing the averaged mouth outputs, fixed-ridge linear heads were fitted to the frozen world field using training frames only. Core weights and all preregistered results stay unchanged. These supervised diagnostic readouts add separately counted coefficients; they test whether an alternative head can recover detail, not whether a new world model passed confirmation.

| Model | Head | Extra fitted parameters | Audio reconstruction MSE | Silent video→audio MSE | Audio→motion MSE | Visible-video appearance MSE |
|---|---|---:|---:|---:|---:|---:|
| EXP-T2-301 | original | 0 | 0.1280 | 0.6563 | 1.6362 | 0.8850 |
| EXP-T2-301 | frozen linear probe | 148,032 | 0.0831 | 0.6437 | 1.6384 | 0.5491 |
| EXP-T2-308 | original | 0 | 0.1126 | 0.6320 | 1.6383 | 0.8559 |
| EXP-T2-308 | frozen linear probe | 221,760 | 0.0715 | 0.6242 | 1.6413 | 0.4909 |

Original heads are already included in the model budget; the extra-count column counts only additional probe weights. The alternative readout improves some observed-signal reconstruction but does not resolve audio-conditioned motion or unknown appearance. The corresponding `audio_linear_*` and `video_linear_*` media are saved beside each selected median validation example.

## Optional external speech-intelligibility diagnostic

[Whisper tiny.en](https://huggingface.co/openai/whisper-tiny.en) is used only as an external, fallible recognizer of already-saved waveforms; it never generates, repairs or supplies the model audio or the closed-set recognition probe.

External evaluation-only model parameters: 37,760,256; revision `87c7102498dcde7456f24cfd30239ca606ed9063`. These are exploratory macro word-error rates on the selected gallery, not a representative ASR benchmark or human listening judgment.

| Audio condition | Mean recognizer word-error rate ↓ |
|---|---:|
| audio_reconstruction | 83.0% |
| codec_audio | 27.8% |
| null_audio | 100.0% |
| reference_audio | 34.7% |
| video_to_audio | 118.5% |

Original and codec errors distinguish recognizer/codec limitations from model errors. A distorted or null waveform can make the recognizer invent a phrase; those transcripts are labeled recognizer outputs and must not be represented as generated model text.
