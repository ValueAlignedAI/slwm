# T2-G1 — conditional generation and frozen-world decoding

## Research question and scope

Can finer-rate sound targets and a single observed appearance frame repair output heads without changing the aligned core? Eleven preregistered decoder runs retain the 4,741,089-parameter frozen M1 core. The core is counted in addition to each decoder. Raw-feature controls use the same decoder and observations but do not pay the core cost; the 100 Hz audio-input diagnostic has richer observations and is not a matched core comparison.

This is diagnostic simultaneous conditional generation. All sound in a window is observed before generating its mouth motion. No policy commits, autonomous future-prediction or general world-model claims are tested.

## Data, compute and provenance

CREMA-D: 5,061 train / 210 validation / 240 test clips, 62 / 7 / 8 disjoint people, 3.636 training hours. This test split was evaluated in M1, so its results are exploratory. External RAVDESS: 64 clips from actors 01,02,13,14, two new utterances, no fitting, normalization, tuning or checkpoint selection. File SHA256 and ZIP member CRC verified. The fixed external crop was chosen from framing before scores. [RAVDESS source](https://zenodo.org/records/1188976), Livingstone & Russo (2018), CC BY-NC-SA 4.0; evaluation-only noncommercial research.

Each arm trains 1,600 updates on the same recorded training data. Audio: batch 32, 51,200 sampled 1.28s windows; visual: batch 16, 25,600 windows. Budgets are matched within each task, not across tasks. Validation selects every 200 steps. Continuous pixel/mouth/edge/temporal losses exclude the supplied seed. All visual heads initialize at persistence. No pretrained neural generator or source waveform supplies generated output. Audio phase is newly estimated by 48-step Griffin-Lim, independently of the recorded phase.

Decoder-training total: 6.27 minutes on one RTX 3090 (data preparation, field extraction, scoring and rendering excluded). Codec fit: 249,248 old PCA/statistical values + 128 new 100 Hz normalization values, training people only. Evaluator-only recognizer: 37,760,256 weights, pinned revision `87c7102498dcde7456f24cfd30239ca606ed9063`.

| ID | Conditioning | New trainable weights | Frozen core | Selected step | Train seconds |
|---|---|---:|---:|---:|---:|
| EXP-T2-401 | frozen audio field | 548,352 | 4,741,089 | 1600 | 9.8 |
| EXP-T2-402 | frozen video field | 548,352 | 4,741,089 | 1600 | 8.6 |
| EXP-T2-403 | raw 25 Hz audio | 548,352 | 0 | 1600 | 7.6 |
| EXP-T2-404 | raw video | 548,352 | 0 | 1600 | 7.5 |
| EXP-T2-405 | observed 100 Hz audio (richer input) | 548,352 | 0 | 1600 | 7.5 |
| EXP-T2-411 | audio field + seed | 213,699 | 4,741,089 | 1600 | 56.4 |
| EXP-T2-412 | raw audio + seed | 213,699 | 0 | 1400 | 56.1 |
| EXP-T2-413 | time + seed | 213,699 | 0 | 1400 | 55.9 |
| EXP-T2-414 | audio field + seed, no flow | 213,699 | 4,741,089 | 1600 | 54.8 |
| EXP-T2-415 | wrong-audio training + seed | 213,699 | 4,741,089 | 1400 | 56.1 |
| EXP-T2-416 | audio field + seed, seed 42 | 213,699 | 4,741,089 | 1200 | 56.1 |

## Reused CREMA test: visual results

Lower MSE is better; positive gains favor correct input. All visual predictions receive exactly one real seed. Motion ratio 0 is static persistence; 1 matches recorded motion energy but does not establish correct timing. Metrics use the fixed 32-frame middle window and exclude the seed from pixel errors. CUDA flow backward has nondeterminism; both seeds and actual checkpoint hashes are retained.

| ID | Mouth MSE | Gain vs seed | Gain vs time only | Correct vs wrong audio | Motion ratio | Gate |
|---|---:|---:|---:|---:|---:|---|
| EXP-T2-411 | 0.008707 | +21.7% | -0.0% | +3.67% [+1.48, +6.63] | 0.026 | FAIL |
| EXP-T2-412 | 0.008929 | +19.7% | -2.6% | +3.59% [+1.65, +6.13] | 0.027 | FAIL |
| EXP-T2-413 | 0.008705 | +21.7% | +0.0% | +0.00% [+0.00, +0.00] | 0.004 | FAIL |
| EXP-T2-414 | 0.008969 | +19.4% | -3.0% | +3.36% [+1.51, +5.67] | 0.013 | FAIL |
| EXP-T2-415 | 0.008657 | +22.2% | +0.5% | +0.00% [+0.00, +0.00] | 0.007 | FAIL |
| EXP-T2-416 | 0.009063 | +18.5% | -4.1% | +3.56% [+1.50, +6.37] | 0.024 | FAIL |

The frozen visual gate requires ≥5% lower mouth MSE than both persistence and time-only, ≥2% correct-versus-wrong audio benefit and a positive person-cluster 95% interval. External direction must persist without tuning. Full-frame and temporal errors and all per-clip values are in each metrics JSON. Four external people limit confidence.

## Reused CREMA test: audio results

All mel errors are measured on the same 100 Hz physical grid with training-only normalization. Old 25 Hz heads are linearly interpolated onto that grid, holding the final three subframes. WER uses the preregistered first four sample IDs per person: 32 CREMA / 16 external clips. Alphabetical CREMA selection concentrates on the DFA prompt; it is a narrow repeated-prompt diagnostic, not a vocabulary benchmark. Recognizer text is measurement, never text produced by SLWM. WER may exceed 100% through insertions.

| Condition | 100 Hz mel MSE | Temporal MSE | Mean clip WER | Audio gate |
|---|---:|---:|---:|---|
| codec25_interpolated | 0.0302 | 0.0223 | 65.9% | control |
| old_audio_interpolated | 0.1645 | 0.0299 | 103.4% | control |
| old_video_interpolated | 0.7464 | 0.0446 | 137.5% | control |
| EXP-T2-401: frozen audio field | 0.0843 | 0.0282 | 60.9% | PASS |
| EXP-T2-402: frozen video field | 0.8453 | 0.0477 | 100.0% | FAIL |
| EXP-T2-403: raw 25 Hz audio | 0.0526 | 0.0250 | 63.5% | PASS |
| EXP-T2-404: raw video | 0.8367 | 0.0472 | 103.1% | FAIL |
| EXP-T2-405: observed 100 Hz audio (richer input) | 0.0573 | 0.0176 | 51.0% | PASS (richer input) |

Original/25 Hz codec/100 Hz codec WER: 37.0% / 65.9% / 40.6%. Video-head wrong-video/null-video WER: 100.0% / 100.0%.

The frozen audio gate requires ≥20% relative recognizer-WER improvement versus the old rendered head, plus lower mel and temporal errors. Cross-modal audio must additionally improve over mean/null and same-person wrong-input controls with positive person-cluster input-dependence interval. A codec or same-modality reconstruction win is not a video-to-speech win. Null errors and paired WER changes/intervals are retained in the JSON reports.


## New external RAVDESS: visual results

Lower MSE is better; positive gains favor correct input. All visual predictions receive exactly one real seed. Motion ratio 0 is static persistence; 1 matches recorded motion energy but does not establish correct timing. Metrics use the fixed 32-frame middle window and exclude the seed from pixel errors. CUDA flow backward has nondeterminism; both seeds and actual checkpoint hashes are retained.

| ID | Mouth MSE | Gain vs seed | Gain vs time only | Correct vs wrong audio | Motion ratio | Gate |
|---|---:|---:|---:|---:|---:|---|
| EXP-T2-411 | 0.010497 | +16.3% | -5.5% | +0.72% [+0.12, +1.45] | 0.069 | FAIL |
| EXP-T2-412 | 0.010312 | +17.8% | -3.7% | -0.44% [-2.54, +1.66] | 0.021 | FAIL |
| EXP-T2-413 | 0.009948 | +20.7% | +0.0% | +0.00% [+0.00, +0.00] | 0.004 | FAIL |
| EXP-T2-414 | 0.010954 | +12.6% | -10.1% | +0.17% [-0.25, +0.72] | 0.025 | FAIL |
| EXP-T2-415 | 0.010023 | +20.1% | -0.8% | +0.00% [+0.00, +0.00] | 0.018 | FAIL |
| EXP-T2-416 | 0.010759 | +14.2% | -8.2% | +0.71% [+0.34, +1.08] | 0.051 | FAIL |

The frozen visual gate requires ≥5% lower mouth MSE than both persistence and time-only, ≥2% correct-versus-wrong audio benefit and a positive person-cluster 95% interval. External direction must persist without tuning. Full-frame and temporal errors and all per-clip values are in each metrics JSON. Four external people limit confidence.

## New external RAVDESS: audio results

All mel errors are measured on the same 100 Hz physical grid with training-only normalization. Old 25 Hz heads are linearly interpolated onto that grid, holding the final three subframes. WER uses the preregistered first four sample IDs per person: 32 CREMA / 16 external clips. Alphabetical CREMA selection concentrates on the DFA prompt; it is a narrow repeated-prompt diagnostic, not a vocabulary benchmark. Recognizer text is measurement, never text produced by SLWM. WER may exceed 100% through insertions.

| Condition | 100 Hz mel MSE | Temporal MSE | Mean clip WER | Audio gate |
|---|---:|---:|---:|---|
| codec25_interpolated | 0.0622 | 0.0458 | 12.5% | control |
| old_audio_interpolated | 0.5001 | 0.0577 | 84.4% | control |
| old_video_interpolated | 1.9461 | 0.0973 | 102.1% | control |
| EXP-T2-401: frozen audio field | 2.0798 | 0.1376 | 102.1% | FAIL |
| EXP-T2-402: frozen video field | 2.4300 | 0.1002 | 101.0% | FAIL |
| EXP-T2-403: raw 25 Hz audio | 0.2667 | 0.0523 | 14.6% | PASS |
| EXP-T2-404: raw video | 2.2296 | 0.0962 | 100.0% | FAIL |
| EXP-T2-405: observed 100 Hz audio (richer input) | 3.6438 | 0.0342 | 6.2% | FAIL (richer input) |

Original/25 Hz codec/100 Hz codec WER: 0.0% / 12.5% / 1.0%. Video-head wrong-video/null-video WER: 101.0% / 100.0%.

The frozen audio gate requires ≥20% relative recognizer-WER improvement versus the old rendered head, plus lower mel and temporal errors. Cross-modal audio must additionally improve over mean/null and same-person wrong-input controls with positive person-cluster input-dependence interval. A codec or same-modality reconstruction win is not a video-to-speech win. Null errors and paired WER changes/intervals are retained in the JSON reports.

## Additional frozen alignment audit

On RAVDESS, the original small frozen core selects the correct shift on 53.1% of 64 clips, versus 3.1% for the energy/motion control and 14.3% chance. This is an exploratory extension to M1; it never selected or tuned G1 checkpoints.

## Paired core versus simpler decoder comparisons

Intervals resample people (2,000 bootstrap draws). These are mean per-clip relative improvements; ratios of aggregate MSE above can differ.

| Dataset | Comparison | Relative gain | Person-cluster 95% interval |
|---|---|---:|---:|
| test | EXP-T2-401_vs_EXP-T2-403 | -65.62% | [-71.17, -60.53]% |
| test | EXP-T2-402_vs_EXP-T2-404 | -2.60% | [-5.73, +0.05]% |
| test | EXP-T2-411_vs_EXP-T2-412 | +1.99% | [+0.23, +4.17]% |
| test | EXP-T2-411_vs_EXP-T2-413 | +0.05% | [-3.80, +5.27]% |
| test | EXP-T2-416_vs_EXP-T2-413 | -4.25% | [-7.90, +0.52]% |
| external | EXP-T2-401_vs_EXP-T2-403 | -786.85% | [-919.61, -657.71]% |
| external | EXP-T2-402_vs_EXP-T2-404 | -22.39% | [-50.29, +7.23]% |
| external | EXP-T2-411_vs_EXP-T2-412 | -3.37% | [-8.95, +1.90]% |
| external | EXP-T2-411_vs_EXP-T2-413 | -5.88% | [-7.53, -4.24]% |
| external | EXP-T2-416_vs_EXP-T2-413 | -9.24% | [-11.11, -6.85]% |

## Artifacts and reproducibility

`summary.json` and each experiment's `registry.json`, configuration, curves, validation/test/external metrics, optimizer-bearing checkpoint and SHA256 retain every arm, including failures. `source_snapshot/` and `source_hashes.json` freeze training implementation. Corpus `*_manifest.json` files bind all array and normalization hashes. `exploration/gallery.json` lists validation low/median/high visual cases, fixed test/external cases and all 48 listening cases with original/codec/model/wrong/null controls. The real seed is saved separately. GIF/WebM and WAV files contain actual decoded predictions. Source artifacts, raw media and decoded caches remain on Fedora; reusable heads and media are mirrored locally.

Reproduce: `bash scripts/train/run_conditional_generation_fedora.sh`. Standalone inference: `python -m exploration.generation_infer --help`. No inference updates weights. There is no generated free text or policy/uncertainty head in this stage.

Research basis: [SadTalker](https://arxiv.org/abs/2211.12194) motivates separating reference appearance and motion; [EnCodec](https://arxiv.org/abs/2210.13438) motivates treating compression quality separately. Neither pretrained generator is used. Full interpretation and next allowed research step are in `docs/research/findings.md`.
