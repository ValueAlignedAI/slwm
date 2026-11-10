# T2-M1 alignment architecture and X0 output probes — 2026-11-04

The user explicitly authorized architecture changes, modest parameter scaling,
matching modalities, and exploring outputs conditioned on audio or video.
This supersedes the preceding recipe's no-scaling recommendation: test a new
recipe, keep the old evidence, and do not claim the old gates passed.

## Sequential work

1. T2-M1: establish physical-time mouth/audio features, small and moderately
   larger shared processors, independent alignment, and missing-modality decode.
2. X0-M1: freeze checkpoints, render diagnostic audio/video outputs and train
   closed-set sentence/emotion probes. These outputs are uncommitted diagnostics.
   Reporting proceeds even if alignment fails; no large model is then justified.

## Research basis

- AV-HuBERT: https://arxiv.org/abs/2201.02184 — mask modalities and predict shared
  information from speech/lip motion. This experiment does not reproduce its
  architecture, pretraining scale, or published capabilities.
- SyncNet: https://github.com/joonson/syncnet_python — compare temporally aligned
  audiovisual segments with same-recording time-shifted negatives.
- Existing SLWM experiments: appearance-heavy frozen features, normalized time,
  and weak paired-modality benefit motivate local motion and fixed physical time.

## Data frozen before new results

Continue pinned CREMA-D `1658cd342dff90010aa843eaeebd53610a08b1dc` under ODbL/DbCL.
Use the old 34 training people plus 28 previously unused training people;
retain the original seven validation people. Reserve eight completely unused
people for confirmation: 1039, 1081, 1064, 1036, 1054, 1029, 1046, 1026.
All 14 people in earlier test sets remain excluded from training/validation.
Training may use all available utterances of its 62 people; sample at most
30 balanced clips per validation/test person. Record failures and hashes.

Use 25 Hz physical timestamps, 16 kHz audio, 64-bin log-power mel, and a fixed
lower-face crop resized to 48×32 grayscale. Fit appearance PCA96, motion PCA64
and per-feature scaling only on training frames. Both neural backbones start
from scratch; no external frozen perception network is required. PCA/basis
fitted statistics are counted separately. Crop position and codec quality are
limitations and are visualized before interpreting model outputs.

## Architecture and planned comparisons

Modality-specific temporal adapters feed a time-aligned shared latent field;
local, spectral, long-convolution and attention blocks update it. All heads
are edge readouts. Compare approximately 5M (256 width, four blocks) with
approximately 15M (384 width, six blocks); record exact counts after allocation.
Vanilla Transformer and legacy-concatenation baselines use identical features,
training clips and step/sample budgets. Include no-local-alignment and shuffled
pair controls. Full/vanilla larger models each use seeds 41 and 42.

Losses: audio mel/delta prediction, visual motion/appearance prediction,
independent clip/token alignment and same-clip synchronization ranking. Train
paired, audio-only and video-only input conditions; target values cannot enter
missing inputs. Random physical-time crops remove a fixed sentence-position
shortcut. Alignment encodes cannot see their partner and omit absolute position
signals. Any sentence metadata used to exclude false negatives is disclosed;
text generation or general ASR is not claimed. X0 label probes are trained only
after freezing the processor, counted separately, and evaluated against priors.

## Frozen gates and controls

Primary alignment: independent same-recording offset selection among seven
valid equal-length windows at shifts -9,-6,-3,0,3,6,9 frames. Chance is 1/7;
no padding, candidate identity, or duration can reveal the aligned answer.
Require ≥30% correct and improvement over untrained motion/audio-envelope
correlation, with a positive person-cluster 95% interval, in both larger seeds.
Report full-set retrieval, same-person wrong-clip correspondence, permuted labels,
null inputs and missing-modality errors. Never substitute fused retrieval.

Cross-modal decoding: video-only audio and audio-only motion must improve ≥5%
over a training-mean/time-profile predictor, and correctly paired input must
improve ≥2% over within-person wrong input with a positive cluster interval.
Report full/vanilla, scaling, negative controls, validation curves, memory,
throughput, parameters and measured time. Scaling only earns support if larger
models improve the alignment/decode gates consistently, not just training loss.

Save all checkpoints and exact source/config/data hashes. Select checkpoints
using validation only. Evaluate the new confirmation people once per frozen
arm, and never change thresholds or budgets from their results. Stop nonfinite
runs immediately and retain the failure record.

## Exploratory generation contract

Audio input → predicted mouth/motion video; silent video → predicted mel →
Griffin-Lim waveform; show same-modality reconstruction for comparison. Inverse
PCA and phase estimation are diagnostic codecs, not evidence of realistic
video/speech generation. An audio-only video starts with a training-mean face,
not an inferred identity. Never add original audio to a generated-audio output
or original video frames to generated video while labeling them predictions.
Provide original/codec reconstruction/model/null controls and reproducible
examples selected from best/median/worst validation errors plus frozen test IDs.
A learned sentence probe verbalizes one of 12 known prompts; label it closed-set
recognition and report confidence and accuracy. No broad-language claim follows.
