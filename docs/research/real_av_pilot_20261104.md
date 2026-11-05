# T2 real audiovisual falsification pilot — 2026-11-04

The user authorized rapid small-model research on `db@192.168.1.200` (RTX 3090).
This cycle finishes T2 using real recorded speech videos before joint text/code
training. Sentence transcripts remain metadata; they are never model inputs.
The existing generated-fixture results are mechanics evidence only.

## Question and frozen decision rule

Can the existing small shared SLWM processor learn useful audio/video future
features, and does paired input help beyond persistence, a vanilla Transformer,
and mismatched/absent modalities on previously unseen people?

Before reading test results, freeze these gates:

1. Prediction: at least 5% lower error than the stronger of train-mean and
   last-observation persistence in at least one modality; report both modalities.
2. Cross-modal contribution: at least 2% lower prediction error with the paired
   other modality than with a within-person shuffled other modality, with a
   person-cluster bootstrap 95% interval above zero. Report absent-modality errors.
3. Architectural promise: at least 5% lower error than the matched vanilla
   Transformer or at least 5 percentage points higher mean bidirectional
   independent retrieval R@1, in both training seeds.
   Otherwise keep the simpler baseline and do not scale the hybrid on this evidence.
4. Independent retrieval: encode each query/candidate with its other modality
   completely absent. Report the entire split candidate count, both directions,
   chance, permuted-label controls, and within-person correspondence AUC.

These are pilot gates, not replacements for the broader R0 success criteria.
No claim about general world models, language/code, hallucination, or policy
is justified by this narrow speech-domain experiment.

## Data and controls

CREMA-D is real paired human audio/video, with 12 fixed English sentence prompts.
Source: https://github.com/CheyneyComputerScience/CREMA-D
Pinned snapshot: `1658cd342dff90010aa843eaeebd53610a08b1dc`.
License: ODbL 1.0 database; DbCL 1.0 individual contents. Preserve attribution.
Use a deterministic emotion-balanced subset; split people before fitting anything.
Record source-file and feature hashes, failures, people, sentence/emotion counts,
and duration. Known broken source clips are excluded and logged.

Features: 32 synchronized normalized-time steps of 64-bin log-mel audio and
frozen ImageNet ResNet-18 frame features, with training-only PCA to 64 dimensions.
Normalization and PCA use training people only. Audio FFT windows are computed
separately on either side of the midpoint; no context window overlaps the future.
Every visual frame is encoded
independently; no temporal encoder can see a hidden future frame. Frozen backbone
parameters and external ImageNet pretraining must be counted separately from
all trainable adapters/core/heads. Actual frame rate/duration is recorded;
normalized time is a limitation, not a claim of learned physical timing.

Train: first half of both streams as context, predict second half; fixed
modality dropout; independent symmetric contrastive alignment. Targets are
detached fixed features. No trainable target encoder or hidden-target values
enter inputs. Labels/transcripts/person IDs are evaluation metadata only.

Planned matched arms: full SLWM, no spectral, no long convolution, vanilla
Transformer (all temporal mixers off), shuffled-pair training, audio-only,
video-only. Same split, batch schedule, optimizer, steps, and seeds. Trainable
parameter differences and measured time/memory must be reported. Two seeds
for the full model and vanilla baseline precede any scale decision.

Validation selects checkpoints; test is evaluated once per frozen arm, after
its training completes. Never tune on test. Report all planned arms, failures,
per-person errors, cluster intervals, training curves, and checkpoint hashes.

## Risks and stop conditions

Speech clips have restricted vocabulary, repeated prompts, and strong identity
and background cues. Unseen-person splits and within-person negatives reduce
these shortcuts without proving general grounding. Frozen ImageNet features may
underrepresent lip motion. Persistence is therefore a required strong control.
Poor alignment is a negative result, not permission to call joint-fusion
retrieval a success. Stop nonfinite training immediately. Do not silently change
losses, preprocessing, budgets, or test selection after results arrive.

## Reproduction

Preparation: `python -m training.t2_prepare_cremad --config configs/t2/cremad_prepare_pilot.json`.
Training/evaluation: `python -m training.t2_real_av --config configs/t2/cremad_pilot_matrix.json`.
Checkpoints include config, normalizer/PCA reference, optimizer, RNG states,
source hashes, and validation-selected step. A separate inference command
loads the selected checkpoint and predicts feature futures on a supplied clip.
