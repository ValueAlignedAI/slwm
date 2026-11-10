# T2-R1 — preserve content, repair gaps, retain audiovisual alignment

The user's continuing research authorization permits this isolated core-training
experiment on the Fedora RTX 3090. Earlier artifacts and user edits remain intact.
This tests simultaneous masked reconstruction, not causal future prediction,
free speech generation, text/code grounding, policy commitment or general world
understanding. Research-selected heads produce diagnostic proposals only.

## Hypothesis and intervention

The old alignment-only core lost new-domain speech detail. Train its adapters and
processor jointly for high-rate signal retention and masked audio repair, while
retaining blind audiovisual correspondence. Preserve an explicit observed-signal
path to the output, with a bounded correction: ±0.25 normalized units in observed
packets and ±3 in hidden packets. A no-skip arm tests the path itself.

Input: 32 physical 25 Hz packets. Hide two disjoint 4-packet (160 ms) audio spans,
keeping boundary anchors, then fill the raw baseline by interpolation of OBSERVED
packets only. Compute audio deltas AFTER filling. Neither a hidden mel value nor
a delta derived from it may reach any input. The core masks hidden packets before
its adapter. All models see the same 32-packet context: no cached wider fields.
Video is present for half of the paired training cases. Video-only supply of
identity is not inferred from sound.

A 100 Hz reconstruction objective (four subframes per packet) emphasizes holes
4×, with temporal L1. Independent alignment loss uses only CREMA audio/video,
with shifted and within-person negatives; neither representation sees its partner.
Gain/EQ augmentation simulates channel changes: uniform ±12 dB gain, four-knot
smooth ±6 dB EQ. Apply the SAME transformation to observations and targets,
preserving actual observed level rather than asking the model to guess an
unknowable original gain. This is a mel-domain approximation, not a room simulator.

AV-HuBERT motivates masked multimodal representation learning, and speech
inpainting motivates a task where partial sound plus video could be useful.
No external pretrained generator, phonetic model or text target supplies outputs:
https://arxiv.org/abs/2201.02184;
https://www.isca-archive.org/interspeech_2023/montesinos23_interspeech.html.

## Fixed data and new holdouts

Keep CREMA's 5,061/210/240 person-disjoint split. Its test is reused research data.
Add real LibriSpeech dev-clean audio: deterministically shuffle its 40 speakers
with seed 7301; 32 train (up to 40 clips each), 8 validation (up to 8 each).
Clips are at least 1.28s, truncated to 5.12s for signal training; transcripts are
metadata only. Mixture per update: 24 CREMA windows + 8 LibriSpeech windows.
All codec/PCA/normalization values remain CREMA-training-only, fitted previously.

New external audio: official LibriSpeech test-clean, first 12 sorted eligible
speakers with at least four complete 1.28–5.12s utterances, four hash-selected clips
per speaker (48). No external fitting, normalization, tuning or selection.
Official archives MD5-checked, individual files SHA256-recorded. Source/license:
https://www.openslr.org/12, CC BY 4.0, Panayotov et al. (2015).

New external audiovisual set: RAVDESS actors 03,04,05,06,15,16,17,18, full AV
speech, normal intensity, repetition 01, both statements/all emotions (128).
No overlap with G1/G2 external actors 01,02,13,14. Keep the previously fixed crop,
without inspecting model scores to adjust framing. Official ZIP member CRC and
raw SHA checked. CC BY-NC-SA 4.0, Livingstone & Russo (2018), held-out evaluation
only: https://zenodo.org/records/1188976. New people, but same corpus/two utterances.

## Arms, accounting and selection frozen before training

Nine runs: two full signal seeds (51,52), vanilla processor, no alignment,
no channel augmentation, no raw skip, raw audio head, raw audiovisual head, frozen
old core. Train 1,600×32 windows each. Core LR .0001; head LR .0004; warmup 80,
cosine decay. Align every second update, weight .15. Initialize neural cores
from the earlier corresponding 1,600-step M1 signal/vanilla checkpoints. The
same M1 pretraining data/update budget is disclosed; raw controls have no such
pretraining. Remove the unused old reconstruction heads; record exact active
trainable/frozen counts. Vanilla and raw controls have different total sizes/costs.
Only data, head architecture and new update budgets are matched, not all weights
or total compute. Raw AV additionally learns its 290→256 input projection.

Use identical sampled indices, windows, corruption, channel draws and video drops
within each seed, including unused random draws for ablations. Validation every
400 updates selects minimum average hidden-packet MSE across CREMA and Libri
validation (equal corpus weights). No test/external values select checkpoints.
Record every arm, failure, curve, source/config/data hash and optimizer/RNG state.

## Gates and frozen evaluation

Evaluate clean sound, 25% missing sound alone, missing sound with correct video,
same-person wrong video and null video. Middle 32-frame windows, all test clips.
Hole MSE and temporal error are measured on the identical 100 Hz physical grid.
Person-cluster 95% bootstrap intervals (2,000 draws) accompany paired changes.

1. Content retention: both full seeds' clean-audio MSE no more than 5% worse than
   the newly trained raw-audio control, including new LibriSpeech.
2. Repair: full model hole MSE at least 5% better than raw-AV and vanilla at seed
   51, positive paired cluster interval; both full seeds improve raw-AV direction.
3. Multimodal contribution: correct video improves hole MSE at least 5% over
   audio alone and at least 2% over wrong same-person video, with positive cluster
   intervals on reused CREMA and direction retained on fresh RAVDESS people.
4. Alignment: at least 90% seven-shift accuracy on CREMA, at least 40% on fresh
   RAVDESS; report chance 14.3%, energy control and wrong-audio check. New-domain
   uncertainty must be explicit. No broader capability claim follows a passed gate.

Listening diagnostic: all 48 complete new LibriSpeech utterances; two evenly
spaced sample IDs per CREMA person (16) and two neutral statements per new
RAVDESS actor (16). Compare clean/reference codec, zero-filled corruption,
interpolation, raw/audio-AV/full repairs and wrong-video. The pinned independent
Whisper-tiny.en evaluator is fallible and never trains or generates SLWM output.
Actual inverse-mel/new-phase audio and error/time plots are retained. Full-clip
rendering uses one global corruption mask before overlapping windows, preventing
hidden audio from reappearing through a neighboring window. Repaired audio is
not a copy of original waveform samples. Keep failed gates and stop scaling if
cheap controls explain the result.
