# T2-R2 — force observable content into the shared field

This separate exploratory extension responds to frozen R1 results. The raw skip
improves output fidelity but the full field's coarse-sound linear readout is worse
than the old frozen core. Removing the skip improves field decodability but hurts
actual rendering. These are readout-dependent measurements, not proof of
irreversible information loss. No R1 training artifacts or primary gates change.

Add a 256→64 auxiliary linear head (16,448 parameters) on the processed shared
field, with MSE ONLY on actually observed coarse audio packets. The auxiliary
head cannot see the raw skip. The main reconstruction/repair head and skip stay
unchanged. This supplies a content-preservation gradient even when the raw path
already reconstructs observations. Alignment is still optimized independently.

Primary weights: auxiliary coefficient .5, seeds 51 and 52. Controls: coefficient
zero, seed 51 (checks the implementation against R1), coefficient 2, seed 51
(predeclared strength sensitivity). All start from the SAME M1 source as R1, not
R1-trained checkpoints. Same 1,600 updates, sampled windows, channel draws,
dropout/masks, learning rates, data and validation-hole selection. Only the
16,448 auxiliary weights and their objective are added. Total compute differs.

CREMA/R1 Libri test-clean/RAVDESS results are now reused exploratory diagnostics.
Reserve official LibriSpeech test-other as a NEW harder recording-condition
check: first 12 eligible sorted speakers, four hash-selected complete 1.28–5.12s
utterances each, no overlap with used Libri speakers, no fitting, normalization,
hyperparameter tuning or checkpoint selection. Source https://www.openslr.org/12,
CC BY 4.0. Official archive MD5 fb5a50374b501bb3bac4815ee91d3135.

Before test-other scores: gate requires both .5 seeds' frozen-field ridge MSE to
improve at least 20% over R1 full fields and be no more than 5% worse than the old
frozen core, on reused test-clean and fresh test-other; positive paired-person
interval for the improvement over R1. Fit exactly the same training-only ridge
readout as the R1 addendum, never an external readout. Also require clean output
MSE no more than 5% worse than R1/raw controls, retain ≥90% CREMA timing and
≥40% reused RAVDESS timing, and report hidden-packet repair without treating it
as a passed R1 multimodal gate. Primary missing-packet MSE must not regress more
than 5% versus the R1 full model on reused CREMA/test-clean.

Listening: all 48 complete new test-other clips, same four conditions and scorer
rules as before, comparing clean/reference codec, interpolation, repaired audio
from all four new models, R1 full/old frozen/raw-audio controls. No original wave
samples are inserted into generated outputs. This is centered-mel packet erasure:
64ms codec windows may straddle boundaries, so it is not sample-exact waveform
inpainting. Independent recognizer outputs are evaluation text, not model text.
