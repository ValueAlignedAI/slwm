# T2-R1 content readout addendum (exploratory, before external scores)

The raw output skip could preserve sound while the core still loses detail.
Therefore independently probe ONLY frozen audio-only core fields. This does not
change training, checkpoint selection or any primary gate.

For each selected core, fit a 256→64 ridge readout on at most 512 hash-selected
CREMA and 512 Libri training clips, one middle 32-packet window each, equal corpus
weight. Targets are observed coarse-time mel frames on the fixed 100 Hz codec
normalization. Readouts never receive raw skip values. Use fixed ridge
lambda=.01*trace(X'X)/256 after centering, no validation search or external fit.
Count the 16,448 output coefficients/bias and 256 fitted input means separately.

Evaluate all CREMA/Libri validation and test clips plus fresh RAVDESS. Compare
new cores to EXP-T2-509 (the old frozen core), including per-person intervals.
These measure linear decodability/distortion of observed sound, not missing-sound
prediction, phonetic understanding or text generation. Raw input trivially
contains the target, so no claim of a core advantage over copying is possible.
This readout is never a source of the primary rendered speech.
