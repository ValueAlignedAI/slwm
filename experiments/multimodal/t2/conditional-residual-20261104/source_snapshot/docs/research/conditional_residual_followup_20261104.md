# T2-G2 — preserve observed signals at the edge

This exploratory, separately registered extension follows the frozen G1 mel and
visual results. G1 audio fields reconstruct CREMA well but lose to a raw decoder
on RAVDESS (2.08 vs 0.267 mel MSE). The normalized core should not be forced to
carry every detail already present in an observed signal. No G1 artifacts,
thresholds, checkpoints or source snapshots change.

Test a fixed physical-grid interpolation of the observed 25 Hz mel as a skip
path, plus a zero-initialized correction bounded to ±0.25 normalized mel units.
The correction uses either the frozen audio field or raw audio features with
the same G1 temporal decoder architecture, with exact parameter counts recorded
at runtime. Both paths observe the same original audio. A frozen-field arm
additionally pays 4,741,089 core weights. No pretrained generator is used.

Run both arms with seeds 41 and 42, 1,600 updates, batch 32, 32-packet windows,
the same crops and G1 training/validation splits and audio objective. Validation
selects every 200 steps. Include deterministic interpolation with no trained
weights. Record all source/config/data/checkpoint hashes.

Gate: in both seeds, ≥5% lower 100 Hz mel MSE than deterministic interpolation,
≥5% lower independent recognizer WER than interpolation rendered at 100 Hz,
and no >2% relative error regression on the external diagnostic. A core benefit
additionally requires ≥5% lower mel error than the matched raw-residual head
with positive person-cluster paired interval. G2 architecture is informed by
G1's external failure, so RAVDESS is now a reused diagnostic, not an untouched
confirmation set. No external clips fit parameters or normalization.

Reuse the same 48 fixed listening cases. Add old audio-head predictions
interpolated and rendered at 100 Hz, separating a renderer improvement from a
learned-head improvement. Compare original, true 100 Hz codec, deterministic
25→100 Hz interpolation and all four learned residual heads. Full CREMA and
external mel metrics remain primary. This tests observation preservation and
reconstruction, not cross-modal generation or causal world dynamics. Retain
the simplest adequate path if the core adds no benefit.
