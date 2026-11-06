# G1 executable calibration and final-data lock

Implementation supplement registered while cores train, before any new-role
calibration predictions or readout fits. The initial case plan and primary
success gates remain fixed. This document supplies the decoder/gating mechanics
and preserves every failed condition. Calibration performance cannot alter core
training, case selection, data roles or the primary final criteria.

For each frozen endpoint, evaluate the five registered physical-mel mixtures of
its direct residual prediction and R1's predicted-phone output on the64 reserved
Libri calibration utterances. Select minimum independent-recognizer corpus WER;
ties prefer less R1 contribution. Actual reference phones never enter this
selection or the primary predictor. They are a separately labelled ceiling.
Known fine FFT frames and all waveform samples outside the160ms gap stay exact.
Outputs use lossless16-bit FLAC; compression changes no evaluated PCM sample.

Use own-active/nonoverlap AMI calibration windows to select one matching-score
threshold from the fixed grid, maximizing balanced accuracy of aligned versus
the four registered same-person timing negatives. Ties prefer greater rejection;
the final value above1 rejects every video. Rejected video takes exactly the
audio-only forward path, including for the fixed-modality Transformer control.
Score quiet/overlap windows separately in the final set. No final speaker
chooses a threshold. Two calibration meeting components limit calibration
precision; six new final components supply the primary timing/speech check.

All core endpoints, both-seed readout weights/projections, recipe/source hashes,
calibration rules and case/source contracts must be fixed before creating the
global evaluation lock. Test extractors refuse final observations without that
lock. Re-running a completed evaluation may read its immutable outputs but may
not fit new weights or change its rules. Record any failure/revision explicitly.

Primary fresh Libri tests retain all256 utterances; primary AMI comparisons use
all available own-nonoverlap cases among the210 fixed cases. Quiet/overlap cases
remain in raw outputs and stratified reports. Empty owner references retain
insertion counts; they are not automatically acoustic silence. Report direct,
predicted-phone, calibrated mixture, practical interpolation, source and oracle
ceilings without substituting an external recognizer transcript for model text.

The window-local character output is labelled local. A complete transcript
requires a separately identified whole-recording rollout and cannot be inferred
from a single1.365s received repair context. The primary speech outcome is
independent waveform recognition, with the fallible scorer/version/count shown.

The64/320ms gap diagnostics use two deterministic hash-ordered primary utterances
per final speech person,64 clips per duration. Every primary160ms case remains.
This bounded secondary subset is metadata-only and fixed before calibration or
final prediction; it cannot select weights, decoder mixtures or reported gates.
