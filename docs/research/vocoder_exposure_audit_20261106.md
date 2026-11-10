# Freshness scope of the waveform pipeline

Audit recorded while G1 trains, before any fresh calibration/test prediction.
No data role, core weight, case, metric or success gate changes in response.

The official [Vocos pretrained-model table](https://github.com/gemelo-ai/vocos#pre-trained-models)
identifies the13.5M-parameter mel24kHz model as trained on LibriTTS. The official
[LibriTTS source](https://www.openslr.org/60/) states that it derives from the
original LibriSpeech recordings, including the train-clean-360 material used
for G1's new speaker cohort. A released per-checkpoint training file list has
not been verified. Exact exposure of each held speaker/recording is therefore
unknown; it cannot be ruled out. This is a potential upstream overlap, not
proof that a particular utterance was memorized.

The32 new Libri final people remain unexposed to our shared cores, R1 fitting,
normalizers, weight selection and calibration, as enforced by their data roles.
That is the scope of the fresh-speaker claim. The full waveform system cannot
claim all-component speaker novelty. The external Whisper scorer's exact
training overlap is also unverified and its WER is a fallible intelligibility
proxy, not a human-listening score.

All architectures and interpolation/codec controls use the same fixed Vocos
weights. Paired waveform effects still measure the incremental predictor under
that common decoder; they do not certify a decoder-unseen speaker pipeline.
Report source and oracle-mel/phone ceilings, actual core phone errors and
whole-recording CTC outputs separately. No recognizer output trains a core or
R1. New-source sensor/action transfer and from-scratch visual encoders do not
use this waveform decoder.

If end-to-end unseen-voice synthesis is the next claim, add a corpus excluded
from upstream vocoder training or train a renderer solely on the registered
training people. Preserve the present results as scoped evidence; do not
silently relabel or replace the held cohort after scoring.
