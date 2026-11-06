# R1: isolate content prediction from acoustic rendering

The previous direct-mel systems failed intelligibility gates. A poor waveform can
come from lost speech content or an inadequate acoustic decoder. Train one small
phone-conditioned renderer shared by every G1 processor; this is an edge codec,
not a claim that a shared world field learned language or speech by itself.

Use actual MFA phone intervals in immutable LibriSpeech100h training audio,
251 training people, with unknown/ambiguous lexical intervals masked. Supplied
forced alignments are external supervised annotations. Select on the previously
exposed20-person dev-other selection source; none of G1's new16 selection,
16 calibration or32 final people trains or selects this codec. The provided
vocabulary has79 phone/allophone labels. Keep its exact hash and source features.

The ~0.29M renderer receives phone IDs, a100-band observed-only global voice
summary and two observed-only/interpolated spectral prosody summaries. Missing
waveform gaps include FFT support; no hidden mel enters conditioning. Drop the
voice/prosody cue20% of training calls to expose a mean-voice fallback. A small
temporal convolution predicts normalized mel, with32ms unit lookahead. Frozen
Vocos subsequently renders waveform and is counted separately (13,531,650
pretrained weights). There are no pretrained TTS or recognition weights here.

At most6000 updates, batch128 of128 fine frames, same fixed AdamW/cosine recipe;
release the common GPU lock after1000 updates. Select on fixed dev-other mel
error, stop after two checks10% worse than the best, retain endpoint/selected
weights, full optimizer/RNG/config/source hashes and actual exposure/time. No
new final person is decoded or scored before all compared G1 weights freeze.

G1 inference supplies predicted phone IDs. Ground-truth-phone rendering is an
explicit acoustic ceiling, separate from model-predicted speech. Compare both
with the direct mel head and practical fine interpolation under the same Vocos
decoder. Calibrate the residual/mixing choice on registered calibration people
only. New final WER never chooses the head, parameters or decoding settings.
An intelligible oracle with poor predicted speech localizes the content problem;
an unintelligible oracle identifies a renderer ceiling. Both outcomes are useful.
