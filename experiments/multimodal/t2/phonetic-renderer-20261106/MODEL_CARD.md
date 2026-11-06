# R1 phonetic acoustic renderer

A 289,860-parameter phone-to-mel edge trained from scratch on the earlier100h/251-speaker speech training corpus. This is a supervised acoustic codec, not an arbitrary-text TTS model and not evidence of a general world representation. G1 supplies predicted phone IDs; reference MFA phones are reserved for ceilings.

The selected and fixed endpoint are update6000. Repeated sampled context exposure is291.27h, distinct from the100h unique source. Selection uses20 previously exposed dev-other speakers; no new final speaker is used. The best normalized oracle-phone selection MSE is0.230614.

| Old-selection condition | WER |
|---|---:|
| interpolation | 9.63% |
| oracle_phone_full | 11.30% |
| oracle_phone_repair | 6.64% |
| silenced | 7.64% |
| source | 5.98% |
| source_mel_roundtrip | 6.64% |

These eight deterministic selection cases contain301 normalized reference words. Oracle-phone conditions receive true phone identities/timing and are acoustic ceilings, never speech-content prediction results. The full oracle-phone output is intelligible enough to diagnose downstream phonetic predictions, but it loses quality relative to the original and source-mel roundtrip. Small selection-only numbers do not establish a fresh-speaker improvement.

Artifacts: `best.pt`, `endpoint.pt`, `resume.pt`, `source_hashes.json`, `curves.json`, and `old_selection_diagnostic/` with every waveform and raw independent-recognizer transcript. Known PCM outside the160ms repair interval is asserted exact.

Inference also needs the separately pinned Vocos renderer (13,531,650 parameters). The Whisper-tiny.en scorer (37,760,256 parameters) is evaluation-only. R1 uses actual supplied MFA labels, an observed-only voice/prosody summary, and about32ms phone-unit lookahead; Vocos uses bounded additional predicted-mel context. No source phase is used.

Data sampling is keyed by seed/update and there are no stochastic post-initialization Torch layers; the resume retains optimizer state and update counters. Raw Torch RNG arrays were not serialized in R1. The frozen training sources and exact hashes are preserved.

The primary next test is G1-predicted phones on the untouched new-speaker cases. A good oracle cannot substitute for that test.
