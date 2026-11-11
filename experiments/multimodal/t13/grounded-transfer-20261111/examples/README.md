# Listen to speech and replay real physical clocks

Each robot stream links original frames from one parent recording. Every join shares two consecutive image-and-pose identities. The stream contains only causal observation history and an explicit four-control input proposal. Later observed motion is kept in separate outcome files. The three replays contain 306, 286 and 271 observations with 299, 279 and 264 scored clocks. They are real recorded continuations, including the blocks, camera and robot telemetry. The hypothetical alternative plans in the decision JSON were never physically executed. [Stream lengths and parents](robot-streams.json) · [Run the streams](../inference/README.md)

| Stream | Causal sensor windows | Source continuity | Gated outputs |
| --- | --- | --- | --- |
| 1 | [Signals](robot-stream-1.npz) | [Frame identities](robot-stream-1-lineage.json) | [Decisions](robot-stream-1-decisions.jsonl) |
| 2 | [Signals](robot-stream-2.npz) | [Frame identities](robot-stream-2-lineage.json) | [Decisions](robot-stream-2-decisions.jsonl) |
| 3 | [Signals](robot-stream-3.npz) | [Frame identities](robot-stream-3-lineage.json) | [Decisions](robot-stream-3-decisions.jsonl) |

**Controlled audio** comes from public human Speech Commands V2 recordings. The actual and opposite words in each row come from the same held speaker. Their association with real robot trajectories is constructed from hindsight text. The outcomes are scored from separate physical observations.

| Real direction word | Actual microphone clip | Same-speaker opposite |
| --- | --- | --- |
| Left | [Listen](voice-left-true.wav) | [Listen](voice-left-foil.wav) |
| Right | [Listen](voice-right-true.wav) | [Listen](voice-right-foil.wav) |
| Up | [Listen](voice-up-true.wav) | [Listen](voice-up-foil.wav) |
| Down | [Listen](voice-down-true.wav) | [Listen](voice-down-foil.wav) |

**Naturally paired audio** comes from the original pitch-shifted HoloAssist instructor microphones. Each foil is an earlier, distinct directive in the same recording. The two examples were selected by source hash without model scores. [First actual instruction](component-A-true.wav) · [First same-scene foil](component-A-foil.wav) · [Second actual instruction](component-B-true.wav) · [Second foil](component-B-foil.wav) · [Source, native text and observed components](component-examples.json)

Robot data are from Lynch and colleagues’ Language-Table project. Natural device recordings are from Wang and colleagues’ HoloAssist. Direction clips are from Warden and the Speech Commands contributors. Original data notices accompany the redistributed audio. [Language-Table](https://github.com/google-research/language-table) · [HoloAssist agreement](HoloAssist-LICENSE.txt) · [Speech Commands license](Speech-Commands-LICENSE.txt) · [Original voice IDs and outcomes](voice-examples.json)
