# Confirmation on eight different speakers and unseen sentence combinations

The 202.35M hybrid passed the registered four-speaker development gate at 2,048 updates for both training seeds. The six-direction mean at eight examples per command was 32.18% versus 24.15% for dense at seed 511, and 26.48% versus 23.03% at seed 512. The favorable directions differed. This motivates a single follow-on replication before attributing broad modality independence.

Freeze both sizes, both cores, both seeds and all eight 2,048-update inference hashes. Use the same support utterances from people 1–16, the same four temporal averages of the shared state, the same ridge penalty and the same 8- and 32-shot readouts. Evaluate once on GRID people 25–32, sentence hash bucket 9. No support person, complete sentence or clip may overlap confirmation. Only the paired GRID views are scored. Earlier project studies evaluated these people; they have not trained or selected the G6 checkpoints. The project-wide unused people 33/34 remain unopened.

Repeat the original eight-shot gate unchanged on the eight held speakers. Report the absolute hybrid and dense cross-view means relative to 25% chance, their speaker-paired interval, every direction, same-view controls, shuffled labels and retrieval. All eight selected checkpoints are reported regardless of their individual development result. This is a confirmation on a controlled four-command speech grammar.

The initial scorer entry point rejected its lock argument before reading confirmation examples. The argument was repaired and verified. All eight checkpoints were scored under the same previously frozen SHA-256 lock, with the original supports and sentence exclusions. [Paired confirmation results](transfer-and-decisions.md)
