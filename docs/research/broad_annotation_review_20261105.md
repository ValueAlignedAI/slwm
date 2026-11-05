# Broad-speech annotation review before fitting

The first cache has2,125/28,515 training records whose aligned word strings differ
from the official LibriSpeech transcript. Training-only inspection shows the
dominant cause is the alignment release's literal `<unk>` spelling, including
rare names and inflections; the original word intervals remain available.
The pending generator launch is held until this review resolves. No T2-B1 generator
weight, validation score or final speaker score exists at this point.

Revision2 restores official words only when token slots match one-to-one and
every aligned token is an unknown placeholder, an exact canonical match or an
identical known CMU pronunciation. Partial unknown spellings must also match
their explicit prefix/suffix. Original start/end times do not change. Cases
with ambiguous token segmentation or unmatched known words receive no phonetic/
CTC/TTS supervision; their acoustic examples remain. The same frozen rule applies
to every split. No speaker, waveform example or test case is removed.

Keep original rows, phone arrays, manifests and dataset/duration cards alongside
the revision. Packed mel values and the video PCA/audio normalizer stay byte for
byte unchanged. Recompute phone labels/duration targets and hashes; record every
unresolved lexical mask. WER references always use the original official text.
This fixes supervision before training, rather than tuning held-out outcomes.

`annotation_review.json` contains exact counts/IDs. The marker
`annotation_review_resolved.json` is written only after checks finish and allows
the already authorized GPU training to proceed. It is a data-integrity guard,
not a request for user approval.
