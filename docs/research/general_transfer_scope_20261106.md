# G1: diverse signals and transfer, acquisition brief2026-11-06

The user explicitly requests broader modalities, more data and action experiments
despite previous B1/P1 gate failures. Keep the500M text comparison independent.
This brief authorizes source collection and schema audits, not a capability
claim or an unfrozen training comparison. Freeze actual data roles, architecture,
loss/compute budgets and success tests before the first shared-core fit.

The previous model was specialized to speech and face motion, with a phonetic
text edge and sensor-choice policy. Fresh speakers test identity generalization
within that task. They do not establish transfer across tasks or source families.
The next objective is reusable state across heterogeneous signals; specialize
small readouts only after shared-state training, then test new tasks/sources.

Acquire the full officially available COCO2017 train/validation scene images
and captions (~118k/5k images), environmental sound/captions from
[Clotho2.1](https://zenodo.org/records/4783391), wrist/chest/ankle inertial,
magnetic/temperature/heart-rate signals from
[PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring),
and the separate
[MHEALTH](https://archive.ics.uci.edu/dataset/319/mhealth+dataset) sensor corpus
for cross-source transfer. Physiological values are generic time-series, with
no clinical claim. Do not use activity labels as a forecasting/classification
input except in explicitly supplied-language conditioning.

[COCO's official downloads](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm)
and per-image licences stay in the source card. Clotho's audio has per-file
Freesound licences and its captions retain their original noncommercial licence.
Preserve attribution and source hashes; do not substitute unofficial mirrors
as verified originals. Record every failed acquisition and actual final counts.

Expand the [official DROID100 preview](https://droid-dataset.github.io/droid/the-droid-dataset)
from two to all100 episodes for a bounded scene/proprioception/language/action
study. It remains a debugging-sized robotics corpus, not adequate proof of a
scale-trained robot model. Validate every action/pose contract, group episodes
by collector/session and obtain environment metadata before claiming unseen
environments. TFRecords lack timestamps: horizons remain ordinal steps until
cadence is independently verified. Actions must have typed physical coordinates,
not only labels such as listen/look. Evaluate consequence prediction, action
ablations and a same-capacity policy against direct-observation/previous-command
baselines. Physical deployment is outside this recorded-data experiment.

Collect new LibriSpeech dev-clean audio plus pinned real phone alignments for
fresh-speaker validation. Exclude every previously exposed participant from
new confirmation roles before fitting, including all former tests. Current
100h train speech and AMI can be reused as training sources, with their actual
exposure declared. Broader paired AMI acquisition must group connected speaker
sessions, include more independent events, and distinguish owner annotations
from audible microphone crosstalk. No final case is dropped after scoring.

Proposed confirmatory axes: SLWM versus matched Transformer, shared versus
specialist representations, cross-source/held-task transfer, shuffled pairs,
wrong-time views and action removal/shuffling. Keep raw-signal and persistence
baselines. Match and report target exposure, total compute, per-task versus
aggregate specialist parameters separately. A collection of narrow heads cannot
by itself establish a general world representation. If procedural dynamics are
added for closed-loop counterfactual tests, label simulated versus recorded data
and hold out dynamics families/parameters; do not count them as real-data hours.

New source archives total about31GB before decoded caches; source_plan.json
pins exact sizes/checksums/generations. Preserve old artifacts. Verified original
archives may be mirrored to the Mac to keep GPU-machine storage for prepared
data; delete only task-owned redundant transport copies after hash verification.
