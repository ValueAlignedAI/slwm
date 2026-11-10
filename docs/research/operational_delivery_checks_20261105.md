# Operational artifact delivery checks

Recorded before B1/P1 final scores. These checks exercise the portable selected
exports through their documented ordinary-file interfaces. They introduce no
fitted weights, thresholds, quality gates or test-selected examples.

`python -m training.operational_bundle_check` runs on CPU after
`export_complete.json` exists. It uses the first recorded AMI test participant,
fixed 80-second start and 40.96-second audio/video generation intervals; the
first already registered long text passage; and a technical eight-second repair
from the first Libri test-clean entry. It also executes a paced 30-decision
actor replay on 42.325 seconds of the original files and short continuations
from both seed81 language exports. All commands, logs, proposals and media stay
in `broad-stream-20261105/bundle_file_checks`.

Assertions check minimum long-media duration, exact frame/sample clocks,
video-to-audio source-sound exclusion, exact observed repair PCM, explicit mel
context conservation, exported-world/actor binding, recorded-arrival order,
automatic decision count, weight invariance and actual language-token exposure.
The file features are prepared before replay; this does not establish online
capture or physical control. Poor generated content still passes an interface
check when its files and provenance are valid. The separate frozen capability
gates and every original evaluation case remain authoritative.
