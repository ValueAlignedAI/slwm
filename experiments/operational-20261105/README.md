# Verified SLWM operational research delivery

Completed2026-11-06. All7,843 manifest-listed files match between Fedora and
the local repository:51,448,669,335 bytes (51.45GB). `local_verification.json`
records the manifest SHA256 and verification result. `scope.json` lists14
artifact roots. `environment.json` records the measured Fedora toolchain.
The methods/result commit is9cb9f05; the following delivery commit adds only
registry/provenance metadata and this index.

The archive retains original full states, optimizers/RNG, validation-selected
and final weights, sources, failed pilots, metrics, every output, codecs and the
frozen Vocos renderer. Raw/prepared training corpora remain on Fedora; their
provenance metadata is mirrored. The bounded two-episode DROID source preview
is included and is not a trained robotics dataset/model.

From the repository root, verify the delivered bytes without PyTorch:

```bash
python3 -m training.artifact_manifest verify --manifest experiments/operational-20261105/artifacts_manifest.json
```

Key entries:

- [Complete findings](../../docs/research/operational_findings_20261106.md).
- [All248 long cases and failures](../multimodal/t2/broad-stream-20261105/watch.html).
- [World/actor inference exports](../multimodal/t2/broad-stream-20261105/inference_bundle/README.md).
- [Separate language exports](../text/t1/extended-throughput-20261105/inference_exports/README.md).
- [Recorded-input sensing replay](../multimodal/t2/broad-stream-20261105/live_demo/README.md).
- [World/model/media audit](../multimodal/t2/broad-stream-20261105/artifact_audit.json).

There are22 primary exports: ten world fields, four separate124/125M language
models and eight sensing actors. All166 regression tests and the actual-state
integrity checks pass. All three broader speech/sensing capability gates fail;
these are research artifacts, including unsuccessful generators. Native video
is64×64 face motion with supplied appearance. Source sound is retained for
audio→video; video→audio and text-conditioned sound are generated through Vocos.

The recorded sensing example makes30 decisions after a bootstrap without
another request. Its input excerpt is42.325s. It passes CPU arrival-order and
weight-invariance checks, with zero deadline misses. It prepares file features
before replay and controls recorded sensor visibility; it does not implement
live capture or physical movement. Use your installed Python environment in
place of the Fedora interpreter recorded as the first element of `command.json`,
and choose a fresh output directory when repeating a CLI run.

For example, with research-stream dependencies installed, from the repository:

```bash
python3 -m exploration.stream_sensing_infer \
  --world experiments/multimodal/t2/broad-stream-20261105/inference_bundle/EXP-T2-801.pt \
  --actor experiments/multimodal/t2/broad-stream-20261105/inference_bundle/EXP-P1-101.pt \
  --codec-dir experiments/multimodal/t2/broad-stream-20261105/inference_bundle \
  --audio experiments/multimodal/t2/broad-stream-20261105/live_demo/recorded_input.wav \
  --video experiments/multimodal/t2/broad-stream-20261105/live_demo/recorded_input.mp4 \
  --goal both --device cpu --pace --output /tmp/slwm_recorded_replay
```

Matched500M-token language continuation is an estimated10.77 additional GPU
hours for both architectures and both seeds, excluding saving/final evaluation.
It is a proposed next experiment, not a running job. No task-owned GPU training
or telemetry process is left running after this delivery.
