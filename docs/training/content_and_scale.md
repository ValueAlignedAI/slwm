# Completed content/timing and early text-scale experiments

These are two separate trained artifact families. T2-C1 tests closed-vocabulary
speech fields and admission of video; T1-S2 tests early causal text learning at
GPT-2-small parameter scale. Neither is a unified 124M multimodal world model.

## Results and saved artifacts

| Study | Artifact directory | Current evidence |
|---|---|---|
| T2-C1 | `experiments/multimodal/t2/content-sync-20261105/` | `report_v2.md`, `summary_v2.json`, `word_metric_correction.json`, `audit.json`, `listen.html` |
| T1-S2 | `experiments/text/t1/scale-20261105/` | `report.md`, `summary.json`, `compute_diagnostic.md`, `audit.json`, all raw `samples.json` |

Directories are retained at `/Users/db/dev/slwm/` locally and
`/home/db/dev/slwm/` on `db@192.168.1.200`. Every run keeps `best.pt` (including
optimizer/RNG state), `last.pt`, selected step, curves, config and source hashes.
The portable `bundle/` directories include their own manifests and instructions.
`delivery_manifest.json` hashes the retained non-log evidence. Large binary
checkpoints, data arrays and generated media are excluded from Git; compact
reports, metrics and frozen code snapshots are committed. These files are in
both machines' experiment trees, not downloaded automatically from GitHub.

T2 revision 2 excludes both `sil` and `sp` from lexical-word scores, correcting
19 short-pause frames in the primary gap. Original evaluation files remain
unchanged. Ridge coefficients are reused exactly; weights, calibration, spectra
and waveforms are unchanged. Use the revision-2 report for current word claims.
`word_evaluation_v2/source_snapshot/` records that correction separately from
the frozen original training and evaluation sources.

## Data and accounting

GRID retains 12,704 training clips from 16 people (10.59 hours), 395 selection
clips/four other people, 278 calibration clips/three other people and 749 primary
test clips/eight new people. Sentence-hash buckets also reserve new complete
sentence combinations. Vocabulary is training-only: 51 words plus two pause
codes. `test_seen` is an old diagnostic name for training-eligible hash buckets;
its exact sentences also have zero actual overlap with training. It must not be
reported as an empirically seen-sentence test. People 33/34 remain unopened.

The source alarm, raw CRC/frame review and exactly 90 excluded truncated sources
are documented in `docs/research/grid_source_quality_review_20261105.md`. No
selection/calibration/primary-test clip is among those exclusions. Decoding uses
original embedded audio with video PTS alignment, a fixed training-chosen crop,
25Hz mouth frames and 100Hz audio features. Prior CREMA codec statistics remain
fixed. Prepared arrays/raw recordings stay in
`artifacts/t2_audio_visual/grid_content_v1/` on Fedora; small metadata copies
are included in the experiment directory.

Speech bases contain 5,133,942 neural weights. Audio candidates contain 6,428,512
(1,294,570 newly trained); video contains 7,713,820 (1,285,308 newly trained),
including the frozen strong audio fallback. Fitted codecs are counted separately.
Video negatives add training compute. Both base controls receive the explicit
timed-text auxiliary loss; the matched ablation removes added unit CE only.

FineWeb-Edu revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` yields
2,600,000,000 training / 14,573,946 validation / 10,013,422 test GPT-2 tokens.
Source parquet and token files have verified SHA256 values. Registrable domains
are disjoint; exact normalized documents, long prefixes and canonical URLs are
deduplicated globally. Near-duplicate freedom is not guaranteed. The corpus is
English text, not a code or audiovisual corpus. Keep its ODC-By attribution and
GRID's research-use conditions; repository MIT licensing does not replace data
permissions.

GPT-2 / SLWM strict text totals are 124,439,808 / 125,131,008. Each of four runs
sees 16,777,216 tokens (32×512×1,024 optimizer steps). The prepared corpus is
large enough for a longer controlled trial; this first stage consumes under 1%
and is severely undertrained. Each validation/test score covers a fixed
131,072-token sample of its reserved stream, not the complete corpus. Full corpus arrays/raw parquet remain in
`artifacts/text/fineweb_edu_scale_v1/` on Fedora. The local metadata and model
bundles support inference without copying that corpus.

## Reproduction on Fedora

The measured runtime is `/home/db/venvs/vllm/bin/python`; research dependencies
are provided through `.research-venv/lib/python3.12/site-packages`. Install the
`research-data` optional dependencies from `pyproject.toml` when rebuilding.
Use `flock /tmp/slwm_research_gpu.lock` to serialize GPU studies.

```bash
python -m training.research_sources --kind grid --root artifacts/t2_audio_visual/grid_content_v1
python -m training.grid_content_data
bash scripts/train/run_content_sync_fedora.sh
bash scripts/train/finish_content_sync_fedora.sh
python -m evals.content_sync_word_correction
python -m training.content_sync_report_v2
```

Those last two commands are an explicit scoring revision, not retraining.
Existing source guards prevent silently rerunning changed code over frozen
artifacts. Text preparation/training/evaluation uses:

```bash
python -m training.research_sources --kind text --root artifacts/text/fineweb_edu_scale_v1
python -m training.prepare_scale_text
python -m training.scale_text_train --stage train
python -m training.scale_text_train --stage evaluate
python -m training.scale_text_report
python -m evals.scale_compute_diagnostic
```

The source commands create pinned lists and download the research corpus. Do not replace
immutable corpora or overwrite completed runs to try another configuration.
Create a new registered study/artifact directory instead. Final package checks:

```bash
bash scripts/train/package_content_scale_fedora.sh
```

This verifies text data/source/checkpoint integrity, runs both families through
CPU command-line inference, reruns the nine relevant contract tests and builds
the final portable bundles/hash inventory. The full repository suite passes
133 tests. Original speech integrity checks are retained in `audit.json`.

## Portable ordinary-file inference

From the repository root, use a bundle without a training corpus or recognizer:

```bash
python -m exploration.content_sync_infer \
  --checkpoint experiments/multimodal/t2/content-sync-20261105/bundle/video.pt \
  --codec-dir experiments/multimodal/t2/content-sync-20261105/bundle \
  --calibration experiments/multimodal/t2/content-sync-20261105/bundle/calibration.json \
  --media recorded_grid.mpg --gaps 1.32:1.64 --device cpu --output /tmp/slwm-repair
```

`--media` uses original GRID stream alignment and the fixed camera crop. For
audio-only inputs use `--audio recording.wav` and the audio checkpoint, or the
video model's exact audio fallback with its calibration. Gaps must fit the
supported context and be at most 480ms. All gaps are erased before transforms;
visible PCM is preserved exactly on the decoded video-aligned timeline. Stream
trim/pad and admission decisions are recorded in `inference.json`. The fixed
float16 feature rounding matches preparation/calibration. Other camera views
remain unvalidated.

The `content_base.pt` checkpoint supports a separately requested `--timed-text`
JSON list of `{start,end,word}` records. This deliberately supplied-label mode
is a diagnostic, never the practical audio/video result. `--probe-words` exposes
the model's closed-vocabulary context logits, not a free-form transcript.

```bash
python -m exploration.scale_text_infer \
  --checkpoint experiments/text/t1/scale-20261105/bundle/slwm_seed81.pt \
  --prompt "The purpose of a model is" --tokens 64 --device cpu
```

Raw continuations from these short early runs can be repetitive or incoherent.
Only 512-token contexts were trained, despite a 1,024-position table. Speech
recognition remains mixed, and wrong-utterance video can still harm repair.
Consult the current reports before interpreting or reusing either artifact.
