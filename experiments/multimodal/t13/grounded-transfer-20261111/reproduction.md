# Reproduce the G13 research series

G13 starts from verified G12 main `736d1f0`. The physical and natural-speech studies were registered before fitting. The first dependent revisions describe subsequent decisions. The original measurement audit prompted a detector repair. The first two model cohorts motivated a prospective action hypothesis. After that test passed, a new acquisition tested the stronger raw reference. The raw-only correction and untrained feature controls are explicitly post-hoc diagnostics. [Original protocol](protocol.md) · [Measurement repair](repair-and-replication.md) · [Prospective action replication](prospective-action-replication.md) · [Direct readout](direct-action-readout.md)

## Inputs and separation

The reference implementation uses Fedora's mounted data SSD at `/mnt/slwm-data/SLWM`, the separately mounted checkpoint HDDs and a single RTX 3090. Use Python 3.12, NumPy, SciPy, OpenCV, Pillow, JAX, the protobuf `Example` type from `tfrecord 1.14.6`, and system `libcrc32c`. The acquisition code checks immutable Google object generations and provider MD5. Each TFRecord header and payload gets a CRC32C check. Frame identities and parent hashes group overlapping hindsight segments. Keep the release's original group partitions for any numerical reproduction. [Acquisition manifest and bank digests](metrics/data.json) · [Per-episode shard offsets and split](metrics/robot-source-index.jsonl.gz)

The first sixteen shards supply train/development/held banks. The next three groups of four shards are sequential external studies. With only the original training bank, fit the linear identification coefficients, raw likelihood/prior, nonlinear basis, role model, recognition temperatures and the readouts. Published examples and prepared banks keep future outcomes distinct from current observation features. The action forecast never consumes the four proposed controls. The active dynamics head accepts only the prefix of controls appropriate for each predicted horizon.

```bash
export PYTHONPATH=.
for source in robot extra prospective readout speech; do python scripts/data/g13_acquire.py "$source"; done
python scripts/data/g13_robot.py
for src in robot-extra robot-prospective robot-readout; do
  python scripts/data/g13_robot.py --external --root "/mnt/slwm-data/SLWM/g13/data/$src"
done
python scripts/data/g13_directions.py
```

Set the path to the pinned protobuf module so Python can import `example_pb2`. The G10–G12 natural media preparation must exist for the HoloAssist studies. Prepare source groups, fresh AV/IMU, timestamped ResNet-18 and native Whisper-small.en using `g13_holo.py`, `g12_media.py`, and `g11_speech.py`. Build the component sweep with `g13_holo.py --component`, the pinned MiniLM snapshot and `g13_components.py`. The final natural verb calibration uses `--quarantine` for the 64 component source names. [Lineage](component-lineage.md) · [Verbs](metrics/holo-clean.json) · [Components](metrics/component-results.json)

## Training and evaluation

Use the normal CUDA wrapper `bash scripts/train/g4_cuda.sh` on this Fedora setup. The GPT-2-style processor has ordinary dense softmax attention. The spatial candidate uses decaying associative memory and geometric weighting. A dense processor with the same geometry bias isolates the spatial input from the attention-kernel change. All cores have identical trainable shapes, objectives, training data and candidate plan interfaces in a matched cell. Neural convergence is evaluated at fixed exposure and by separately chosen development checkpoints. The primary robot schedule samples 128 examples for 1,024 updates; 35,526 training windows overlap within 583 independent parent groups. The source-count and sample-exposure numbers serve different purposes. [Core and head implementation](../../../../models/g13_world.py) · [Pilot selection](metrics/robot-pilot-lock.json) · [108 frozen confirmatory choices](metrics/model-development-lock.json)

```bash
python -m scripts.train.g13_robot_fit pilot
python -m scripts.train.g13_robot_fit confirm
python scripts/data/g13_repair.py select --ratings experiments/multimodal/t13/grounded-transfer-20261111/audit/ratings.json
for src in robot robot-extra robot-prospective robot-readout; do
  python scripts/data/g13_repair.py build --root "/mnt/slwm-data/SLWM/g13/data/$src"
done
python -m scripts.train.g13_robot_fit corrected
python -m evals.g13_predict develop
python -m evals.g13_predict confirm
python -m scripts.train.g13_robot_fit newseeds
python -m evals.g13_prospective develop
python -m evals.g13_prospective test
for role in train dev; do python -m evals.g13_direct fields --role "$role"; done
python -m evals.g13_direct develop
python -m evals.g13_direct fields --role last
python -m evals.g13_direct test
```

The direct correction uses five-fold out-of-group raw predictions during training. Fields are frozen at their registered checkpoints. Development chooses squared versus robust fitting, ridge penalty and residual strength. Any gain attributed to the final shared correction is paired against the development-selected robust raw fit. Per-window data allow the comparisons to be recomputed. [Third cohort](metrics/prospective-actions.npz) · [Fourth cohort](metrics/direct-actions.npz) · [Raw attribution controls](metrics/readout-controls.json)

The released four-word edge uses `g13_speech.py train --portable` followed by `seal --portable`. The six original convolutional fits remain in the accounting. Use `g13_voice.py develop --portable` and `confirm --portable` for the controlled voice comparison. The new noun models use `g13_component_fit.py develop` and `test`. The normal release checks cover CPU/CUDA parity, actual microphone substitution, held model hashes, chronological single-clock execution, and independently read copies on both HDDs. [Recognizer repair](speech-portability.md) · [Detailed compute receipt](metrics/compute.json) · [Custody](metrics/release.json)
