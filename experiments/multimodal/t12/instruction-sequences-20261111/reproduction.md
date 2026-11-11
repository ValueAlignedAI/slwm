# Reproduce the G12 study

The protocol and source grouping are registered at `d90a1d9`, the width extension at `3e0a57a` and the development-derived early joint comparison at `470a11d`. The early extension was locked before inspecting G12 held results. State recalibration is a post-hoc diagnostic whose parameters use development data. All trainable arrays were optimized from scratch on the new task. Public codec parameters are frozen and shared across all processors. [Primary protocol](protocol.md) · [Training and controls](training.md) · [Width](width.md) · [Early joint](development-followup.md)

## Data

The Fedora reproduction uses the original public G10 HoloAssist labels, split, media and IMU archives and the G11 earlier-instruction matching, speech and recording metadata. Use the previous releases to prepare the inputs. The 48 extra original video and sensor-feature hashes are in the data inventory. The split excludes older G10/G11 recording groups for the fresh cohort. Duplicate part names are grouped. Cohort sizes, training vocabularies, all 8 bank digests, components and missingness are public. [G10 preparation](../../t10/context-alignment-20261111/report.md) · [G11 preparation](../../t11/instructions-physical-effects-20261111/reproduction.md) · [G12 bank and source hashes](metrics/data.json)

Run the following entry points from the repository root with `PYTHONPATH=.`. The reference layout is the mounted Fedora research SSD at `/mnt/slwm-data/SLWM` and the two separately mounted checkpoint HDDs. Data preparation needs Python 3.12, NumPy, Pillow, requests, soundfile and FFmpeg. The visual lane uses PyTorch, torchvision, NVIDIA PyNvVideoCodec and ImageNet ResNet-18 weights. Whisper-small.en uses the registered Hugging Face snapshot. Keep its local files at `g12/data/whisper-small` or link that path to the G11 snapshot.

```bash
g12=experiments/multimodal/t12/instruction-sequences-20261111
python scripts/data/g12_events.py
python scripts/data/g12_media.py vision
python scripts/data/g11_speech.py /mnt/slwm-data/SLWM/g12/data --recognizer small
python scripts/data/g12_media.py pairs
python scripts/data/g12_states.py "$g12/audit/blind-ratings.json"
python scripts/data/g12_banks.py
```

The NVIDIA decoder provides each source presentation timestamp. A visual observation must precede its clock by at least 110 ms. Audio and IMU use completed samples only. The public input banks have no future verbs or state indicators in the predictive feature arrays. Target arrays are separate. The near-duplicate cue and future guard tests run in the repository suite.

## Training and evaluation

With the prepared banks, a CUDA-enabled or TPU-enabled JAX installation runs the model. The reference sweep batches 96 examples, compiles blocks of 16 optimizer updates, and resumes from verified optimizer generations. The 8-second scene history and the completed speech produce a field that forecasts all three positions without teacher forcing. The principal processor kernels are dense causal attention, a gated recurrence, a damped rotary recurrence, and normalized decaying outer-product memory. All processors have the same parameter tensors. Their compute costs differ. Checkpoint receipts include development curves, time including first compilation and post-warmup throughput. [Processor code](../../../../models/g12_world.py) · [Checkpoint hashes](metrics/checkpoints.json)

```bash
for stage in warm joint eval; do python -m scripts.train.g12_fit "$stage"; done
for stage in warm joint eval; do python -m scripts.train.g12_fit "$stage" --scale; done
python -m scripts.train.g12_early
out=/mnt/slwm-data/SLWM/g12/release/metrics
python -m evals.g12_compare "$out"
python -m evals.g12_width "$out"
python -m evals.g12_early "$out"
python -m evals.g12_calibration "$out"
```

The confirmed benchmark comprises 48 primary fits, 16 wider fits and 48 early joint fits. The paired prediction archives contain per-position winning class, probability of the observed verb and the state probability for the registered interventions. Whole-recording bootstrap draws are reused for paired comparisons and equal-weight the two held families in each fold. Seeds are reported separately. Exact action accuracy requires a correct verb in all three positions. Late instructions, missing state, previous annotated actions, raw pooled readouts, state prevalence and lexical controls are identified in the full results. [Primary paired scores](metrics/paired-predictions.npz) · [Width scores](metrics/width-predictions.npz) · [Early scores](metrics/early-predictions.npz)

## Replay and custody

The independent live fixture takes the first 180 seconds, or the available duration, in one hash-selected fresh recording per family. It uses all completed prospective directives in the selected interval. A normal clock tick can run without a directive. Source selection and clock construction do not use later action annotations. Offline quiet-clock evaluation uses those annotations separately. Model weights, labeled outcomes, example speech and full blind-review sheets accompany the release. [Inference entry point](inference/README.md) · [Examples](examples/README.md) · [Audit](audit/README.md)

The four-processor width wave was offered to the native v6e after the input banks were ready. Six allocations across the study returned a timeout or service-unavailable response. The official CLI metered only this study’s own named assignments. Both native ledgers were reconciled with no owned runtime left active. The RTX 3090 completed all fitting. Neither Mac storage nor the Fedora system volume held research media or training checkpoints. The Mac publication checkout contains the compact public inference packages. [Compute](metrics/compute.json) · [Release integrity](metrics/release.json)
