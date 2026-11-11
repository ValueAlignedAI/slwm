# Reproducing G11

G11 measures added predictive information in an actually recorded prospective speech cue. It has two independent optimizations per device split: choosing model parameters using training recordings, and an explicitly labeled secondary checkpoint choice using development data. The primary test always uses update 1,024. Fresh test actions and depth never enter the training sampler. [Registered design](protocol.md) · [Checked source groups](metrics/data.json) · [Held decision identities](metrics/held-decisions.jsonl.gz)

## Source preparation

Use the public HoloAssist annotations, split and AV/IMU indexes from G10. The original pitch-shifted AV members contain the microphone channel. The compressed video members alone have no sound. Additional archive indexes for depth, hands, eyes and calibration use immutable byte-range checks and handle extended tar path names. The project projection code and PSI range semantics are linked in the measurement card. The smaller recognizer is Whisper tiny.en at revision `87c7102498dcde7456f24cfd30239ca606ed9063`. The stronger substitution uses small.en at `e8727524f962ee844a7319d92be39ac1bd25655a`. The source data agreement accompanies the examples and blind review sheets. [HoloAssist](https://holoassist.github.io/) · [G10 data setup](../../t10/context-alignment-20261111/reproduction.md)

On the research workstation the G10 inputs are under `/mnt/slwm-data/SLWM/g10/data/holo` and the G11 working data are under `/mnt/slwm-data/SLWM/g11/data`. The following commands run from the repository root with JAX, NumPy, scipy, Pillow, requests, ffmpeg and the documented speech-recognition dependencies.

```bash
export PYTHONPATH=.
for archive in ahat_depth eyes hands cam_info; do
  python scripts/data/g11_index.py "$g11_data" "$archive"
done
python scripts/data/g11_events.py "$g11_data" --fresh
python scripts/data/g11_speech.py "$g11_data"
python scripts/data/g11_speech.py "$g11_data" --recognizer small
python scripts/data/g11_geometry.py "$g11_data" --mode main
python scripts/data/g11_banks.py "$g11_data" base
python scripts/data/g11_banks.py "$g11_data" effects
```

The word file and all synchronized physical input features come from information available by the instruction cutoff. Effect quality flags are loss/scoring masks only. Every published physical-eligibility grade can be resolved to its original sample. [Numbers, units and controls](metrics/measurement.json) · [Blinded center review](audit/README.md)

## One primary fit and its interventions

Use unquantized verified G10 variable-context parent stages as the parent directory. The adaptation code writes deltas and optimizer state. Loading a scientific checkpoint requires its recorded parent hash. Public inference packages include matching compact bases and adapters.

```bash
for phase in action joint; do
  python scripts/train/g11_fit.py "$g11_data/banks" "$g10_parents" "$runs" "$phase" \
    --folds A --seeds 1001 --cores hybrid --treatments paired
done
python evals/g11_baselines.py "$g11_data/banks" "$scores" joint
```

Exact data exposure, source masks and optimizer schedule are shared across the primary treatment cells of a seed. The larger comparison instead uses two equal-exposure G9 parent sizes. Its four newly trained gated parents and the eight restored hybrid/dense identities have separate receipts. Training and inference stage digests were independently read from both checkpoint disks. Data processing and the released examples contain no private recordings. [Primary identities](metrics/primary-artifacts.json) · [Scale identities](metrics/scale-artifacts.json) · [Measured compute](metrics/hardware.json)
