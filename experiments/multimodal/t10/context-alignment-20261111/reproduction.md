# Reproducing the comparisons

Use the configuration, public source inventory and model/optimizer receipts to identify a run. Seeds 1001 and 1002 trained their context cores on v5e-1 assignments and seed 1003 on the 3090. Every architecture, length and optimizer schedule is paired within a seed. The second stage starts from its corresponding variable-context parent and resets the optimizer identically across treatments. [Configuration](../../../../configs/t10/study.json) · [Context receipt table](metrics/context-artifacts.json) · [Grounding receipt table](metrics/grounding-artifacts.json)

## Public inputs

The STARSS23 training bank is the original G9 seven-room bank, SHA-256 `5aa519fb7782bb6d6e3451d6c144619446c6541418d4e7a9bc39c1faebca3089`. The G9 real-data preparation script builds it from the pinned FOA, video and metadata archives. The HoloAssist input directory takes the public `data-annotation-trainval-v1_1.json` as `labels.json` and `data-splits-v1_2.zip` as `split.zip`. G10 checks both SHA-256 values before selection. The range index checks each archive's published generation. No raw dataset is required on the Mac. [HoloAssist source downloads](https://holoassist.github.io/) · [Recorded hashes and splits](metrics/data.json)

```bash
export PYTHONPATH=.
for archive in imu video_compress video_pitch_shifted; do
  python scripts/data/g10_holo_index.py "$holo_directory" "$archive"
done
python scripts/data/g10_holo.py "$holo_directory"
for role in train dev familiar novel; do
  python scripts/data/g10_pack.py "$holo_directory" "$role"
done
```

The prepared bank includes session boundaries and the future correctness labels as a separate array. The model training sampler selects only the whitelisted observed signals. Correctness annotations are used by frozen readouts. The public train/validation/test role in the upstream dataset and each G10 role are different concepts. G10 never used the upstream official test set for a labeled benchmark claim.

## One matched seed

The examples assume locally mounted storage and a working JAX backend. Both full model weights and optimizer moments are committed atomically at the registered update counts. The controls use the same commands with the other registered seed, core or treatment names.

```bash
python scripts/train/g10_fit.py "$starss_train" "$runs" \
  --seeds 1001 --cores hybrid gpt2 gated --regimes short variable long
python scripts/train/g10_align_fit.py "$starss_train" "$holo_directory/holo-train.npz" \
  "$runs" "$runs" --seeds 1001 --cores hybrid gpt2 --modes paired reconstruct shifted
python scripts/train/g10_align_fit.py "$starss_train" "$starss_train" \
  "$runs" "$runs" --seeds 1001 --cores hybrid gpt2 --modes rooms
PYTHONPATH=. python evals/g10_control.py \
  "$runs/context-hybrid-variable-s1001/step-05120" --dest "$scores/control"
```

The locked scientific release includes count-checked stages, source-level uncertainty, transient-command interventions, original-range sensor forecasts, error prediction probabilities and three compact runnable models. The source-preparation implementation requires NumPy, requests, soundfile and ffmpeg. Frozen outcome readouts additionally use SciPy. Hardware time includes compilation, data staging and artifact transfers where specified. A cloud assignment disappeared before the larger HoloAssist bank was delivered. Its immutable context and room-control hashes were verified, and the replacement lanes checked the bank and parents before post-training.
