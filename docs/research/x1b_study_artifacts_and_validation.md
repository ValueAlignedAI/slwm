# Run the SLWM study artifacts

These are reference checkpoints for an **unpublished manuscript draft (v0.3)**. The three Hugging Face model repositories contain one fixed checkpoint per wave (seed `2001`), its original configuration and run records, BF16 and INT8 exports, a NumPy loader, and checksums. The paper's headline comparisons summarize three seeds per cohort. A run of one checkpoint is not a reproduction of those aggregate results.

| Study artifact | Experiment | Parameter count | Hugging Face repository |
|---|---:|---:|---|
| SLWM-X1B | Wave 5, doubled sequence exposure | 1,008,454,410 | [`valueai-org/SLWM-X1B`](https://huggingface.co/valueai-org/SLWM-X1B) |
| SLWM-X450M-Memory | Wave 6, full memory and ablation reference | 453,199,938 | [`valueai-org/SLWM-X450M-Memory`](https://huggingface.co/valueai-org/SLWM-X450M-Memory) |
| SLWM-v2-453M | Wave 7, spatial refinement | 453,199,938 | [`valueai-org/SLWM-v2-453M`](https://huggingface.co/valueai-org/SLWM-v2-453M) |

The source entry points and evaluation suites are in this repository. The paper, result figures, and an interactive Wave 4 replay are available at [the companion article](https://diogoantunes.ai/publications/signal-latent-world-models). The replay is an earlier example; it is not a validation run of the three checkpoints above.

## 1. Prepare a Python environment

Use Python 3.11 or newer. From a checkout of this repository:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install 'numpy>=2' 'jax[cpu]' huggingface_hub
python scripts/eval/x1b_study_examples.py --mode self-test
```

The last command uses a small, randomly initialized fixture. It checks that the source code can generate an input and produce a finite latent field of shape `[2, 8, 24, 128]`. It does **not** load a study checkpoint or measure model quality. The source's full evaluations are much more demanding; use an environment sized for the original experimental setup when running them.

### Check an artifact before execution

The script downloads only the selected weights, `config.json`, and `receipt.json` from the public Hugging Face repository. It verifies the recorded SHA-256 and strict total parameter count before an example or evaluation. To point it at a checkpoint directory already on disk, pass `--checkpoint-dir /path/to/checkpoint`.

```sh
python scripts/eval/x1b_study_examples.py \
  --mode example \
  --repo-id valueai-org/SLWM-v2-453M \
  --format int8 \
  --output /tmp/slwm-v2-example.json
```

This runs one deterministic, generated two-episode, eight-tick example through the **trained Wave 7 checkpoint**. The output records the input and latent shapes and confirms that the spatial readout returns finite positions. The Wave 7 example uses the `8 × 8` pure-visual feature grid in `models/x1b_wave7_core.py`. It is a functional example, not a reported validation score.

Use `valueai-org/SLWM-X1B` or `valueai-org/SLWM-X450M-Memory` in `--repo-id` for the corresponding Wave 5 or 6 generated example. Set `--format bf16` when comparing with the original BF16 study evaluation; `int8` selects the smaller portable file. Both formats are decoded to float32 for this script. The INT8 file's size does not represent the runtime memory used for matrix computation.

```sh
python scripts/eval/x1b_study_examples.py \
  --mode example \
  --repo-id valueai-org/SLWM-X1B \
  --format bf16 \
  --seed 20261114 \
  --output /tmp/slwm-x1b-example.json
```

Each model repository also contains a pickle-free NumPy `load_checkpoint.py` and `SHA256SUMS` for inspecting the files without importing the JAX research runtime. The example script above uses the original source loader and checks the run receipt before executing the model.

## 2. Know which inputs are available

The generated synthetic examples and the synthetic control, memory, and camera-fault evaluations are created by the source code from fixed seeds. They do **not** require the recorded media or physical-robot banks.

The recorded-data checks require locally prepared NPZ banks. These banks are **not included in the model repositories**. The source preparation scripts expect a data-root layout under `/mnt/slwm-data/SLWM` and write the following files under `x1b/datasets/`:

| Bank | Available splits | Preparation script | Used for |
|---|---|---|---|
| `x1b_real_bank_v1.npz` | `train`, `dev`, `test` | `scripts/data/x1b_prepare_fedora.py` | Wave 5 audio/video, contact, and text checks |
| `x1b_real_robot_v2.npz` | `train`, `dev`, `test` | `scripts/data/x1b_prepare_real_robot_fedora.py` | Wave 5 recorded xArm6 trajectories |
| `x1b_pymunk_panels_64.npz` | task panels | `scripts/data/x1b_prepare_fedora.py` | Wave 5 thin-barrier and Push-T panels |
| `x1b_wave6_grounded_v1.npz` | `train`, `dev` | `scripts/data/x1b_prepare_wave6_grounded_fedora.py` | Waves 6–7 physical and spoken development checks |
| `x1b_wave6_vault_test.npz` | `vault` | `scripts/data/x1b_prepare_wave6_grounded_fedora.py` | Waves 6–7 held-out final confirmation |

The preparation scripts specify their input archive layout. The real multimodal bank uses GreatestHits, STARSS23, PyMunk Push-T, and FineWeb-Edu material. The robot and spoken banks additionally require the source physical trajectories and spoken-utterance inputs described in those scripts. Obtain the underlying data through its own permitted source and follow its terms. The model repositories contain the trained checkpoints and metadata; they do not redistribute those recordings or data banks.

Each generated bank has a matching `*.card.json` with a SHA-256 and split counts. The banks and cards must be made from the same source snapshot to reproduce its recorded measurements. A new bank made with other recordings can still test the code path, but its score should be reported as a new evaluation.

### Inspect a development bank and its card

```sh
python scripts/eval/x1b_study_examples.py \
  --mode inspect-data \
  --bank /mnt/slwm-data/SLWM/x1b/datasets/x1b_wave6_grounded_v1.npz \
  --card /mnt/slwm-data/SLWM/x1b/datasets/x1b_wave6_grounded_v1.card.json \
  --split dev
```

This prints the split's array names, number of episodes, video shape, and whether the card checksum matches. It does not output recordings. The `--split test` option applies to the Wave 5 banks. Wave 6 and 7 use a separate `vault` file for held-out confirmation. Use development results while choosing settings; treat the `test`/`vault` splits as the final check of a frozen procedure.

## 3. Run recorded development checks

The following commands validate the supplied checkpoint against the **development** split of a bank you already have. The results are written to new JSON files and include the bank fingerprint, wave, seed, and checkpoint format.

**Wave 5, recorded audiovisual and physical robot:**

```sh
python scripts/eval/x1b_study_examples.py \
  --mode validate --repo-id valueai-org/SLWM-X1B --format bf16 \
  --split dev \
  --bank /mnt/slwm-data/SLWM/x1b/datasets/x1b_real_bank_v1.npz \
  --card /mnt/slwm-data/SLWM/x1b/datasets/x1b_real_bank_v1.card.json \
  --robot-bank /mnt/slwm-data/SLWM/x1b/datasets/x1b_real_robot_v2.npz \
  --robot-card /mnt/slwm-data/SLWM/x1b/datasets/x1b_real_robot_v2.card.json \
  --output /tmp/slwm-wave5-dev.json
```

**Wave 6, synthetic panels plus recorded spoken/robot development data:**

```sh
python scripts/eval/x1b_study_examples.py \
  --mode validate --repo-id valueai-org/SLWM-X450M-Memory --format bf16 \
  --split dev \
  --bank /mnt/slwm-data/SLWM/x1b/datasets/x1b_wave6_grounded_v1.npz \
  --card /mnt/slwm-data/SLWM/x1b/datasets/x1b_wave6_grounded_v1.card.json \
  --output /tmp/slwm-wave6-dev.json
```

**Wave 7, the same development bank with the spatial readout:**

```sh
python scripts/eval/x1b_study_examples.py \
  --mode validate --repo-id valueai-org/SLWM-v2-453M --format bf16 \
  --split dev \
  --bank /mnt/slwm-data/SLWM/x1b/datasets/x1b_wave6_grounded_v1.npz \
  --card /mnt/slwm-data/SLWM/x1b/datasets/x1b_wave6_grounded_v1.card.json \
  --output /tmp/slwm-wave7-dev.json
```

For Waves 6–7, the default validation command runs the original evaluators' smaller `smoke=True` setting for their synthetic panels and uses the development bank if supplied. It still includes a `T = 160` goal-switch path and can be computationally costly. `--full` selects the original sample counts and horizons. If no bank is supplied, the synthetic panels run and the recorded-data result is marked unavailable. The Wave 5 development command calls its recorded-data panels directly because the original full Wave 5 suite reads the `test` split by design.

## 4. Run held-out confirmation when the procedure is frozen

The original Wave 5 full suite reads the `test` splits of both recorded banks and the PyMunk panels:

```sh
python scripts/eval/x1b_study_examples.py \
  --mode validate --repo-id valueai-org/SLWM-X1B --format bf16 \
  --split test --full \
  --bank /mnt/slwm-data/SLWM/x1b/datasets/x1b_real_bank_v1.npz \
  --card /mnt/slwm-data/SLWM/x1b/datasets/x1b_real_bank_v1.card.json \
  --robot-bank /mnt/slwm-data/SLWM/x1b/datasets/x1b_real_robot_v2.npz \
  --robot-card /mnt/slwm-data/SLWM/x1b/datasets/x1b_real_robot_v2.card.json \
  --panels /mnt/slwm-data/SLWM/x1b/datasets/x1b_pymunk_panels_64.npz \
  --panels-card /mnt/slwm-data/SLWM/x1b/datasets/x1b_pymunk_panels_64.card.json \
  --output /tmp/slwm-wave5-test.json
```

Waves 6–7 use the `vault` split from a separate NPZ. For example:

```sh
python scripts/eval/x1b_study_examples.py \
  --mode validate --repo-id valueai-org/SLWM-v2-453M --format bf16 \
  --split vault --full \
  --bank /mnt/slwm-data/SLWM/x1b/datasets/x1b_wave6_vault_test.npz \
  --card /mnt/slwm-data/SLWM/x1b/datasets/x1b_wave6_vault_test.card.json \
  --output /tmp/slwm-wave7-vault.json
```

Replace the model with `valueai-org/SLWM-X450M-Memory` to run its Wave 6 evaluator against the same vault bank. These commands call the original `run_full_evaluation`, `run_wave6_evaluation`, or `run_wave7_evaluation` functions, as appropriate. They do not train or alter the checkpoint weights.

## 5. Interpret what was run

The wrapper checks the selected run's `config.json`, `receipt.json`, weight SHA-256, and strict parameter count. Each JSON report records its checkpoint format and split. Original `eval_metrics.json` files in the model repositories contain the archived metrics for the checkpoint. The manuscript's three-seed results also depend on the other seeds and its locked evaluation protocol. A synthetic smoke run, a new data bank, or an INT8 rerun should be identified as such when sharing results.

The Wave 5, Wave 6, and Wave 7 checkpoints do not share a single interchangeable data setting. Wave 5 uses object coordinates in proprioception; Wave 6 removes that shortcut and exposes a visual localization weakness; Wave 7 adds spatial visual readouts while retaining a difficult associative-recall trade-off. Keep those differences in any comparison.

The original paper PDF is preserved unchanged. This guide and the Hub model cards describe it as an **unpublished manuscript draft**, and the checkpoint repositories as **study artifacts for reference**.
