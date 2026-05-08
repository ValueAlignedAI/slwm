#!/usr/bin/env bash
set -euo pipefail

# Sprint T1 EXP-T1-402 SLWM text-only 124M MPS run helper.
#
# Default behavior is a non-destructive preflight. The long 122,070-step
# training job only starts when --run is passed explicitly.
#
# Usage:
#   scripts/train/run_t1_exp_t1_402_slwm_mps.sh
#   scripts/train/run_t1_exp_t1_402_slwm_mps.sh --write-config-only
#   scripts/train/run_t1_exp_t1_402_slwm_mps.sh --run
#   scripts/train/run_t1_exp_t1_402_slwm_mps.sh --run --allow-existing-artifacts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIG_REL="configs/t1/slwm_text_torch_124m_1b_m4max.json"
TEMPLATE_REL="configs/t1/slwm_text_torch_124m_larger_local.json"
CORPUS_REL="artifacts/t1_text_code/gpt2_bpe_1b_v0"
EXP_ID="EXP-T1-402"
ARTIFACT_REL="experiments/text/t1/${EXP_ID}"

RUN_TRAINING=0
WRITE_CONFIG_ONLY=0
FORCE_CONFIG=0
ALLOW_EXISTING_ARTIFACTS=0

usage() {
  "$PYTHON_BIN" - <<'PY'
print("""Usage: scripts/train/run_t1_exp_t1_402_slwm_mps.sh [options]

Options:
  --run                       Launch python -m training.t1_torch_text after preflight.
  --write-config-only         Generate/refresh the EXP-T1-402 config and exit.
  --force-config              Overwrite the config if it differs from the generated config.
  --allow-existing-artifacts  Allow --run when experiments/text/t1/EXP-T1-402 already has files.
  -h, --help                  Show this help.

Default: generate the config if needed, run preflight checks, and do not start training.
""")
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      RUN_TRAINING=1
      ;;
    --write-config-only)
      WRITE_CONFIG_ONLY=1
      ;;
    --force-config)
      FORCE_CONFIG=1
      ;;
    --allow-existing-artifacts)
      ALLOW_EXISTING_ARTIFACTS=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      usage
      exit 2
      ;;
  esac
  shift
done

cd "$ROOT_DIR"

export SLWM_T1_CONFIG_REL="$CONFIG_REL"
export SLWM_T1_TEMPLATE_REL="$TEMPLATE_REL"
export SLWM_T1_CORPUS_REL="$CORPUS_REL"
export SLWM_T1_EXP_ID="$EXP_ID"
export SLWM_T1_ARTIFACT_REL="$ARTIFACT_REL"
export SLWM_T1_FORCE_CONFIG="$FORCE_CONFIG"
export SLWM_T1_WRITE_CONFIG_ONLY="$WRITE_CONFIG_ONLY"
export SLWM_T1_RUN_TRAINING="$RUN_TRAINING"
export SLWM_T1_ALLOW_EXISTING_ARTIFACTS="$ALLOW_EXISTING_ARTIFACTS"

"$PYTHON_BIN" - <<'PY'
import importlib.util
import json
import os
import sys
from pathlib import Path

root = Path.cwd()
config_rel = Path(os.environ["SLWM_T1_CONFIG_REL"])
template_rel = Path(os.environ["SLWM_T1_TEMPLATE_REL"])
corpus_rel = Path(os.environ["SLWM_T1_CORPUS_REL"])
artifact_rel = Path(os.environ["SLWM_T1_ARTIFACT_REL"])
exp_id = os.environ["SLWM_T1_EXP_ID"]
force_config = os.environ["SLWM_T1_FORCE_CONFIG"] == "1"
write_config_only = os.environ["SLWM_T1_WRITE_CONFIG_ONLY"] == "1"
run_training = os.environ["SLWM_T1_RUN_TRAINING"] == "1"
allow_existing_artifacts = os.environ["SLWM_T1_ALLOW_EXISTING_ARTIFACTS"] == "1"

config_path = root / config_rel
template_path = root / template_rel
corpus_path = root / corpus_rel
artifact_path = root / artifact_rel


def fail(message: str, code: int = 2) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(code)


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        fail(f"Missing required JSON file: {path}")
    except json.JSONDecodeError as exc:
        fail(f"Invalid JSON in {path}: {exc}")


def build_config() -> dict:
    base = load_json(template_path)
    base["runtime"]["device"] = "mps"
    base["data"]["prepared_corpus_dir"] = str(corpus_rel)
    base["training"]["batch_size"] = 1
    base["training"]["gradient_accumulation_steps"] = 8
    base["training"]["steps"] = 122070
    base["training"]["warmup_steps"] = 2000
    base["training"]["validation_batches"] = 128
    base["training"]["claim_scope"] = "slwm_text_only_1b_token_m4max_t1_guardrail_run"
    base["model"]["name"] = "SLWM-text-only-T1-PyTorch-124M-1B-M4Max"
    base["registry"] = {
        "experiment_id": exp_id,
        "artifact_dir": str(artifact_rel),
        "path": str(artifact_rel / "registry.json"),
        "metrics_path": str(artifact_rel / "metrics.json"),
        "samples_path": str(artifact_rel / "samples.json"),
        "report_path": str(artifact_rel / "report.md"),
        "checkpoint_path": str(artifact_rel / "checkpoint.pt"),
        "config_copy_path": str(artifact_rel / "config.json"),
    }
    base["baselines_compared"] = []
    base["pending_baselines"] = [
        {
            "required_variant": "gpt2_baseline",
            "suggested_experiment_id": "EXP-T1-401",
            "required_match": (
                "same prepared corpus, tokenizer, sequence length, optimizer family, seed policy, "
                "and train-token budget as EXP-T1-402"
            ),
            "comparison_notes": (
                "EXP-T1-101 is historical limited-run context only and is intentionally not listed "
                "in baselines_compared because it is not a matched 1B-token/corpus baseline."
            ),
        }
    ]
    return base


def canonical_json(data: dict) -> str:
    return json.dumps(data, indent=2) + "\n"


generated = canonical_json(build_config())
if config_path.exists():
    current = config_path.read_text()
    if current != generated:
        if force_config:
            config_path.write_text(generated)
            print(f"Updated config with generated {exp_id} settings: {config_rel}")
        else:
            fail(
                f"Existing config differs from generated {exp_id} settings: {config_rel}. "
                "Use --force-config to overwrite, or inspect the diff first."
            )
    else:
        print(f"Config already matches generated {exp_id} settings: {config_rel}")
else:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(generated)
    print(f"Wrote config: {config_rel}")

if write_config_only:
    print("Write-config-only requested; skipping corpus/training preflight.")
    sys.exit(0)

cfg = load_json(config_path)

errors = []
warnings = []

if cfg.get("sprint", {}).get("id") != "T1":
    errors.append("sprint.id must be T1")
if cfg.get("runtime", {}).get("device") != "mps":
    errors.append("runtime.device must be mps")
if cfg.get("model", {}).get("variant") != "slwm_text_only":
    errors.append("model.variant must be slwm_text_only")
if cfg.get("model", {}).get("architecture_flags", {}).get("policy_commit_gate") is not False:
    errors.append("policy_commit_gate must remain false for this T1 text-only run")
if cfg.get("data", {}).get("modalities") != ["text_code"]:
    errors.append("data.modalities must be exactly ['text_code']")
dataset_mix = cfg.get("data", {}).get("dataset_mix", {})
if dataset_mix.get("audio") is not None or dataset_mix.get("visual_video") is not None:
    errors.append("T1 dataset_mix must not include audio or visual_video")
if Path(cfg.get("data", {}).get("prepared_corpus_dir", "")) != corpus_rel:
    errors.append(f"data.prepared_corpus_dir must be {corpus_rel}")
training = cfg.get("training", {})
expected_training = {
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "steps": 122070,
    "warmup_steps": 2000,
    "validation_batches": 128,
    "sequence_length": 1024,
}
for key, expected in expected_training.items():
    if training.get(key) != expected:
        errors.append(f"training.{key} must be {expected}")
if cfg.get("registry", {}).get("experiment_id") != exp_id:
    errors.append(f"registry.experiment_id must be {exp_id}")

required_corpus_files = [
    corpus_path / "dataset_card.json",
    corpus_path / "train.tokens.npy",
    corpus_path / "validation.tokens.npy",
]
missing_corpus_files = [path for path in required_corpus_files if not path.exists()]
if missing_corpus_files:
    errors.append(
        "prepared corpus is incomplete or absent: "
        + ", ".join(str(path.relative_to(root)) for path in missing_corpus_files)
    )
else:
    dataset_card = load_json(corpus_path / "dataset_card.json")
    if dataset_card.get("modalities") != ["text_code"]:
        errors.append("dataset_card.modalities must be exactly ['text_code']")
    tokenizer = dataset_card.get("tokenizer", {})
    if tokenizer.get("effective_type") != "gpt2_bpe":
        errors.append("dataset_card tokenizer.effective_type must be gpt2_bpe")
    if tokenizer.get("vocab_size") != 50257:
        errors.append("dataset_card tokenizer.vocab_size must be 50257")
    train_tokens = dataset_card.get("split_counts", {}).get("train", {}).get("tokens")
    validation_tokens = dataset_card.get("split_counts", {}).get("validation", {}).get("tokens")
    print(f"Corpus: {corpus_rel}")
    print(f"  train tokens: {train_tokens}")
    print(f"  validation tokens: {validation_tokens}")

if artifact_path.exists():
    if not artifact_path.is_dir():
        errors.append(f"Artifact path exists but is not a directory: {artifact_rel}")
    elif any(artifact_path.iterdir()):
        message = f"Artifact directory already contains files: {artifact_rel}"
        if run_training and not allow_existing_artifacts:
            errors.append(message + " (pass --allow-existing-artifacts only if overwrite/resume behavior is intended)")
        else:
            warnings.append(message)

torch_available = importlib.util.find_spec("torch") is not None
if torch_available:
    import torch

    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if not mps_available:
        message = "PyTorch MPS is not available in this environment"
        if run_training:
            errors.append(message)
        else:
            warnings.append(message)
else:
    message = "PyTorch is not importable; training cannot start until torch is installed"
    if run_training:
        errors.append(message)
    else:
        warnings.append(message)

effective_tokens = (
    int(training.get("batch_size", 0))
    * int(training.get("gradient_accumulation_steps", 0))
    * int(training.get("sequence_length", 0))
    * int(training.get("steps", 0))
)
print("Run summary:")
print(f"  experiment: {exp_id}")
print(f"  config: {config_rel}")
print(f"  command: python -m training.t1_torch_text --config {config_rel}")
print(f"  optimizer steps: {training.get('steps')}")
print(f"  effective train-token budget: {effective_tokens:,}")
print(f"  validation batches: {training.get('validation_batches')}")
print("  scope: T1 text/code only; no audio/visual claims")
print("  baseline note: no matched GPT-2 1B-token/corpus baseline is configured; runner should record baseline_missing, not guardrail_pass/fail.")

for warning in warnings:
    print(f"WARNING: {warning}", file=sys.stderr)
if errors:
    print("Preflight failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    sys.exit(2)

print("Preflight passed.")
if not run_training:
    print("Dry run only. Re-run with --run to launch the full training job.")
PY

if [[ "$RUN_TRAINING" -ne 1 ]]; then
  exit 0
fi

"$PYTHON_BIN" -m training.t1_torch_text --config "$CONFIG_REL"
