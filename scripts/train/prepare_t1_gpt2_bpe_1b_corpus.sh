#!/usr/bin/env bash
set -euo pipefail

# Prepare the Sprint T1 GPT-2-BPE text/code corpus required by EXP-T1-402.
#
# Default behavior is dry preflight only. The large Hugging Face streaming job
# starts only when --run is passed explicitly.
#
# Usage:
#   scripts/train/prepare_t1_gpt2_bpe_1b_corpus.sh
#   scripts/train/prepare_t1_gpt2_bpe_1b_corpus.sh --run
#   scripts/train/prepare_t1_gpt2_bpe_1b_corpus.sh --run --allow-existing-output
#   scripts/train/prepare_t1_gpt2_bpe_1b_corpus.sh --clean-failed-output

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_REL="configs/t1/text_code_gpt2_bpe_1b_v0_prepare.json"
RUN_PREP=0
ALLOW_EXISTING_OUTPUT=0
CLEAN_FAILED_OUTPUT=0

usage() {
  "$PYTHON_BIN" - <<'PY'
print("""Usage: scripts/train/prepare_t1_gpt2_bpe_1b_corpus.sh [options]

Options:
  --run                    Launch python -m training.t1_prepare_text_code after preflight.
  --allow-existing-output  Allow --run when the output corpus directory already has files.
  --clean-failed-output    Remove only known incomplete temp files from a failed prep attempt.
  -h, --help               Show this help.

Default: validate config/dependencies/disk budget and do not download/tokenize data.
""")
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      RUN_PREP=1
      ;;
    --allow-existing-output)
      ALLOW_EXISTING_OUTPUT=1
      ;;
    --clean-failed-output)
      CLEAN_FAILED_OUTPUT=1
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

export SLWM_T1_PREP_CONFIG_REL="$CONFIG_REL"
export SLWM_T1_RUN_PREP="$RUN_PREP"
export SLWM_T1_ALLOW_EXISTING_OUTPUT="$ALLOW_EXISTING_OUTPUT"
export SLWM_T1_CLEAN_FAILED_OUTPUT="$CLEAN_FAILED_OUTPUT"

"$PYTHON_BIN" - <<'PY'
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

config_rel = Path(os.environ["SLWM_T1_PREP_CONFIG_REL"])
run_prep = os.environ["SLWM_T1_RUN_PREP"] == "1"
allow_existing_output = os.environ["SLWM_T1_ALLOW_EXISTING_OUTPUT"] == "1"
clean_failed_output = os.environ["SLWM_T1_CLEAN_FAILED_OUTPUT"] == "1"
root = Path.cwd()
config_path = root / config_rel


def fail(message: str, code: int = 2) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(code)


try:
    config = json.loads(config_path.read_text(encoding="utf-8"))
except FileNotFoundError:
    fail(f"Missing prep config: {config_rel}")
except json.JSONDecodeError as exc:
    fail(f"Invalid JSON in prep config {config_rel}: {exc}")

data = config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}
errors = []
warnings = []

if config.get("sprint", {}).get("id") != "T1":
    errors.append("sprint.id must be T1")
if config.get("tokenizer", {}).get("type") != "gpt2_bpe":
    errors.append("tokenizer.type must be gpt2_bpe")
if data.get("modalities") != ["text_code"]:
    errors.append("data.modalities must be exactly ['text_code']")
if data.get("preparation_mode") != "streaming":
    errors.append("data.preparation_mode must be streaming for the 1B corpus")
dataset_mix = data.get("dataset_mix", {}) if isinstance(data.get("dataset_mix", {}), dict) else {}
if dataset_mix.get("audio") is not None or dataset_mix.get("visual_video") is not None:
    errors.append("T1 corpus must not include audio or visual_video")

output_dir = Path(str(data.get("output_dir", "")))
if output_dir.is_absolute() or not str(output_dir).startswith("artifacts/t1_text_code/"):
    errors.append("data.output_dir must be relative and under artifacts/t1_text_code/")
output_path = root / output_dir
if clean_failed_output:
    if not output_path.exists():
        print(f"No output directory to clean: {output_dir}")
        sys.exit(0)
    final_files = ["dataset_card.json", "train.tokens.npy", "validation.tokens.npy", "test.tokens.npy", "manifest.jsonl", "prepare_config.json"]
    present_final = [name for name in final_files if (output_path / name).exists()]
    if present_final:
        fail(
            "Refusing to clean because finalized corpus files exist: "
            + ", ".join(present_final)
            + ". Move or inspect the corpus manually."
        )
    removable = [
        path
        for path in output_path.iterdir()
        if path.name.endswith(".tmp") or path.name.endswith(".tokens.uint32.tmp")
    ]
    for path in removable:
        path.unlink()
    print(f"Removed {len(removable)} failed-prep temp files from {output_dir}")
    sys.exit(0)

if output_path.exists():
    if not output_path.is_dir():
        errors.append(f"output path exists but is not a directory: {output_dir}")
    elif any(output_path.iterdir()):
        message = f"output corpus directory already contains files: {output_dir}"
        if run_prep and not allow_existing_output:
            errors.append(message + " (pass --allow-existing-output only if replacing/continuing intentionally)")
        else:
            warnings.append(message)

sources = data.get("sources", []) if isinstance(data.get("sources", []), list) else []
source_max_tokens = 0
category_targets = {}
for source in sources:
    if not isinstance(source, dict):
        errors.append("data.sources entries must be objects")
        continue
    category = str(source.get("category", "unknown"))
    max_tokens = int(source.get("max_tokens", 0))
    source_max_tokens += max_tokens
    category_targets[category] = category_targets.get(category, 0) + max_tokens

target_total = int(data.get("target_tokens", {}).get("total", 0)) if isinstance(data.get("target_tokens", {}), dict) else 0
if target_total and source_max_tokens != target_total:
    warnings.append(f"configured source max_tokens sum {source_max_tokens:,} differs from target total {target_total:,}")

missing_deps = [name for name in ("transformers", "datasets") if importlib.util.find_spec(name) is None]
if missing_deps:
    message = "missing optional T1 full dependencies: " + ", ".join(missing_deps)
    if run_prep:
        errors.append(message)
    else:
        warnings.append(message)

free_bytes = shutil.disk_usage(root).free
estimated_final_bytes = max(source_max_tokens, target_total) * 4
estimated_working_bytes = int(estimated_final_bytes * 2.5)
if estimated_working_bytes and free_bytes < estimated_working_bytes:
    message = (
        f"free disk {free_bytes / (1024**3):.1f} GiB is below estimated working need "
        f"{estimated_working_bytes / (1024**3):.1f} GiB"
    )
    if run_prep:
        errors.append(message)
    else:
        warnings.append(message)

print("Corpus prep summary:")
print(f"  config: {config_rel}")
print(f"  output_dir: {output_dir}")
print(f"  tokenizer: {config.get('tokenizer', {}).get('type')}")
print(f"  preparation_mode: {data.get('preparation_mode')}")
print(f"  source max_tokens total: {source_max_tokens:,}")
for category, tokens in sorted(category_targets.items()):
    print(f"  category {category}: {tokens:,} max tokens")
print(f"  estimated final token arrays: {estimated_final_bytes / (1024**3):.1f} GiB")
print(f"  estimated working disk: {estimated_working_bytes / (1024**3):.1f} GiB")
print("  output contract: dataset_card.json, manifest.jsonl, train/validation/test.tokens.npy")
print("  caveat: code source is an accessible Python-only StarCoder-derived fallback, not the preferred authenticated BigCode permissive-filtered source.")

for warning in warnings:
    print(f"WARNING: {warning}", file=sys.stderr)
if errors:
    print("Preflight failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    sys.exit(2)

print("Preflight passed.")
if not run_prep:
    print("Dry run only. Re-run with --run to start corpus preparation.")
PY

if [[ "$RUN_PREP" -ne 1 ]]; then
  exit 0
fi

"$PYTHON_BIN" -m training.t1_prepare_text_code --config "$CONFIG_REL"
