#!/usr/bin/env bash
# Reproduce T2-G1 on the authorized Fedora GPU; never installs system packages.
set -euo pipefail
cd "$(dirname "$0")/../.."
research_python="${SLWM_PYTHON:-/home/db/venvs/vllm/bin/python}"
research_config="configs/t2/conditional_generation.json"
export OMP_NUM_THREADS=4
if [[ "${1:-}" != "--finish-existing" ]]; then
  if [[ ! -f artifacts/t2_audio_visual/ravdess_external_v1/dataset_card.json ]]; then
    "$research_python" -u -m training.generation_external
  fi
  "$research_python" -u -m training.t2_prepare_generation --config "$research_config"
  "$research_python" -u -m training.t2_prepare_generation --config "$research_config" --external
  "$research_python" -u -m training.t2_generation_train --config "$research_config"
fi
"$research_python" -u -m training.t2_generation_train --config "$research_config" --evaluate
"$research_python" -u -m evals.generation_controls --config "$research_config"
"$research_python" -u -m exploration.generation_outputs --config "$research_config"
"$research_python" -u -m evals.generation_intelligibility --config "$research_config"
"$research_python" -u -m evals.generation_shuffled_audit --config "$research_config"
PYTHONPATH=".research-venv/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}" "$research_python" -u -m training.t2_generation_report --config "$research_config"
