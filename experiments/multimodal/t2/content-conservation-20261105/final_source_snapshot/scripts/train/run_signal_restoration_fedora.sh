#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
research_python="${SLWM_PYTHON:-/home/db/venvs/vllm/bin/python}"
research_config="configs/t2/signal_restoration.json"
export OMP_NUM_THREADS=4
export PYTHONPATH=".research-venv/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${1:-}" != "--finish-existing" ]]; then
  "$research_python" -u -m training.restoration_data --config "$research_config"
  "$research_python" -u -m training.restoration_data --config "$research_config" --external-only
  "$research_python" -u -m training.restoration_train --config "$research_config"
fi
"$research_python" -u -m training.restoration_train --config "$research_config" --evaluate
"$research_python" -u -m evals.restoration_content_probe --config "$research_config"
"$research_python" -u -m exploration.restoration_outputs --config "$research_config"
"$research_python" -u -m evals.restoration_silence_control --config "$research_config"
