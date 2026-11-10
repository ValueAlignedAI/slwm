#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
research_python="${SLWM_PYTHON:-/home/db/venvs/vllm/bin/python}"
research_config="configs/t2/content_conservation.json"
export OMP_NUM_THREADS=4
export PYTHONPATH=".research-venv/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${1:-}" != "--finish-existing" ]]; then
  "$research_python" -u -m training.restoration_other --config "$research_config"
  "$research_python" -u -m training.restoration_content --config "$research_config"
fi
"$research_python" -u -m training.restoration_content --config "$research_config" --evaluate
"$research_python" -u -m evals.content_conservation_probe --config "$research_config"
"$research_python" -u -m exploration.content_conservation_outputs --config "$research_config"
"$research_python" -u -m evals.restoration_inference_checks --r1 configs/t2/signal_restoration.json --r2 "$research_config"
"$research_python" -u -m training.restoration_report --r1 configs/t2/signal_restoration.json --r2 "$research_config"
