#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
research_python="${SLWM_PYTHON:-/home/db/venvs/vllm/bin/python}"
research_config="configs/t2/conditional_generation_residual.json"
export OMP_NUM_THREADS=4
"$research_python" -u -m training.t2_generation_residual --config "$research_config"
"$research_python" -u -m training.t2_generation_residual --config "$research_config" --evaluate
"$research_python" -u -m exploration.generation_residual_outputs --config "$research_config"
"$research_python" -u -m training.t2_generation_residual_report --config "$research_config"
