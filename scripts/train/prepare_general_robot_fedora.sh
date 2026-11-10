#!/usr/bin/env bash
set -euo pipefail
cd "${SLWM_ROOT:-/home/db/dev/slwm}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}.research-venv/lib/python3.12/site-packages"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
GENERAL_PYTHON="${SLWM_PYTHON:-/home/db/venvs/vllm/bin/python}"
"$GENERAL_PYTHON" -m training.general_droid_sample --stage download
"$GENERAL_PYTHON" -m training.general_droid_sample --stage prepare
"$GENERAL_PYTHON" -m training.general_robot_metadata --family droid_expanded
