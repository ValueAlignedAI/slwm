#!/usr/bin/env bash
set -euo pipefail
cd "${SLWM_ROOT:-/home/db/dev/slwm}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}.research-venv/lib/python3.12/site-packages"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
STREAM_PYTHON="${SLWM_PYTHON:-/home/db/venvs/vllm/bin/python}"
"$STREAM_PYTHON" -m training.text_500m --stage train
"$STREAM_PYTHON" -m training.text_500m --stage evaluate
