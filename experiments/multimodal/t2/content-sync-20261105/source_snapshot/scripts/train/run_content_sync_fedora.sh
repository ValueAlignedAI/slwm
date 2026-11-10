#!/usr/bin/env bash
set -euo pipefail
cd "${SLWM_ROOT:-/home/db/dev/slwm}"
CONTENT_PYTHON="${SLWM_PYTHON:-/home/db/venvs/vllm/bin/python}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}.research-venv/lib/python3.12/site-packages"
export OMP_NUM_THREADS=4
"$CONTENT_PYTHON" -m training.content_sync_train --stage smoke
"$CONTENT_PYTHON" -m training.content_sync_train --stage train
"$CONTENT_PYTHON" -m training.content_sync_train --stage calibrate
"$CONTENT_PYTHON" -m training.content_sync_train --stage evaluate
