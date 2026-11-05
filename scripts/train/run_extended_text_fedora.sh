#!/usr/bin/env bash
set -euo pipefail
cd "${SLWM_ROOT:-/home/db/dev/slwm}"
TEXT_EXTEND_PYTHON="${SLWM_PYTHON:-/home/db/venvs/vllm/bin/python}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}.research-venv/lib/python3.12/site-packages"
export OMP_NUM_THREADS=4
"$TEXT_EXTEND_PYTHON" -m training.extend_text_train --stage train
"$TEXT_EXTEND_PYTHON" -m training.extend_text_train --stage evaluate
