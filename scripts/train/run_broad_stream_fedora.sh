#!/usr/bin/env bash
set -euo pipefail
cd "${SLWM_ROOT:-/home/db/dev/slwm}"
STREAM_PYTHON="${SLWM_PYTHON:-/home/db/venvs/vllm/bin/python}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}.research-venv/lib/python3.12/site-packages"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
test -f artifacts/t2_audio_visual/broad_stream_prepared_v1/annotation_review_resolved.json
"$STREAM_PYTHON" -m training.broad_stream_train --stage smoke
"$STREAM_PYTHON" -m training.broad_stream_train --stage train
