#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
SLWM_PYTHON="${SLWM_PYTHON:-/home/db/venvs/vllm/bin/python}"
SLWM_MATRIX="${SLWM_MATRIX:-configs/t2/cremad_pilot_matrix.json}"
"$SLWM_PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1])); c=json.load(open(m["corpus_dir"]+"/dataset_card.json")); assert "midpoint" in c["feature_spec"]["audio"], "Regenerate midpoint-safe audio features first"' "$SLWM_MATRIX"
"$SLWM_PYTHON" -u -m training.t2_real_av --config "$SLWM_MATRIX"
# Plotting dependencies are isolated from the GPU runtime.
PYTHONPATH="${PWD}/.research-venv/lib/python3.12/site-packages${PYTHONPATH:+:${PYTHONPATH}}" \
  "$SLWM_PYTHON" -m training.t2_real_av_report --config "$SLWM_MATRIX"
