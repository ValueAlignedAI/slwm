#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
SLWM_PYTHON="${SLWM_PYTHON:-/home/db/venvs/vllm/bin/python}"
SLWM_MATRIX="${SLWM_MATRIX:-configs/t2/motion_alignment_matrix.json}"
"$SLWM_PYTHON" -u -m training.t2_motion_train --config "$SLWM_MATRIX"
"$SLWM_PYTHON" -u -m evals.motion_causal_checks --config "$SLWM_MATRIX"
"$SLWM_PYTHON" -u -m exploration.motion_outputs --config "$SLWM_MATRIX"
if ! "$SLWM_PYTHON" -u -m evals.motion_audio_intelligibility --config "$SLWM_MATRIX"; then
  printf '%s\n' 'Optional external ASR check unavailable; see exploration/intelligibility.json.'
fi
PYTHONPATH="${PWD}/.research-venv/lib/python3.12/site-packages${PYTHONPATH:+:${PYTHONPATH}}" \
  "$SLWM_PYTHON" -m training.t2_motion_report --config "$SLWM_MATRIX"
