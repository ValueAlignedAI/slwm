#!/usr/bin/env bash
set -euo pipefail
python_root=${SLWM_CUDA_PYTHON:-"$HOME/venvs/vllm"}
libs=$(find "$python_root/lib" -type d -path '*/site-packages/nvidia/*/lib' -print | paste -sd: -)
export LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION="${SLWM_JAX_MEMORY_FRACTION:-.84}"
export OMP_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=1
exec "$HOME/dev/slwm-g4/.jax-gpu/bin/python" -u "$@"
