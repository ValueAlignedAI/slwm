#!/usr/bin/env bash
set -euo pipefail
umask 077
cd /content/slwm-g4
export RCLONE_CONFIG=/dev/shm/slwm-drive/rclone.conf
export RCLONE_LOG_LEVEL=ERROR
source /content/g4-topology.env
export JAX_PLATFORMS=tpu
export JAX_COMPILATION_CACHE_DIR=/content/g4-xla-cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export OMP_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=1
mkdir -p /content/g4data /content/g4-out /content/g4-spool
python -c 'import optax' || python -m pip install -q optax
python -c 'import zstandard' || python -m pip install -q zstandard
python scripts/train/g4_cloud_fetch.py dev &
dev_pid=$!
python scripts/train/g4_cloud_fetch.py train
wait "$dev_pid"
flock -n /content/g4-uploader.lock python -u scripts/train/colab_artifacts.py sync --spool /content/g4-spool --watch > /content/g4-out/uploader.log 2>&1 &
python -u -m training.g4_bench --backend tpu --data /content/g4data --out /content/g4-out/batch-tpu.json --scale pilot --batch 256 --scan 4 --reps 3
