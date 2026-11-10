#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/dev/slwm-g7"
for phase in train lock test; do
  for seed in 711 712; do
    for core in hybrid gpt2 raw; do
      if [[ $phase == lock && ( $core != hybrid || $seed != 711 ) ]]; then continue; fi
      flock /tmp/slwm_research_gpu.lock bash scripts/train/g4_cuda.sh -m training.g7_camera "$phase" \
        --core "$core" --seed "$seed" --root /mnt/slwm-data/SLWM/g7 \
        --parents /mnt/slwm-checkpoints-a/SLWM/g6 --out /mnt/slwm-checkpoints-a/SLWM/g7/camera
    done
  done
done
