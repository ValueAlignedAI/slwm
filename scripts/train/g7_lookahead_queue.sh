#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/dev/slwm-g7"
export MUJOCO_GL=egl SLWM_JAX_MEMORY_FRACTION=.58
for phase in transition readout lock test; do
  for seed in 711 712; do
    for core in hybrid gpt2; do
      if [[ ( $phase == transition || $phase == lock ) && ( $core != hybrid || $seed != 711 ) ]]; then continue; fi
      flock /tmp/slwm_research_gpu.lock bash scripts/train/g4_cuda.sh -m training.g7_lookahead "$phase" \
        --core "$core" --seed "$seed" --root /mnt/slwm-data/SLWM/g7 --parents /mnt/slwm-checkpoints-a/SLWM/g6 \
        --out /mnt/slwm-checkpoints-a/SLWM/g7/planning --sim-site "$HOME/dev/slwm-g7/.reacher/lib/python3.12/site-packages"
    done
  done
done
