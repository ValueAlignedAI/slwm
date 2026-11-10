#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/dev/slwm-g7"
root=/mnt/slwm-checkpoints-a/SLWM/g7
export MUJOCO_GL=egl
export SLWM_JAX_MEMORY_FRACTION=.58
for seed in 711 712; do
  for core in hybrid gpt2; do
    if ! python3 -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["models"][sys.argv[2]]["gate"])' "$root/policy-prior/sft-lock.json" "$core-$seed"; then continue; fi
    flock /tmp/slwm_research_gpu.lock bash scripts/train/g4_cuda.sh -m training.g7_rl \
      --parents /mnt/slwm-checkpoints-a/SLWM/g6 --out "$root/rl" --sft "$root/policy-prior" \
      --prior "$root/reacher-prior/prior.npz" --lock "$root/policy-prior/sft-lock.json" \
      --sim-site "$HOME/dev/slwm-g7/.reacher/lib/python3.12/site-packages" --core "$core" --seed "$seed"
  done
done
