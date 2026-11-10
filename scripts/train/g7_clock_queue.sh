#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/dev/slwm-g7"
root=/mnt/slwm-data/SLWM/g7
out=/mnt/slwm-checkpoints-a/SLWM/g7/physical-time
parents=/mnt/slwm-checkpoints-a/SLWM/g6
deps="$HOME/dev/slwm-g7:$HOME/dev/slwm-g2/.g2-deps:$HOME/dev/slwm/.research-venv/lib/python3.12/site-packages"
for role in train confirm; do
  if [[ $role == confirm ]]; then
    PYTHONPATH="$deps" "$HOME/venvs/vllm/bin/python" -m training.g7_clock_fit lock --root "$root" --out "$out" --core hybrid --seed 711 --recipe both-signs
    limit=$((SECONDS+7200))
    while [[ $SECONDS -lt $limit && ( ! -s "$root/physical-time/crema/confirm/card.json" || ! -s "$root/physical-time/events/confirm/card.json" ) ]]; do sleep 20; done
  fi
  for seed in 711 712; do
    for core in hybrid gpt2 raw; do
      flock /tmp/slwm_research_gpu.lock bash scripts/train/g4_cuda.sh -m training.g7_clock_cache --root "$root" --parents "$parents" --core "$core" --seed "$seed" --role "$role"
      for recipe in older-only both-signs; do
        PYTHONPATH="$deps" flock /tmp/slwm_research_gpu.lock "$HOME/venvs/vllm/bin/python" -u -m training.g7_clock_fit "$role" --root "$root" --out "$out" --core "$core" --seed "$seed" --recipe "$recipe"
      done
    done
  done
done
