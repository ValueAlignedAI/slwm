#!/usr/bin/env bash
set -euo pipefail
deadline=${1:?absolute UTC finish time}
root=/mnt/slwm-checkpoints-a/SLWM/g6
test "$(findmnt -nro TARGET -T "$root")" = /mnt/slwm-checkpoints-a
until [[ -s /mnt/slwm-checkpoints-a/SLWM/g5/terminal-test.json ]]; do
    (( $(date +%s) < deadline )) || exit 0
    sleep 50
done
cd /home/db/dev/slwm-g5
mkdir -p "$root/throughput"
for scale in anchor larger; do
    for core in hybrid gpt2; do
        for micro in 32 64 128; do
            file="$root/throughput/gpu-$scale-$core-$micro"
            [[ ! -s $file.json && ! -f $file.oom ]] || continue
            if ! flock /tmp/slwm_research_gpu.lock bash scripts/train/g4_cuda.sh -m training.g6_gpu_bench --data /mnt/slwm-data/SLWM/g5 --scale "$scale" --core "$core" --micro "$micro" --output "$file.json" >"$file.log" 2>&1; then
                if grep -Eiq 'RESOURCE_EXHAUSTED|out of memory' "$file.log"; then touch "$file.oom"; else exit 1; fi
            fi
        done
    done
done
