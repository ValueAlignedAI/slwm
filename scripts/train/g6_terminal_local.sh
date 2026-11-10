#!/usr/bin/env bash
# One set of recorded decisions and conditional media tests per verified final pair.
set -euo pipefail
deadline=${1:?absolute finish time in Unix seconds}
root=/mnt/slwm-checkpoints-a/SLWM/g6
data=/mnt/slwm-data/SLWM/g5
test "$(findmnt -nro TARGET -T "$root")" = /mnt/slwm-checkpoints-a
test "$(findmnt -nro TARGET -T "$data")" = /mnt/slwm-data
repo=/home/db/dev/slwm-g5
cd "$repo"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
mkdir -p "$root/terminal"
sim=/home/db/dev/slwm-g5/.sim/lib/python3.12/site-packages
py=/home/db/venvs/vllm/bin/python
media=/home/db/dev/slwm/.research-venv/lib/python3.12/site-packages
until test -s /mnt/slwm-checkpoints-a/SLWM/g5/terminal-test.json; do
    (( $(date +%s) < deadline )) || exit 0
    sleep 40
done
while (( $(date +%s) < deadline )); do
    completed=0
    for s in 511 512; do
        for scale in larger anchor; do
            for gen in v5e v6e; do
                id="$gen-$scale-$s"
                work="$root/terminal/$id"
                first="$root/g6-$gen-$scale-hybrid-$s/step-008192"
                second="$root/g6-$gen-$scale-gpt2-$s/step-008192"
                [[ -f $first/verified.json && -f $second/verified.json ]] || continue
                mkdir -p "$work"
                for core in hybrid gpt2; do
                    ckpt="$root/g6-$gen-$scale-$core-$s/step-008192"
                    if [[ ! -s $work/contact-$core.json ]]; then
                        flock /tmp/slwm_research_gpu.lock bash scripts/train/g4_cuda.sh -m training.g6_contact_audit --run "$root/g6-$gen-$scale-$core-$s" --data "$data" --out "$work/contact-$core.json"
                    fi
                    if [[ ! -s $work/text-$core.json ]]; then
                        flock /tmp/slwm_research_gpu.lock bash scripts/train/g4_cuda.sh -m training.g6_text_probe --run "$ckpt" --data "$data" --out "$work/text-$core.json"
                    fi
                    if [[ ! -s $work/scene-$core/scene-controls.json ]]; then
                        flock /tmp/slwm_research_gpu.lock bash scripts/train/g4_cuda.sh -m training.g6_scene_probe score --run "$ckpt" --data "$data" --scenes "$data/g6-unseen-600000" --output "$work/scene-$core"
                    fi
                done
                if [[ ! -f $work/decisions.json ]]; then
                    flock /tmp/slwm_research_gpu.lock bash scripts/train/g4_cuda.sh -m training.g6_decision_probe --root "$root" --data "$data" --sim-site "$sim" --output "$work/decisions.json" --scale "$scale" --generation "$gen" --seed "$s"
                fi
                if [[ ! -f $work/speech-mels.npz ]]; then
                    flock /tmp/slwm_research_gpu.lock bash scripts/train/g4_cuda.sh -m training.g6_speech_features --root "$root" --data "$data" --out "$work" --scale "$scale" --generation "$gen" --seed "$s"
                fi
                if [[ ! -s $work/speech-dev.json ]]; then
                    PYTHONPATH="/home/db/dev/slwm-g3:/home/db/dev/slwm-g2/.g2-deps:$media" TRANSFORMERS_OFFLINE=1 "$py" -u "$repo/training/g5_content.py" speech --repo /home/db/dev/slwm-g3 --out "$work"
                fi
                if [[ $scale == larger && $s == 511 ]]; then
                    for core in hybrid gpt2; do
                        ckpt="$root/g6-$gen-$scale-$core-$s/step-008192"
                        folder="$work/live-$core"
                        bash scripts/train/g6_live_media.sh "$ckpt" "$folder" gpu
                    done
                fi
                completed=$((completed+1))
            done
        done
    done
    (( completed < 4 )) || exit 0
    sleep 55
done
