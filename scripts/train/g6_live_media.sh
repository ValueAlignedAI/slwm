#!/usr/bin/env bash
set -euo pipefail
ckpt=${1:?verified terminal stage}
folder=${2:?private result directory}
device=${3:?cpu or gpu}
root=/mnt/slwm-checkpoints-a/SLWM/g6
data=/mnt/slwm-data/SLWM/g5
repo=/home/db/dev/slwm-g5
[[ $ckpt == "$root/"*/step-008192 && $folder == "$root/terminal/"* ]]
test "$(findmnt -nro TARGET -T "$root")" = /mnt/slwm-checkpoints-a
test -f "$ckpt/verified.json"
cd "$repo"
mkdir -p "$folder"
exec 8>"$folder/.media.lock"
flock 8
if [[ ! -s $folder/media-verified.json && ( ! -s $folder/predictions.npz || ! -s $folder/live.json ) ]]; then
    if [[ $device == cpu ]]; then
        JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/db/dev/slwm-g4/.jax-cpu/bin/python -m training.g5_live predict --data "$data" --run "$ckpt" --out "$folder"
    elif [[ $device == gpu ]]; then
        flock /tmp/slwm_research_gpu.lock bash scripts/train/g4_cuda.sh -m training.g5_live predict --data "$data" --run "$ckpt" --out "$folder"
    else
        exit 2
    fi
fi
py=/home/db/venvs/vllm/bin/python
export PYTHONPATH="$repo:/home/db/dev/slwm-g2/.g2-deps:/home/db/dev/slwm/.research-venv/lib/python3.12/site-packages"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3
if [[ ! -s $folder/media-verified.json ]]; then
    "$py" -u "$repo/training/g5_live.py" render --data "$data" --repo /home/db/dev/slwm-g3 --out "$folder"
    "$py" scripts/train/g6_media_metrics.py "$folder" "$ckpt"
fi
mkdir -p "$folder/previews"
for name in audio-to-video video-to-audio text-to-audio-video; do
    [[ ! -s $folder/previews/$name.webm ]] || continue
    ffmpeg -y -v error -i "$folder/$name.mp4" -vf scale=128:128:flags=neighbor -c:v libvpx-vp9 -deadline realtime -cpu-used 6 -b:v 40k -crf 42 -c:a libopus -b:a 32k "$folder/previews/$name.partial.webm"
    mv "$folder/previews/$name.partial.webm" "$folder/previews/$name.webm"
done
