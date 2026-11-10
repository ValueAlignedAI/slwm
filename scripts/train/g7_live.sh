#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/dev/slwm-g7"
root=/mnt/slwm-checkpoints-a/SLWM/g7
data=/mnt/slwm-data/SLWM/g7
test "$(findmnt -nro TARGET -T "$root")" = /mnt/slwm-checkpoints-a
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3
py="$HOME/venvs/vllm/bin/python"
deps="$HOME/dev/slwm-g7:$HOME/dev/slwm-g2/.g2-deps:$HOME/dev/slwm/.research-venv/lib/python3.12/site-packages"
for arm in fresh replay; do
  for core in hybrid gpt2; do
    tag="g7-v5e-anchor-$core-511-$arm"
    stage="$root/$tag/step-016384"; out="$root/live/$tag"
    limit=$((SECONDS+36000))
    while [[ ! -s "$stage/verified.json" && $SECONDS -lt $limit ]]; do sleep 25; done
    test -s "$stage/verified.json"
    mkdir -p "$out/previews"
    if [[ ! -s "$out/live.json" ]]; then
      flock /tmp/slwm_research_gpu.lock bash scripts/train/g4_cuda.sh -m training.g5_live predict --starss --data "$data" --run "$stage" --out "$out"
    fi
    if [[ ! -s "$out/media-verified.json" ]]; then
      PYTHONPATH="$deps" "$py" -m training.g5_live render --starss --data "$data" --repo "$HOME/dev/slwm-g3" --out "$out"
      PYTHONPATH="$deps" "$py" scripts/train/g6_media_metrics.py "$out" "$stage"
    fi
    if [[ -s "$out/previews.json" ]]; then continue; fi
    ffmpeg -y -v error -i "$out/audio-to-video.mp4" -an -vf scale=128:128:flags=neighbor -c:v libvpx-vp9 -b:v 36k -crf 40 "$out/previews/audio-to-video.webm"
    ffmpeg -y -v error -i "$out/video-to-audio.wav" -c:a libopus -b:a 40k "$out/previews/video-to-audio.ogg"
    ffmpeg -y -v error -i "$out/text-to-audio-video.mp4" -vf scale=128:128:flags=neighbor -c:v libvpx-vp9 -b:v 36k -crf 40 -c:a libopus -b:a 40k "$out/previews/text-to-audio-video.webm"
    python3 scripts/train/g7_check_previews.py "$out"
  done
done
