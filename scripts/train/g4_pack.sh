#!/usr/bin/env bash
set -euo pipefail
repo=${1:?source checkout}
ops=${2:?G4 checkout}
data=${3:?mounted data directory}
test "$(findmnt -nro TARGET -T "$data")" = /mnt/slwm-data
export PATH="$HOME/.local/bin:$PATH"
export RCLONE_CONFIG="$HOME/.config/slwm-colab/rclone.conf"
export RCLONE_LOG_LEVEL=ERROR
export RCLONE_DRIVE_CHUNK_SIZE=32M
export PYTHONPATH="$HOME/dev/slwm-g2/.g2-deps:$repo:$HOME/dev/slwm/.research-venv/lib/python3.12/site-packages"
for role in dev train test; do
    date -Is
    "$HOME/venvs/vllm/bin/python" -u "$ops/training/g4_data.py" --config "$ops/configs/t4/crossmodal_tpu.json" --repo "$repo" --output "$data" --role "$role"
    file="$role.tar.zst"
    (cd "$data" && sha256sum "$file" > "$file.sha256")
    target="slwm-drive:SLWM/g4/paired-v1/$file"
    rclone copyto --immutable --checksum "$data/$file" "$target"
    expected=$(cut -d ' ' -f1 "$data/$file.sha256")
    remote=$(rclone hashsum SHA-256 "$target" | awk '{print $1}')
    test "$expected" = "$remote"
    rclone copyto --immutable "$data/$file.sha256" "$target.sha256"
    echo "published $role $remote"
done
