#!/usr/bin/env bash
# Publish verified, reproducible source packs from Fedora. Manifest is written last.
set -euo pipefail
umask 077
root=${1:?mounted G5 data}
test "$(findmnt -nro TARGET -T "$root")" = /mnt/slwm-data
export PATH="$HOME/.local/bin:$PATH" RCLONE_CONFIG="$HOME/.config/slwm-colab/rclone.conf"
export RCLONE_LOG_LEVEL=ERROR RCLONE_DRIVE_CHUNK_SIZE=64M
mkdir -p "$root/cloud"
exec 8>"$root/cloud/ship.lock"
flock -n 8
for role in train dev test; do
    list="$root/cloud/$role.sha256.partial"
    : > "$list"
    for family in paired impact pusht; do
        until test -f "$root/$family/$role/card.json"; do sleep 10; done
        file="$root/cloud/$family-$role.tar.zst"
        if [[ ! -f $file ]]; then
            if [[ $family == paired ]] && [[ -f $root/paired/$role.tar.zst ]]; then
                ln "$root/paired/$role.tar.zst" "$file"
            else
                tar -chf "$file.partial" --use-compress-program='zstd -q -1 -T2' -C "$root/$family" "$role"
                mv "$file.partial" "$file"
            fi
        fi
        remote="slwm-drive:SLWM/g5/data/${file##*/}"
        expected=$(sha256sum "$file" | cut -d' ' -f1)
        rclone copyto --immutable --checksum --retries 3 "$file" "$remote"
        actual=$(rclone hashsum SHA-256 "$remote" | awk 'NR==1{print $1}')
        test "$expected" = "$actual"
        printf '%s  %s\n' "$actual" "${file##*/}" >> "$list"
        echo "verified ${file##*/} $actual"
    done
    mv "$list" "$root/cloud/$role.sha256"
    rclone copyto --immutable --checksum "$root/cloud/$role.sha256" "slwm-drive:SLWM/g5/data/$role.sha256"
done
