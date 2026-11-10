#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ $(findmnt -n -o FSTYPE /dev/shm) == tmpfs ]]
mkdir -p /content/slwm-ops
install -d -m 700 /dev/shm/slwm-drive
if ! command -v rclone >/dev/null; then
    [[ $(uname -m) == x86_64 ]] || { echo "Install rclone for this VM architecture" >&2; exit 1; }
    temp=$(mktemp -d)
    trap 'rm -rf -- "$temp"' EXIT
    cd "$temp"
    archive=rclone-v1.76.0-linux-amd64.zip
    curl -fsSL --retry 3 "https://downloads.rclone.org/v1.76.0/$archive" -o "$archive"
    echo "b00f00b3266c3764413afe599d3ec316f6f43b3e0ea8a0bc1d186d2133e46df7  $archive" | sha256sum -c -
    unzip -q "$archive" '*/rclone'
    install -m 755 rclone-v1.76.0-linux-amd64/rclone /usr/local/bin/rclone
fi
rclone version
