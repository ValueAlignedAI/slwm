#!/usr/bin/env bash
# Acquire original paired recordings directly into the mounted study data volume.
set -euo pipefail
umask 077
root=${1:?mounted G5 data directory}
family=${2:?libri360 or greatesthits}
test "$(findmnt -nro TARGET -T "$root")" = /mnt/slwm-data
mkdir -p "$root/sources"
cd "$root/sources"
case "$family" in
  libri360)
    f=train-clean-360.tar.gz
    urls=(https://www.openslr.org/resources/12/train-clean-360.tar.gz https://openslr.trmal.net/resources/12/train-clean-360.tar.gz https://openslr.elda.org/resources/12/train-clean-360.tar.gz)
    ;;
  greatesthits)
    f=vis-data-256.zip
    urls=(https://web.eecs.umich.edu/~ahowens/vis/vis-data-256.zip)
    ;;
  *) exit 2 ;;
esac
if [[ ! -f $f ]]; then
    test "$(df -B1 --output=avail . | tail -1)" -gt 50000000000
    downloaded=0
    for url in "${urls[@]}"; do
        if curl -fsSL --connect-timeout 15 --speed-time 120 --speed-limit 1024 --retry 2 --continue-at - "$url" -o "$f.partial"; then
            downloaded=1
            break
        fi
    done
    test "$downloaded" = 1
    if [[ $family == libri360 ]]; then
        echo "c0e676e450a7ff2f54aeade5171606fa  $f.partial" | md5sum -c -
    else
        unzip -tq "$f.partial"
    fi
    mv "$f.partial" "$f"
fi
sha256sum "$f" > "$f.sha256.new"
mv "$f.sha256.new" "$f.sha256"
stat -c '%n %s bytes' "$f"
