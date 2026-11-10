#!/usr/bin/env bash
# Pin local SSD packs independently of cloud API quotas, so an admitted TPU can start immediately.
set -euo pipefail
umask 077
root=${1:?mounted G5 data}
test "$(findmnt -nro TARGET -T "$root")" = /mnt/slwm-data
localpacks="$root/local-packs"
mkdir -p "$localpacks"
for role in dev train test; do
    : > "$localpacks/$role.sha256.partial"
    for family in paired impact pusht; do
        test -s "$root/$family/$role/card.json"
        file="$family-$role.tar.zst"
        if [[ ! -f $localpacks/$file ]]; then
            base=$(realpath "$root/$family/$role")
            if [[ $family == paired && -f $base.tar.zst ]]; then
                ln "$base.tar.zst" "$localpacks/$file"
            else
                tar -chf "$localpacks/$file.partial" --use-compress-program='zstd -q -1 -T2' -C "$root/$family" "$role"
                mv "$localpacks/$file.partial" "$localpacks/$file"
            fi
        fi
        (cd "$localpacks" && sha256sum "$file") >> "$localpacks/$role.sha256.partial"
    done
    mv "$localpacks/$role.sha256.partial" "$localpacks/$role.sha256"
done
(cd "$root" && sha256sum prior.npz) > "$localpacks/prior.sha256"
