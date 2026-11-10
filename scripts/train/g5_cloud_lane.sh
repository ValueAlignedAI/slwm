#!/usr/bin/env bash
# One architecture per native, authorized v6e assignment. No session extension.
set -euo pipefail
umask 077
core=${1:?hybrid or gpt2}
deadline=${2:?absolute study deadline}
scale=${3:-medium}
[[ $core == hybrid || $core == gpt2 ]]
case "$scale" in medium) steps=4096; micro=128; cut=1024;; anchor) steps=8192; micro=64; cut=2048;; *) exit 2;; esac
cd /content/g5-code
export RCLONE_CONFIG=/dev/shm/slwm-drive/rclone.conf RCLONE_LOG_LEVEL=ERROR
source /content/g4-topology.env
export JAX_PLATFORMS=tpu JAX_COMPILATION_CACHE_DIR=/content/g5-xla OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3
python -c 'import optax,zstandard,scipy' || python -m pip -q install optax zstandard scipy
python -c 'import jax; d=jax.devices()[0]; print(d.platform,d.device_kind,flush=True); assert d.platform=="tpu" and ("v6" in d.device_kind.lower() or "trillium" in d.device_kind.lower())'
mkdir -p /content/g5-out /content/g5-spool /content/g5-data
python scripts/train/g5_cloud_data.py train --root /content/g5-data --pins pins &
train_fetch=$!
python scripts/train/g5_cloud_data.py dev --root /content/g5-data --pins pins
wait "$train_fetch"
cp pins/prior.npz /content/g5-data/prior.npz
(cd /content/g5-data && sha256sum -c /content/g5-code/pins/prior.sha256)
python -u scripts/train/colab_artifacts.py sync --spool /content/g5-spool --remote slwm-drive:SLWM/g5/resumable --watch >/content/g5-sync.log 2>&1 &
sync_pid=$!
trap 'kill "$sync_pid" 2>/dev/null || true' EXIT
for seed in 511 512; do
    # Reserve a full hour at the end of the authorized wall time for durable export.
    (( $(date +%s) + 3600 < deadline )) || break
    name="g5-$scale-$core-$seed-residual-v6e"
    path=/content/g5-out/$name
    markers=$(rclone lsf --files-only "slwm-drive:SLWM/g5/resumable/runs/$name" 2>/dev/null || true)
    if [[ -n $markers && ! -e $path ]]; then
        python scripts/train/colab_artifacts.py restore --remote slwm-drive:SLWM/g5/resumable --run "$name" --to "$path"
        # A verified terminal optimizer may predate its separate inference export.
        # Supply the recovered counter only; missing timing or loss must stay unknown.
        python - "$path" <<'RECOVER'
import json
from pathlib import Path
import sys
import numpy as np

folder=Path(sys.argv[1])
card=json.loads((folder/"config.json").read_bytes())
with np.load(folder/"resume.npz",allow_pickle=False) as f:
    count=int(f["step"])
if count==card["steps"] and not (folder/"progress.json").exists():
    (folder/"progress.json").write_text(json.dumps({"run":card["run"],"step":count,"target":count,
        "restored_from_verified_optimizer":True,"timing_available":False},sort_keys=True)+"\n")
RECOVER
    fi
    python -u -m training.g5_residual_fit --backend tpu --data /content/g5-data --output /content/g5-out \
        --core "$core" --seed "$seed" --scale "$scale" --steps "$steps" --micro "$micro" --label residual-v6e \
        --scan 4 --save-every "$cut" --spool /content/g5-spool
    python -u -m training.g5_eval --backend tpu --data /content/g5-data --role dev --run "$path" --out "$path"
    python -u scripts/train/g5_promote.py "$path"
    for file in dev-metrics.json dev-predictions.npz verified.json; do
        remote="slwm-drive:SLWM/g5/checkpoints/$name/$file"
        rclone copyto --immutable --checksum "$path/$file" "$remote"
        expected=$(sha256sum "$path/$file" | awk '{print $1}')
        actual=$(rclone hashsum SHA-256 "$remote" | awk 'NR==1{print $1}')
        [[ $actual == "$expected" ]]
    done
    echo "Durable endpoint $name"
done
