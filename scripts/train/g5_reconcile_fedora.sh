#!/usr/bin/env bash
# Run only on the study account. Never start a worker on an unmounted root filesystem.
set -euo pipefail
data=/mnt/slwm-data/SLWM/g5
out=/mnt/slwm-checkpoints-a/SLWM/g5
mirror=/mnt/slwm-checkpoints-b/SLWM/g5
for pair in "$data /mnt/slwm-data" "$out /mnt/slwm-checkpoints-a" "$mirror /mnt/slwm-checkpoints-b"; do
    read -r path mount <<<"$pair"
    if [[ ! -d $path || $(findmnt -nro TARGET -T "$path") != "$mount" ]]; then
        echo "waiting-for-volume $mount"
        exit 10
    fi
done
test -s "$out/medium-residual-decision.json"
test -s "$data/prior.npz"
repo=/home/db/dev/slwm-g5
logs=/home/db/.local/state/slwm
mkdir -p "$logs"
if ! tmux has-session -t slwm-g5 2>/dev/null; then
    tmux new-session -d -s slwm-g5 -n recovered -c "$repo"
fi
if [[ ! -s $out/terminal-test.json ]] && flock -n /tmp/slwm-g5-scale.lock true; then
    win="scale-recovery-$(date +%s)"
    tmux new-window -d -t slwm-g5 -n "$win" -c "$repo"
    tmux pipe-pane -o -t "slwm-g5:$win" "cat >> $logs/g5-scale-recovery.log"
    command="PYTHONPATH=$repo OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu flock -n /tmp/slwm-g5-scale.lock /home/db/dev/slwm-g4/.jax-cpu/bin/python -u scripts/train/g5_scale_watch.py --root $out --mirror $mirror --data $data"
    tmux send-keys -l -t "slwm-g5:$win" -- "$command"
    tmux send-keys -t "slwm-g5:$win" Enter
    echo "restarted-registered-stage"
fi
python3 - <<'PY'
import json
from pathlib import Path
p=Path("/mnt/slwm-checkpoints-a/SLWM/g5")
files=sorted(p.glob("g5-anchor-*/progress.json"))
result={"terminal":(p/"terminal-test.json").is_file(),"anchor":[[v["run"],v["step"],v["target"]] for f in files for v in [json.loads(f.read_bytes())]]}
print(json.dumps(result))
PY
