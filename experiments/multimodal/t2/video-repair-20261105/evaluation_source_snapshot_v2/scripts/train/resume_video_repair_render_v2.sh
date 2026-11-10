#!/usr/bin/env bash
set -euo pipefail
cd "${SLWM_ROOT:-/home/db/dev/slwm}"
REPAIR_PYTHON="${SLWM_PYTHON:-/home/db/venvs/vllm/bin/python}"
export OMP_NUM_THREADS=4
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}.research-venv/lib/python3.12/site-packages"
"$REPAIR_PYTHON" - <<'PY'
import json,shutil
from pathlib import Path
from exploration.video_repair_outputs import listening_cases
from training.t2_prepare_cremad import sha256
from training.t2_real_av import write_json
cfg=json.loads(Path("configs/t2/video_repair.json").read_text()); root=Path(cfg["artifacts"])
failed=root/"exploration_initial_edge_padding_error"
if not failed.exists():
    (root/"exploration").rename(failed)
    shutil.copyfile(root/"postprocess.log",root/"postprocess_initial_edge_padding_error.log")
assert json.loads((failed/"fixed_cases.json").read_text())==listening_cases(cfg)
old=json.loads((root/"evaluation_source_hashes.json").read_text())
names=list(old)+["docs/research/video_repair_output_correction_20261105.md","scripts/train/resume_video_repair_render_v2.sh"]
hashes={}
for name in names:
    path=Path(name); copy=root/"evaluation_source_snapshot_v2"/name; copy.parent.mkdir(parents=True,exist_ok=True)
    if copy.exists() and sha256(copy)!=sha256(path): raise ValueError("Revision2 source changed: "+name)
    if not copy.exists(): shutil.copyfile(path,copy)
    hashes[name]=sha256(path)
write_json(root/"evaluation_source_hashes_v2.json",hashes)
write_json(root/"render_correction.json",{"revision":2,"changed_original_files":[name for name,digest in old.items() if hashes[name]!=digest],
    "reason":"Repeat at most three terminal analyzed subframes, as in original packetize; globally erased input only",
    "feature_metrics_and_training_unchanged":True,"fixed_ids_unchanged":True,"original_source_snapshot_preserved":True})
PY
"$REPAIR_PYTHON" -m exploration.video_repair_outputs --config configs/t2/video_repair.json
"$REPAIR_PYTHON" -m evals.video_repair_intelligibility --config configs/t2/video_repair.json
"$REPAIR_PYTHON" -m evals.video_repair_audit --config configs/t2/video_repair.json
"$REPAIR_PYTHON" -m training.video_repair_report --config configs/t2/video_repair.json
"$REPAIR_PYTHON" -m exploration.video_repair_gallery --config configs/t2/video_repair.json
