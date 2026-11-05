#!/usr/bin/env bash
set -euo pipefail
cd "${SLWM_ROOT:-/home/db/dev/slwm}"
CONTENT_PYTHON="${SLWM_PYTHON:-/home/db/venvs/vllm/bin/python}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}.research-venv/lib/python3.12/site-packages"
export OMP_NUM_THREADS=4
"$CONTENT_PYTHON" - <<'PY'
from pathlib import Path
import json,shutil
from training.research_sources import file_sha
from training.t2_real_av import write_json
root=Path("experiments/multimodal/t2/content-sync-20261105")
names=("evals/content_sync_probe.py","exploration/content_sync_infer.py","exploration/content_sync_outputs.py","evals/content_sync_intelligibility.py","evals/content_sync_audit.py","training/content_sync_report.py","scripts/train/finish_content_sync_fedora.sh","exploration/video_repair_audio.py","exploration/video_repair_infer.py")
hashes={}
for name in names:
    target=root/"evaluation_source_snapshot"/name; target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and file_sha(target)!=file_sha(name): raise ValueError("Evaluation source changed: "+name)
    if not target.exists(): shutil.copyfile(name,target)
    hashes[name]=file_sha(name)
write_json(root/"evaluation_source_hashes.json",hashes)
PY
"$CONTENT_PYTHON" -m evals.content_sync_probe
"$CONTENT_PYTHON" -m exploration.content_sync_outputs
"$CONTENT_PYTHON" -m evals.content_sync_intelligibility
"$CONTENT_PYTHON" -m evals.content_sync_audit
"$CONTENT_PYTHON" -m training.content_sync_report
