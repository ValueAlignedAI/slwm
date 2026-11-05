#!/usr/bin/env bash
set -euo pipefail
cd "${SLWM_ROOT:-/home/db/dev/slwm}"
DELIVERY_PYTHON="${SLWM_PYTHON:-/home/db/venvs/vllm/bin/python}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}.research-venv/lib/python3.12/site-packages"
export OMP_NUM_THREADS=4
"$DELIVERY_PYTHON" -m evals.scale_text_audit
"$DELIVERY_PYTHON" - <<'PY'
import json,subprocess,sys
from pathlib import Path
root=Path("experiments/multimodal/t2/content-sync-20261105"); bundle=root/"bundle"
raw=json.loads(Path("artifacts/t2_audio_visual/grid_content_v1/test.json").read_text())[0]["raw_path"]
subprocess.run([sys.executable,"-m","exploration.content_sync_infer","--checkpoint",str(bundle/"video.pt"),"--codec-dir",str(bundle),"--calibration",str(bundle/"calibration.json"),"--media",raw,"--gaps","1.32:1.64","--device","cpu","--output",str(root/"portable_cpu")],check=True)
text=Path("experiments/text/t1/scale-20261105")
for name in ("gpt2_seed81","slwm_seed81"):
    subprocess.run([sys.executable,"-m","exploration.scale_text_infer","--checkpoint",str(text/"bundle"/(name+".pt")),"--prompt","The purpose of a model is","--tokens","3","--device","cpu","--output",str(text/("portable_cpu_"+name+".json"))],check=True)
PY
"$DELIVERY_PYTHON" -m pytest -q tests/test_content_sync_scale.py
"$DELIVERY_PYTHON" -m exploration.content_sync_gallery
"$DELIVERY_PYTHON" -m training.package_content_scale
