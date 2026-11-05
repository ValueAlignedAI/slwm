#!/usr/bin/env bash
set -euo pipefail
cd "${SLWM_ROOT:-/home/db/dev/slwm}"
STREAM_PYTHON="${SLWM_PYTHON:-/home/db/venvs/vllm/bin/python}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}.research-venv/lib/python3.12/site-packages"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
"$STREAM_PYTHON" - <<'PY'
from pathlib import Path
import subprocess,sys,time
marker=Path("artifacts/t2_audio_visual/broad_speech_v1/raw/train-clean-100.tar.gz.extracted.json")
while not marker.exists(): time.sleep(5)
subprocess.run([sys.executable,"-m","training.broad_data","--stage","plan","--family","libri"],check=True)
subprocess.run([sys.executable,"-m","training.broad_data","--stage","prepare","--family","libri"],check=True)
PY
"$STREAM_PYTHON" - <<'PY'
from pathlib import Path
import json,subprocess,sys,time
cfg=json.loads(Path("configs/t2/broad_stream_data.json").read_text()); root=Path(cfg["ami_root"]); paths=[]
for names in cfg["ami_meetings"].values():
    for name in names:
        for index in range(4):
            paths += [root/"sources"/name/(f"{name}.Headset-{index}.wav.source.json"),root/"sources"/name/(f"{name}.Closeup{index+1}.avi.source.json")]
while not all(p.exists() for p in paths): time.sleep(5)
subprocess.run([sys.executable,"-m","training.broad_data","--stage","plan","--family","ami"],check=True)
subprocess.run([sys.executable,"-m","training.broad_data","--stage","prepare","--family","ami"],check=True)
subprocess.run([sys.executable,"-m","training.broad_data","--stage","codecs"],check=True)
PY
