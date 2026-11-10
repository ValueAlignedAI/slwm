#!/usr/bin/env bash
set -euo pipefail
cd "${SLWM_ROOT:-/home/db/dev/slwm}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}.research-venv/lib/python3.12/site-packages"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
GENERAL_PYTHON="${SLWM_PYTHON:-/home/db/venvs/vllm/bin/python}"
"$GENERAL_PYTHON" - <<'PY'
import json,time
from pathlib import Path
root=Path('experiments/multimodal/t2/general-transfer-20261106')
deadline=time.monotonic()+48*3600
while not (root/'training_complete.json').exists():
    pid=json.loads((root/'pipeline_pid.json').read_text())['pid']
    process=Path('/proc')/str(pid)
    if not process.exists() or process.joinpath('stat').read_text().split()[2]=='Z':
        raise RuntimeError('G1 trainer exited before all weights froze; inspect pipeline.log and retain partial artifacts')
    if time.monotonic()>deadline:raise TimeoutError('G1 evaluation wait exceeded48h')
    time.sleep(30)
PY
"$GENERAL_PYTHON" -m training.general_evaluation_lock --stage register
"$GENERAL_PYTHON" -m training.general_readout_train --stage features
"$GENERAL_PYTHON" -m training.general_readout_train --stage fit
if [[ ! -f experiments/multimodal/t2/general-transfer-20261106/speech/calibration.json ]]; then
  "$GENERAL_PYTHON" -m training.general_speech_calibrate
fi
"$GENERAL_PYTHON" -m training.general_evaluation_lock --stage freeze
"$GENERAL_PYTHON" -m training.general_readout_eval
"$GENERAL_PYTHON" -m training.general_speech_eval
"$GENERAL_PYTHON" -m training.general_consequences
"$GENERAL_PYTHON" -m training.general_alignment_eval
"$GENERAL_PYTHON" -m training.general_live_eval
"$GENERAL_PYTHON" -m training.general_long_media
"$GENERAL_PYTHON" -m training.general_report
