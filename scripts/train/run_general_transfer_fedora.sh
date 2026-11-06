#!/usr/bin/env bash
set -euo pipefail
cd "${SLWM_ROOT:-/home/db/dev/slwm}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}.research-venv/lib/python3.12/site-packages"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
GENERAL_PYTHON="${SLWM_PYTHON:-/home/db/venvs/vllm/bin/python}"
"$GENERAL_PYTHON" - <<'PY'
import json
from pathlib import Path
p=Path('artifacts/general_world/diverse_v1/ami/acquisition_progress.json')
d=json.loads(p.read_text())
assert len(d['verified'])==d['requested']==168 and not d['errors'], 'All fixed AMI sources must verify before fitting'
PY
"$GENERAL_PYTHON" -m training.general_ami --stage prepare
if [[ ! -f artifacts/general_world/diverse_v1/prepared/evaluation_cases.json ]]; then
  "$GENERAL_PYTHON" -m training.general_transfer_cases
fi
"$GENERAL_PYTHON" -m training.general_dataset --stage freeze
"$GENERAL_PYTHON" -m pytest -q tests/test_general_data.py tests/test_general_world.py tests/test_general_batch.py tests/test_general_checkpoint.py tests/test_phonetic_renderer.py
"$GENERAL_PYTHON" -m training.general_train --stage smoke
"$GENERAL_PYTHON" -m training.general_train --stage train
