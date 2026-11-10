#!/usr/bin/env bash
set -euo pipefail
cd "${SLWM_ROOT:-/home/db/dev/slwm}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}.research-venv/lib/python3.12/site-packages"
TEXT_PYTHON="${SLWM_PYTHON:-/home/db/venvs/vllm/bin/python}"
"$TEXT_PYTHON" - <<'PY'
import time
from pathlib import Path
root=Path('experiments/text/t1/500m-20261106');deadline=time.monotonic()+48*3600
while not (root/'summary.json').exists():
    if time.monotonic()>deadline:raise TimeoutError('Text report wait exceeded48h; inspect driver.log')
    time.sleep(30)
PY
"$TEXT_PYTHON" -m training.text_500m_report --stage report
