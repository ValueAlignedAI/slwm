#!/usr/bin/env bash
set -euo pipefail
python_bin="${SLWM_RESEARCH_PYTHON:-python}"
study="experiments/multimodal/t4/joint-transfer-20261109"
"$python_bin" -m training.joint_pusht --role screen
while [[ ! -f "$study/confirmation_lock.json" ]]; do
    sleep 45
done
"$python_bin" -m training.joint_pusht --role scale
