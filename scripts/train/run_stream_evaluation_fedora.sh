#!/usr/bin/env bash
set -euo pipefail
cd "${SLWM_ROOT:-/home/db/dev/slwm}"
STREAM_PYTHON="${SLWM_PYTHON:-/home/db/venvs/vllm/bin/python}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}.research-venv/lib/python3.12/site-packages"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
while ! test -f experiments/multimodal/t2/broad-stream-20261105/selection.json; do sleep 20; done
"$STREAM_PYTHON" -m training.stream_sensing --stage fit
"$STREAM_PYTHON" -m training.stream_sensing --stage paced > experiments/policy/continuous-sensing-20261105/paced_cpu.log 2>&1 &
STREAM_PACED_PID=$!
"$STREAM_PYTHON" -m training.stream_sensing --stage evaluate
for STREAM_STAGE in speech fields av-repair long score; do
    "$STREAM_PYTHON" -m exploration.broad_stream_evaluate --stage "$STREAM_STAGE"
done
wait "$STREAM_PACED_PID"
CUDA_VISIBLE_DEVICES= "$STREAM_PYTHON" -m training.stream_exposure_audit
"$STREAM_PYTHON" -m training.broad_stream_report
while ! test -f experiments/text/t1/extended-throughput-20261105/summary.json; do sleep 20; done
"$STREAM_PYTHON" -m training.extended_text_report
"$STREAM_PYTHON" -m training.export_operational_artifacts
CUDA_VISIBLE_DEVICES= "$STREAM_PYTHON" -m training.operational_artifact_audit
"$STREAM_PYTHON" -m training.operational_registry --complete
