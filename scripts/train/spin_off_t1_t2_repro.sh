#!/usr/bin/env bash
set -euo pipefail

# Reproducible T1/T2 spin-off launcher.
#
# This starts long-running commands under nohup and writes logs under
# /data/slwm/logs/repro_training. It does not make model-quality claims; the
# produced registry artifacts must be reviewed together with data-control reports.

MODE="${1:-help}"
ROOT="/home/db/dev/slwm"
DATA_ROOT="${SLWM_DATA_ROOT:-/data/slwm}"
LOG_DIR="$DATA_ROOT/logs/repro_training"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
mkdir -p "$LOG_DIR"
cd "$ROOT"

if [[ -f "$DATA_ROOT/env_defaults.sh" ]]; then
  # shellcheck disable=SC1091
  source "$DATA_ROOT/env_defaults.sh"
fi
if [[ -f "$ROOT/.env.local" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env.local"
  set +a
fi

usage() {
  cat <<'EOF'
Usage: scripts/train/spin_off_t1_t2_repro.sh <mode>

Modes:
  data-controls       Run manifest-level T1/T2 data-control report now.
  t1-1b               Launch matched T1 124M 1B-token configs sequentially.
  t2-vggsound-124m    Launch VGGSound available-subset T2 124M + no-spectral sequentially.
  t2-smokes           Run external-control and VGGSound loader/training smokes now.
  describe            Describe larger T2 configs without training.

Outputs:
  Logs: /data/slwm/logs/repro_training/
  T1 registry: experiments/text/t1/EXP-T1-411..413/
  T2 registry: experiments/multimodal/t2/EXP-T2-111..112/

Claim limits:
  R-R0-9 leakage/dedup/eval-contamination and T1 PII/secrets scans must be
  reviewed before model-quality/training-ready claims.
EOF
}

run_data_controls() {
  "$PYTHON_BIN" -m training.t1_t2_data_controls \
    --out-dir "$DATA_ROOT/manifests/controls/t1_t2_manifest_controls_v0" \
    2>&1 | tee "$LOG_DIR/data_controls_$(date -u +%Y%m%dT%H%M%SZ).log"
}

launch_t1_1b() {
  local log="$LOG_DIR/t1_1b_124m_matched_$(date -u +%Y%m%dT%H%M%SZ).log"
  nohup bash -c 'set -euo pipefail
    cd /home/db/dev/slwm
    .venv/bin/python -m training.t1_torch_text --config configs/t1/gpt2_text_torch_124m_1b_v0.json
    .venv/bin/python -m training.t1_torch_text --config configs/t1/slwm_text_torch_124m_1b_v0.json
    .venv/bin/python -m training.t1_torch_text --config configs/t1/slwm_text_no_spectral_torch_124m_1b_v0.json
  ' > "$log" 2>&1 &
  printf 'launched_t1_1b_pid=%s\nlog=%s\n' "$!" "$log"
}

launch_t2_vggsound_124m() {
  local log="$LOG_DIR/t2_vggsound_124m_matched_$(date -u +%Y%m%dT%H%M%SZ).log"
  nohup bash -c 'set -euo pipefail
    cd /home/db/dev/slwm
    .venv/bin/python -m training.t2_train_latents --config configs/t2/slwm_124m_vggsound_available_v0.json
    .venv/bin/python -m training.t2_train_latents --config configs/t2/slwm_124m_vggsound_available_no_spectral_v0.json
  ' > "$log" 2>&1 &
  printf 'launched_t2_vggsound_124m_pid=%s\nlog=%s\n' "$!" "$log"
}

run_t2_smokes() {
  "$PYTHON_BIN" -m training.t2_train_latents \
    --config configs/t2/slwm_t2_external_control_smoke.json \
    --max-steps 2 --no-checkpoint \
    2>&1 | tee "$LOG_DIR/t2_external_control_smoke_$(date -u +%Y%m%dT%H%M%SZ).log"
  "$PYTHON_BIN" -m training.t2_train_latents \
    --config configs/t2/slwm_t2_vggsound_available_smoke.json \
    --max-steps 2 --no-checkpoint \
    2>&1 | tee "$LOG_DIR/t2_vggsound_available_smoke_$(date -u +%Y%m%dT%H%M%SZ).log"
}

describe() {
  "$PYTHON_BIN" -m training.t2_train_latents --config configs/t2/slwm_124m_vggsound_available_v0.json --describe-only
  "$PYTHON_BIN" -m training.t2_train_latents --config configs/t2/slwm_124m_vggsound_available_no_spectral_v0.json --describe-only
  "$PYTHON_BIN" -m training.t2_train_latents --config configs/t2/slwm_700m_audio_visual_24gb_fitcheck.json --describe-only
}

case "$MODE" in
  data-controls) run_data_controls ;;
  t1-1b) launch_t1_1b ;;
  t2-vggsound-124m) launch_t2_vggsound_124m ;;
  t2-smokes) run_t2_smokes ;;
  describe) describe ;;
  help|-h|--help) usage ;;
  *) usage >&2; exit 2 ;;
esac
