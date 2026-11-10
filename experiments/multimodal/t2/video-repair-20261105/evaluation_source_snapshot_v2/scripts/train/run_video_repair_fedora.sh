#!/usr/bin/env bash
set -euo pipefail
cd "${SLWM_ROOT:-/home/db/dev/slwm}"
REPAIR_PYTHON="${SLWM_PYTHON:-/home/db/venvs/vllm/bin/python}"
REPAIR_CONFIG="configs/t2/video_repair.json"
export OMP_NUM_THREADS=4
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}.research-venv/lib/python3.12/site-packages"

prepare_data() {
  if ! test -f artifacts/t2_audio_visual/video_repair_v1/crema_test_manifest.json; then
    "$REPAIR_PYTHON" -m training.video_repair_data --config "$REPAIR_CONFIG" --stage crema
  fi
  for pair in 'train_rav rav_train' 'val_rav rav_validation' 'test_rav rav_test'; do
    read -r stage split <<< "$pair"
    if ! test -f "artifacts/t2_audio_visual/video_repair_v1/${split}_manifest.json"; then
      "$REPAIR_PYTHON" -m training.video_repair_data --config "$REPAIR_CONFIG" --stage "$stage"
    fi
  done
}

train_matrix() {
  "$REPAIR_PYTHON" -m training.video_repair_train --config "$REPAIR_CONFIG" --stage anchors
  "$REPAIR_PYTHON" -m training.video_repair_train --config "$REPAIR_CONFIG" --stage branches
}

evaluate_matrix() {
  "$REPAIR_PYTHON" -m training.video_repair_select --config "$REPAIR_CONFIG"
  "$REPAIR_PYTHON" - <<'PY'
import json,shutil
from pathlib import Path
from exploration.video_repair_outputs import listening_cases
from training.t2_prepare_cremad import sha256
from training.t2_real_av import write_json
cfg=json.loads(Path("configs/t2/video_repair.json").read_text()); root=Path(cfg["artifacts"])
out=root/"exploration"; out.mkdir(exist_ok=True); cases=listening_cases(cfg)
path=out/"fixed_cases.json"
if path.exists() and json.loads(path.read_text())!=cases: raise ValueError("Listening IDs changed")
write_json(path,cases)
names=("training/video_repair_select.py","exploration/video_repair_audio.py","exploration/video_repair_infer.py","exploration/video_repair_outputs.py","evals/video_repair_intelligibility.py","evals/video_repair_audit.py","training/video_repair_report.py","docs/research/video_repair_decoder_diagnostic_20261105.md","scripts/train/run_video_repair_fedora.sh","tests/test_video_repair.py")
hashes={}
for name in names:
    source=Path(name); target=root/"evaluation_source_snapshot"/name; target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and sha256(target)!=sha256(source): raise ValueError("Evaluation source changed: "+name)
    if not target.exists(): shutil.copyfile(source,target)
    hashes[name]=sha256(source)
write_json(root/"evaluation_source_hashes.json",hashes)
PY
  "$REPAIR_PYTHON" -m training.video_repair_train --config "$REPAIR_CONFIG" --stage evaluate
  "$REPAIR_PYTHON" - <<'PY'
import json
from pathlib import Path
import torch
from training.video_repair_train import load_model,evaluate
from training.video_repair_common import load_data,get_codec
from training.t2_real_av import write_json
cfg=json.loads(Path("configs/t2/video_repair.json").read_text()); root=Path(cfg["artifacts"]); codec=get_codec(cfg); torch.set_num_threads(4)
for split in ("crema_test","rav_test"):
    data=load_data(cfg,split)
    for run in cfg["anchors"]+cfg["runs"]:
        model,_=load_model(root/run["id"]/"best.pt")
        for gap in (4,8,12):
            value=evaluate(model,codec,data,gap,detailed=False,eligible_only=False)
            value["scope"]="All-clip correct-video diagnostic; no invalid shifted-control claim"
            write_json(root/run["id"]/(split+f"_gap{gap}_allclips.json"),value)
        del model
    del data
    torch.cuda.empty_cache()
PY
  "$REPAIR_PYTHON" -m exploration.video_repair_outputs --config "$REPAIR_CONFIG"
  "$REPAIR_PYTHON" -m evals.video_repair_intelligibility --config "$REPAIR_CONFIG"
  "$REPAIR_PYTHON" -m evals.video_repair_audit --config "$REPAIR_CONFIG"
  "$REPAIR_PYTHON" -m training.video_repair_report --config "$REPAIR_CONFIG"
}

case "${1:-all}" in
  data) prepare_data ;;
  train) train_matrix ;;
  evaluate) evaluate_matrix ;;
  all) prepare_data; train_matrix; evaluate_matrix ;;
  *) echo 'Usage: run_video_repair_fedora.sh [all|data|train|evaluate]' >&2; exit 2 ;;
esac
