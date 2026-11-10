#!/usr/bin/env bash
set -euo pipefail
tree=${1:?Git tree required}
[[ $tree =~ ^[0-9a-f]{40}$ ]]
base=/mnt/slwm-data/SLWM/g7/integration
cd "$base/$tree"
test "$(findmnt -nro TARGET -T "$PWD")" = /mnt/slwm-data
export PATH="$HOME/.local/bin:$PATH"
command -v rclone >/dev/null
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HOME=/mnt/slwm-data/SLWM/g7/hf-cache MPLCONFIGDIR="$base/cache-mpl"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="$HOME/dev/slwm/.research-venv/lib/python3.12/site-packages:$HOME/dev/slwm-g2/.g2-deps:$HOME/dev/slwm-g4/.jax-cpu/lib/python3.12/site-packages"
py="$HOME/venvs/vllm/bin/python"
"$py" - <<'PY'
from pathlib import Path
import cv2
assert cv2.__version__.startswith("4.12.") and hasattr(cv2,"CascadeClassifier")
src=Path.home()/"dev/slwm-g3/artifacts";dst=Path.cwd()/"artifacts";dst.mkdir(exist_ok=True)
for name in ("decoders","general_world","predictive_world"):
    p=dst/name
    if not p.exists():p.symlink_to(src/name,target_is_directory=True)
for name in ("t2_audio_visual","t1_text_code"):
    target=dst/name;target.mkdir(exist_ok=True)
    for f in (src/name).iterdir() if (src/name).is_dir() else ():
        q=target/f.name
        if f.is_dir() and not q.exists() and not q.is_symlink():q.symlink_to(f,target_is_directory=True)
PY
log="$base/logs/$tree.review.log"
set +e
"$py" -m pytest -q -rs -p no:cacheprovider --disable-warnings tests > "$log" 2>&1
rc=$?
set -e
RC="$rc" TREE="$tree" LOG="$log" SCRIPT="$HOME/dev/slwm-g7/scripts/train/g7_review_suite.sh" python3 - <<'PY'
import datetime,hashlib,json,os,re
from pathlib import Path
s=Path(os.environ["LOG"]).read_text(errors="replace")
last=s.splitlines()[-1] if s else ""
match=re.search(r"(\d+) passed",last)
doc={"tree":os.environ["TREE"],"exit":int(os.environ["RC"]),"passed":int(match[1]) if match else 0,
     "summary":last,"runner_sha256":hashlib.sha256(Path(os.environ["SCRIPT"]).read_bytes()).hexdigest(),"fixture_scope":"Pinned OpenCV 4.12, real prepared data, external Vocos, Fedora CPU",
     "finished_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
target=Path(os.environ["LOG"]).with_suffix(".json");tmp=target.with_suffix(".new")
tmp.write_text(json.dumps(doc,sort_keys=True)+"\n");tmp.replace(target)
print(json.dumps(doc))
PY
exit "$rc"
