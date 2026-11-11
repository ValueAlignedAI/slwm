"""Score verified scale fits on the same held rows without reserving the GPU while waiting."""
import argparse
import json
from pathlib import Path
import subprocess
import time

def main(data):
    r=Path("/mnt/slwm-checkpoints-a/SLWM/g11");out=data/"eval/scale-predictions";out.mkdir(exist_ok=True,parents=True)
    wanted={f"g11s-{f}-{size}-{c}-paired-s{s}" for f in "AB" for size in ("small","large") for c in ("hybrid","gpt2","gated") for s in (911,912)}
    while True:
        left={t for t in wanted if not (out/(t+"-1024.json")).exists()}
        if not left:return
        ready=sorted(tag for tag in left if (r/"scale"/tag/"step-01024/verified.json").is_file())
        if not ready:time.sleep(8);continue
        subprocess.run(["flock","/tmp/slwm_research_gpu.lock","bash","scripts/train/g4_cuda.sh","evals/g11_scale_predict.py",str(data/"data/banks"),str(r/"scale-parents"),str(r/"scale"),str(out),*ready,
                        "--cache",str(data/"cache")],check=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("data",type=Path);a=p.parse_args();main(a.data)
