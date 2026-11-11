"""Release the shared Fedora GPU between waves of arrived immutable artifacts."""
import subprocess
import time
from pathlib import Path

BASE=Path("/mnt/slwm-data/SLWM/g10")
RUNS=Path("/mnt/slwm-checkpoints-a/SLWM/g10/runs")
OUT=BASE/"eval/real"
wanted={f"align-{c}-{mode}-s{seed}" for c in ("hybrid","gpt2") for mode in ("rooms","reconstruct","shifted","paired") for seed in (1001,1002,1003)}

def main():
    while True:
        left={tag for tag in wanted if not (OUT/(tag+".json")).exists()}
        if not left:return
        if any((RUNS/tag/"step-01536/verified.json").exists() for tag in left):
            subprocess.run(["flock","/tmp/slwm_research_gpu.lock","bash","scripts/train/g4_cuda.sh","evals/g10_real.py",str(RUNS),str(BASE/"data/holo"),
                            "/mnt/slwm-data/SLWM/g9/data",str(OUT),"--cache",str(BASE/"cache"),"--ready-only"],check=True)
        else:time.sleep(10)

if __name__=="__main__":main()
