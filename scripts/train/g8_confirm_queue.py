"""Score the fresh G8 panels only after every weight, fitted transfer head and input has been frozen."""
import sys
import time
import subprocess
from pathlib import Path
from training.g8_models import freeze

if __name__=="__main__":
    until=time.monotonic()+10*3600
    while time.monotonic()<until:
        if freeze() is not None:break
        time.sleep(20)
    else:raise TimeoutError("G8 confirmation inputs incomplete")
    sim=Path.home()/"dev/slwm-g5/.sim/lib/python3.12/site-packages"
    subprocess.run(["flock","/tmp/slwm_research_gpu.lock","bash","scripts/train/g4_cuda.sh","-m","training.g8_score","--sim-site",str(sim)],check=True)
    subprocess.run([sys.executable,"-u","-m","scripts.train.g8_analysis","/mnt/slwm-checkpoints-a/SLWM/g8/public"],check=True)
