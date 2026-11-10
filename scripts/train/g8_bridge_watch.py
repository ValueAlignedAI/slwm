"""Publish immutable cross-wave bridge proofs to the physical research volumes as they complete."""
import json
from pathlib import Path
import subprocess
import time
from g7_inputs import FED
from g8_state import update

def main(ctl):
    until=time.time()+10*3600;sent=set()
    while time.time()<until:
        for p in sorted(ctl.glob("bridge-*.json")):
            if p.name in sent:continue
            d=json.loads(p.read_bytes())
            if not {"core","seed","expected_optimizer","observed"}<=set(d):continue
            subprocess.run([*FED,"cd /home/db/dev/slwm-g8 && /home/db/dev/slwm-g4/.jax-cpu/bin/python -m scripts.train.g8_custody"],input=p.read_bytes(),check=True)
            sent.add(p.name)
        update(ctl)
        budget=json.loads((ctl/"budget.json").read_bytes())
        if budget["phase"]=="complete":return
        time.sleep(25)
    raise TimeoutError("G8 bridge monitoring window elapsed")

if __name__=="__main__":
    import sys
    main(Path(sys.argv[1]))
