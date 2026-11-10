"""Watch one named assignment, record its lifecycle and enforce a wall-clock allocation budget."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session",required=True); p.add_argument("--minutes",type=float,required=True)
    p.add_argument("--started",type=float,required=True); p.add_argument("--state",type=Path,required=True)
    a=p.parse_args()
    deadline=a.started+60*a.minutes
    base=Path.home()/".config/slwm-colab"
    env=dict(os.environ,CLOUDSDK_CONFIG=str(base/"gcloud"),COLAB_CLI_REQUEST_TIMEOUT="25")
    cli=[str(Path.home()/".local/bin/colab"),"--auth","adc"]
    a.state.parent.mkdir(parents=True,exist_ok=True)
    endpoint=None
    def write(phase):
        row={"session":a.session,"phase":phase,"checked_unix":time.time(),"deadline_unix":deadline}
        tmp=a.state.with_suffix(".tmp"); tmp.write_text(json.dumps(row)+"\n"); tmp.replace(a.state)
    while True:
        try:
            listed=subprocess.run([*cli,"sessions"],env=env,capture_output=True,text=True,timeout=40,check=True).stdout
            matches=[line.split()[1] for line in listed.splitlines() if line.startswith("["+a.session+"] ")]
            if matches:
                if endpoint and matches!=[endpoint]:
                    raise ValueError("Assignment identity changed")
                endpoint=matches[0]
                if time.time()>=deadline-45:
                    write("releasing")
                    subprocess.run([*cli,"stop","-s",a.session],env=env,capture_output=True,timeout=40,check=True)
                else:
                    write("active")
            elif endpoint or time.time()>=deadline:
                write("released" if endpoint else "no_assignment")
                return
            else:
                write("awaiting_allocation")
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError):
            write("provider_unreachable")
        time.sleep(12 if time.time()>=deadline-90 else 25)


if __name__=="__main__": main()
