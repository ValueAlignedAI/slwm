"""Freeze the deployment speech edge using only whole-utterance development likelihood."""
import json
from pathlib import Path
from scripts.train.g9_storage import write,sha
ROOT=Path("/mnt/slwm-data/SLWM/g14")
def main():
    rows=[]
    for directory in ("cue","cue-semantic"):
        file=ROOT/"work"/directory/"choices.json"
        for tag,entry in json.loads(file.read_bytes()).items():
            rows.append(dict(directory=directory,tag=tag,nll=entry["dev"]["nll"],joint=entry["dev"]["joint"],
                             step=entry["selected"],weights=entry["sha256"],choices=sha(file)))
    selected=min(rows,key=lambda x:(x["nll"],x["step"],x["tag"]))
    write(ROOT/"work/speech-choice.json",dict(**selected,selection="lowest development slot negative log likelihood",candidates=rows))
    print("Development-selected speech",selected)
if __name__=="__main__":main()
