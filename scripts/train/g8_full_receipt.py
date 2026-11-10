"""Verify full-utterance recognition counts and publish paired per-reader evidence without raw transcripts."""
import json
import re
import shutil
from pathlib import Path
from scripts.train.g6_artifacts import roots
from scripts.train.g7_lock_heads import sha
from training.g7_units import put
from training.g8_models import G8_DATA,read


def main():
    a,b=roots("g8");folder=a/"full-speech";second=b/"full-speech";second.mkdir(exist_ok=True)
    meta=read(folder/"complete.json");obs=read(folder/"recognitions.json");sel=read(G8_DATA/"full-speech/selection.json")
    if meta["rows_sha256"]!=sha(folder/"recognitions.json") or obs["source_selection"]!=sha(G8_DATA/"full-speech/selection.json"):raise ValueError("Recognition source changed")
    counts=[len(re.sub("[^a-z0-9 ]"," ",r["reference"].lower().replace("'","").replace("’","")).split()) for r in sel["utterances"]]
    if len(counts)!=meta["clips"] or sum(counts)!=meta["word_count"]:raise ValueError("Independent denominator disagreement")
    for tag,scores in meta["scorers"].items():
        for cond,v in scores["conditions"].items():
            errors=obs["edits"][tag][cond]
            if sum(errors)!=v["edits"] or len(errors)!=len(counts) or abs(sum(errors)/sum(counts)-v["word_error"])>1e-12:raise ValueError("Recognizer row count disagreement")
    source=G8_DATA/"full-speech/test-clean.tar.gz"
    for dest in (folder,second):
        target=dest/"test-clean.tar.gz"
        if not target.exists():
            temp=target.with_suffix(".partial");shutil.copyfile(source,temp);temp.replace(target)
        if sha(target)!=sel["source_sha256"]:raise ValueError("Official source archive does not match its physical replica")
    for name in ("complete.json","recognitions.json","lock.json"):
        if not (second/name).exists() or sha(second/name)!=sha(folder/name):shutil.copyfile(folder/name,second/name)
        if sha(second/name)!=sha(folder/name):raise ValueError("Second recognition replica differs")
    out=a/"public";out.mkdir(exist_ok=True)
    put(out/"full-utterance.json",{**meta,"official_archive_sha256":sel["source_sha256"],"official_archive_md5":sel["source_md5"],
        "excluded_overlapping_speakers":sel["excluded_overlapping_speakers"],"ids":[r["id"] for r in sel["utterances"]],
        "speaker":[r["speaker"] for r in sel["utterances"]],"reference_word_counts":counts,"per_recording_edits":obs["edits"]})
    print("Both scorers and source archives independently verified",len(counts),sum(counts),flush=True)


if __name__=="__main__":main()
