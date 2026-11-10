"""Prepare the two additional Sony rooms for independent modality-invariance testing."""
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from g9_recordings import pack,starss_one

def main(root,dest):
    role="dev-train-sony"
    with zipfile.ZipFile(root/"video_dev.zip") as z:
        stems=sorted(Path(n).stem for n in z.namelist() if f"/{role}/" in n and n.endswith(".mp4"))
    if len(stems)!=28 or {n.split("_")[1] for n in stems}!={"room21","room22"}:raise ValueError("Unexpected additional room panel")
    with ThreadPoolExecutor(8) as pool:rows=list(zip(stems,pool.map(lambda n:starss_one(root,role,n),stems)))
    doc=pack(rows,dest/"starss-invariance.npz");(dest/"starss-invariance.json").write_text(json.dumps(doc,indent=2)+"\n")
    print("additional G9 rooms",len(rows),doc["ticks"],flush=True)

if __name__=="__main__":main(Path(sys.argv[1]),Path(sys.argv[2]))
