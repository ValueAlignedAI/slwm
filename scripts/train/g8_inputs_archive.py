"""Retain the new immutable evaluation arrays and official source archives on both checkpoint volumes."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from training.g8_models import G8_DATA,read
from training.g7_units import put
from scripts.train.g6_artifacts import roots
from scripts.train.g7_lock_heads import sha


def main():
    a,b=roots("g8");out=a/"data";out.mkdir(exist_ok=True);mirror=b/"data";mirror.mkdir(exist_ok=True)
    entries={}
    def pin(file):
        digest=sha(file);target=mirror/file.name
        if not target.is_file() or sha(target)!=digest:
            tmp=target.with_suffix(".partial");shutil.copyfile(file,tmp)
            if sha(tmp)!=digest:raise ValueError("Data replica failed physical readback")
            tmp.replace(target)
        entries[file.name]={"sha256":digest,"bytes":file.stat().st_size}
    files=[]
    for task in ("pusht","puck_barrier"):
        files.extend(str(Path("panels")/task/n) for n in ("card.json","scenes.npz"))
    files.extend(str(p.relative_to(G8_DATA)) for p in sorted((G8_DATA/"panels/reacher_test").glob("*.npy")));files.append("panels/reacher_test/card.json")
    for name,part in (("speech","test-other"),("speech-rate-confirm","dev-other")):
        root=G8_DATA/name;selection=read(root/"sources/selection.json");src=root/"sources"/(part+".tar.gz")
        if sha(src)!=selection["archive_sha256"]:raise ValueError("Original source archive changed")
        target=out/src.name
        if not target.is_file():tmp=target.with_suffix(".partial");shutil.copyfile(src,tmp);tmp.replace(target)
        if sha(target)!=selection["archive_sha256"]:raise ValueError("Stored original data changed")
        pin(target);files.append(str(Path(name)/"sources/selection.json"))
        for f in ("libri","crema"):
            folder=root/"confirm"/f;card=read(folder/"card.json");files.append(str((folder/"card.json").relative_to(G8_DATA)))
            for key,expected in card["sha256"].items():
                path=folder/(key+".npy")
                if sha(path)!=expected:raise ValueError("Held waveform array changed")
                files.append(str(path.relative_to(G8_DATA)))
    package=out/"g8-confirmation-inputs.tar.zst"
    if not package.exists():
        tmp=package.with_suffix(".partial")
        subprocess.run(["tar","--use-compress-program","zstd -q -1 -T2","-cf",str(tmp),"-C",str(G8_DATA),*sorted(files)],check=True);tmp.replace(package)
    pin(package);doc={"files":entries,"confirmation_members":len(files),"source_archives":"OpenSLR12 test-other and dev-other, CC BY 4.0",
        "shared_training_source":"Original G5/G7 data identities and masks are recorded in data-and-mask-audit.json. This package contains new G8 evaluation data."}
    for d in (out,mirror):put(d/"complete.json",doc)
    print("Independent input disk copies",sum(v["bytes"] for v in entries.values()),flush=True)

if __name__=="__main__":main()
