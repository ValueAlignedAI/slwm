"""Prepare the pinned, headless Language-Table environment on the research computer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from urllib.request import Request,urlopen

ROOT=Path("/mnt/slwm-data/SLWM/g14")
UPSTREAM="42b319d46ec408cfe382cdbf45e71b6cf4be2900"
RUNTIME=Path.home()/".local/state/slwm-native/g14-py311"

def json_file(path,value):
    tmp=path.with_suffix(".new");tmp.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n");tmp.replace(path)


def simulator():
    upstream=ROOT/"upstream/language-table"
    if not upstream.is_dir():subprocess.run(["git","clone","https://github.com/google-research/language-table.git",str(upstream)],check=True)
    if subprocess.check_output(["git","-C",str(upstream),"rev-parse","HEAD"],text=True).strip()!=UPSTREAM:
        subprocess.run(["git","-C",str(upstream),"fetch","origin",UPSTREAM],check=True)
        subprocess.run(["git","-C",str(upstream),"checkout","--detach",UPSTREAM],check=True)
    folder=ROOT/"upstream/runtime";shutil.copytree(upstream/"language_table",folder/"language_table",dirs_exist_ok=True)
    util=folder/"language_table/environments/utils/utils_pybullet.py";s=util.read_text()
    if s.count("import tensorflow as tf")!=1 or s.count("with tf.io.gfile.GFile")!=2:raise ValueError("Unrecognized state-I/O file")
    # TensorFlow is used only by upstream's optional serialized-file helpers. Preserve those helpers and defer their import.
    util.write_text(s.replace("import tensorflow as tf\n","").replace("  with tf.io.gfile.GFile","  import tensorflow as tf\n  with tf.io.gfile.GFile"))
    py=Path.home()/".local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11"
    uv=Path.home()/".local/bin/uv";env=dict(os.environ,UV_CACHE_DIR=str(ROOT/"cache/uv"))
    if not (RUNTIME/"bin/python").exists():
        RUNTIME.parent.mkdir(exist_ok=True,parents=True);subprocess.run([str(uv),"venv","--python",str(py),str(RUNTIME)],check=True,env=env)
    pins=["pybullet==3.2.7","gym==0.23.0","numpy==1.26.4","scipy==1.15.3","opencv-python-headless==4.11.0.86","six==1.17.0","absl-py==2.3.1","Pillow==11.3.0"]
    subprocess.run([str(uv),"pip","install","--python",str(RUNTIME/"bin/python"),*pins],env=env,check=True)
    json_file(ROOT/"work/simulator-source.json",dict(repository="https://github.com/google-research/language-table",commit=UPSTREAM,pins=pins,
           patch="Deferred the TensorFlow import in two optional serialized-file helpers",modified_sha256=hashlib.sha256(util.read_bytes()).hexdigest()))
    print("Native simulator ready",UPSTREAM,flush=True)


def voices():
    base="https://spandh.dcs.shef.ac.uk/gridcorpus";root=ROOT/"data/grid-new";root.mkdir(exist_ok=True,parents=True)
    cases=[(person,kind) for person in (33,34) for kind in ("video","align")]
    def one(arg):
        person,kind=arg;name=f"s{person}.mpg_vcd.zip" if kind=="video" else f"s{person}.tar"
        url=f"{base}/s{person}/{kind}/{name}";path=root/name;tmp=path.with_suffix(path.suffix+".part")
        with urlopen(Request(url,method="HEAD"),timeout=30) as meta:
            size=int(meta.headers["Content-Length"]);etag=meta.headers["ETag"]
        if not path.exists() or path.stat().st_size!=size:
            for repeat in range(4):
                start=tmp.stat().st_size if tmp.exists() else 0
                try:
                    request=Request(url,headers={"Range":f"bytes={start}-","If-Match":etag} if start else {"If-Match":etag})
                    with urlopen(request,timeout=100) as src,tmp.open("ab" if start and src.status==206 else "wb") as out:shutil.copyfileobj(src,out,1<<20)
                    if tmp.stat().st_size!=size:raise IOError("Incomplete GRID object")
                    tmp.replace(path);break
                except OSError:
                    if repeat==3:raise
        with path.open("rb") as f:h=hashlib.file_digest(f,"sha256").hexdigest()
        print("Verified GRID source",name,size,flush=True);return dict(person=person,kind=kind,name=name,url=url,etag=etag,bytes=size,sha256=h)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(4) as pool:docs=list(pool.map(one,cases))
    json_file(root/"sources.json",docs)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("simulator","voices"));a=p.parse_args()
    (simulator if a.stage=="simulator" else voices)()
