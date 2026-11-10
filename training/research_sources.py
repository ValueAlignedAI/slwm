"""Resumable public research downloads with bounded, respectful retries."""
from __future__ import annotations
import email.utils
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(4*1024*1024),b""): h.update(part)
    return h.hexdigest()


def fetch(url,path,expected_sha=None):
    """Download atomically, honor Retry-After; partial files can resume."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); card=path.with_suffix(path.suffix+".source.json")
    if path.exists() and card.exists():
        d=json.loads(card.read_text())
        if file_sha(path)!=d["sha256"] or (expected_sha and d["sha256"]!=expected_sha): raise ValueError("Source bytes changed: "+str(path))
        return d
    partial=path.with_suffix(path.suffix+".part")
    for attempt in range(6):
        begin=partial.stat().st_size if partial.exists() else 0
        try:
            request=urllib.request.Request(url,headers={"Range":f"bytes={begin}-"} if begin else {})
            with urllib.request.urlopen(request,timeout=90) as response:
                if begin and response.status==206:
                    if not response.headers.get("Content-Range","").startswith(f"bytes {begin}-"): raise ValueError("Wrong resume range")
                    mode="ab"
                else: mode="wb"; begin=0
                remaining=response.headers.get("Content-Length"); total=begin+int(remaining) if remaining else None
                metadata={"url":url,"etag":response.headers.get("ETag"),"last_modified":response.headers.get("Last-Modified"),"expected_sha256":expected_sha}
                with partial.open(mode) as f:
                    for chunk in iter(lambda:response.read(1024*1024),b""): f.write(chunk)
            if total is not None and partial.stat().st_size!=total: raise OSError("Truncated public download")
            digest=file_sha(partial)
            if expected_sha and digest!=expected_sha: raise ValueError("Published SHA256 mismatch: "+str(path))
            partial.replace(path); metadata.update({"sha256":digest,"bytes":path.stat().st_size})
            card.write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n"); print(json.dumps({"downloaded":str(path),"bytes":metadata["bytes"]}),flush=True); return metadata
        except urllib.error.HTTPError as exc:
            if exc.code!=429 and exc.code<500: raise
            value=exc.headers.get("Retry-After","")
            if value.isdigit(): delay=float(value)
            else:
                try: delay=max(0,email.utils.parsedate_to_datetime(value).timestamp()-time.time())
                except (ValueError,TypeError): delay=30.*(attempt+1)
            print(json.dumps({"source_retry":exc.code,"seconds":delay,"path":str(path)}),flush=True); time.sleep(delay)
        except (urllib.error.URLError,TimeoutError,OSError) as exc:
            if attempt==5: raise
            print(json.dumps({"source_retry":type(exc).__name__,"path":str(path)}),flush=True); time.sleep(10*(attempt+1))
    raise RuntimeError("Public download retry budget exhausted")


def main():
    import argparse
    from concurrent.futures import ThreadPoolExecutor
    p=argparse.ArgumentParser(); p.add_argument("--kind",choices=("grid","text"),required=True); p.add_argument("--root",required=True); a=p.parse_args(); root=Path(a.root); root.mkdir(parents=True,exist_ok=True)
    if a.kind=="grid":
        people=list(range(1,21))+list(range(22,33)); jobs=[]
        for person in people:
            for suffix in (f"video/s{person}.mpg_vcd.zip",f"align/s{person}.tar"):
                url=f"https://spandh.dcs.shef.ac.uk/gridcorpus/s{person}/{suffix}"; jobs.append((url,root/"sources"/f"s{person:02d}"/Path(suffix).name,None))
        (root/"source_scope.json").write_text(json.dumps({"train_people":list(range(1,17)),"selection_people":[17,18,19,20],"calibration_people":[22,23,24],"reserved_test_people":list(range(25,33)),"unopened_lockbox":[33,34],"unavailable_video":21,"source":"https://spandh.dcs.shef.ac.uk/gridcorpus/","permission":"Freely available for research use, per official source"},indent=2)+"\n")
    else:
        revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
        with urllib.request.urlopen(f"https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu/tree/{revision}/sample/10BT?limit=100") as r: listing=json.load(r)
        chosen=sorted(listing,key=lambda d:d["path"])[:5]; (root/"pinned_sources.json").write_text(json.dumps({"dataset":"HuggingFaceFW/fineweb-edu","revision":revision,"license":"ODC-By","files":chosen},indent=2)+"\n")
        jobs=[(f"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/{revision}/{d['path']}",root/"sources"/Path(d["path"]).name,d["lfs"]["oid"]) for d in chosen]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for _ in pool.map(lambda j:fetch(*j),jobs): pass
    print(json.dumps({"acquisition_complete":a.kind,"files":len(jobs)}),flush=True)


if __name__=="__main__": main()
