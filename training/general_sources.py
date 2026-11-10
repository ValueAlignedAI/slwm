"""Official heterogeneous source acquisition, checksummed before any model fitting."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile
from training.research_sources import fetch,file_sha

ROOT=Path("artifacts/general_world/diverse_v1")


def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); temporary.replace(path)


def load_url(url):
    with urllib.request.urlopen(url,timeout=60) as response: return json.load(response)


def pin():
    path=ROOT/"source_plan.json"
    if path.exists(): return json.loads(path.read_text())
    ROOT.mkdir(parents=True,exist_ok=True); jobs=[]
    def add(family,name,url,**kwargs): jobs.append({"family":family,"name":name,"url":url,**kwargs})
    # These metadata responses were acquired separately before data/score access.
    clotho=json.loads((ROOT/"clotho_api.json").read_text())
    for item in clotho["files"]:
        add("clotho",item["key"],"https://zenodo.org/records/4783391/files/"+item["key"]+"?download=1",
            bytes=item["size"],md5=item["checksum"].removeprefix("md5:"),license_scope="Original per-file Freesound licences plus original caption LICENSE; research use")
    droid=json.loads((ROOT/"droid_objects.json").read_text())
    for item in droid["items"]:
        name=item["name"].rsplit("/",1)[-1]
        add("droid",name,"https://storage.googleapis.com/gresearch/"+item["name"]+"?generation="+item["generation"],
            bytes=int(item["size"]),md5_base64=item["md5Hash"],generation=item["generation"],license_scope="Official DROID100 debugging release; bounded exploratory action data, not a scale claim")
    for name,url in (("train2017.zip","https://s3.amazonaws.com/images.cocodataset.org/zips/train2017.zip"),
                     ("val2017.zip","https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip"),
                     ("annotations_trainval2017.zip","https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip")):
        with urllib.request.urlopen(urllib.request.Request(url,method="HEAD"),timeout=30) as response:
            add("coco",name,url,bytes=int(response.headers["Content-Length"]),etag=response.headers["ETag"],
                license_scope="COCO annotations CC BY4.0; preserve each image's original Flickr licence metadata")
    for family,url,maximum in (("pamap2","https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",800*1024**2),
                               ("mhealth","https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip",100*1024**2)):
        add(family,family+".zip",url,max_bytes=maximum,license_scope="UCI CC BY4.0; generic signal/activity research, no clinical endpoint")
    add("fresh_speech","dev-clean.tar.gz","https://www.openslr.org/resources/12/dev-clean.tar.gz",bytes=337926286,
        license_scope="Official LibriSpeech CC BY4.0; require exclusion of every prior used participant before new splits")
    release=json.loads((ROOT/"alignment_release.json").read_text())
    for item in release["assets"]:
        if item["name"]=="LibriSpeech-aligned-dev-clean.tar.gz":
            add("fresh_speech",item["name"],item["browser_download_url"],bytes=item["size"],github_asset_id=item["id"],
                sha256=(item.get("digest") or "").removeprefix("sha256:") or None,license_scope="Pinned MFA alignment release v1.0.0; preserve its original provenance")
    plan={"source_sha256":file_sha(__file__),"jobs":jobs,"workers":2,"minimum_free_bytes_before_job":15*1024**3,
          "declared_archive_bytes":sum(j.get("bytes",j.get("max_bytes",0)) for j in jobs),
          "scope":"Real datasets remain distinct; no synthetic pairing across COCO/Clotho/wearable/DROID/Libri sources. Counts/splits/content hashes validated before fitting. Existing corpora/checkpoints are not deleted."}
    write(path,plan); return plan


def download(job):
    path=ROOT/job["family"]/"sources"/job["name"]
    if not path.exists() and shutil.disk_usage(ROOT).free<15*1024**3: raise OSError("Less than15GiB free; archive verified sources before further acquisition")
    result=fetch(job["url"],path,job.get("sha256"))
    if "bytes" in job: assert result["bytes"]==job["bytes"],job["name"]
    if "max_bytes" in job: assert result["bytes"]<=job["max_bytes"],job["name"]
    if job.get("etag"): assert result["etag"]==job["etag"],job["name"]
    if "md5" in job or "md5_base64" in job:
        checksum=hashlib.md5()
        with path.open("rb") as stream:
            for block in iter(lambda:stream.read(8*1024**2),b""): checksum.update(block)
        if "md5" in job: assert checksum.hexdigest()==job["md5"],job["name"]
        if "md5_base64" in job: assert base64.b64encode(checksum.digest()).decode()==job["md5_base64"],job["name"]
    if path.suffix==".zip":
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                item=Path(name)
                if item.is_absolute() or ".." in item.parts: raise ValueError("Archive path traversal: "+name)
            failed=archive.testzip(); assert failed is None,(job["name"],failed)
    card={**job,"file":str(path),"source":result,"verified":True}
    write(path.with_suffix(path.suffix+".verified.json"),card)
    return card


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--stage",choices=("pin","download"),required=True)
    parser.add_argument("--family",action="append"); args=parser.parse_args(); plan=pin()
    if args.stage=="pin": print(json.dumps({"sources_pinned":len(plan["jobs"]),"archive_budget_bytes":plan["declared_archive_bytes"]}),flush=True); return
    jobs=[j for j in plan["jobs"] if not args.family or j["family"] in args.family]
    jobs.sort(key=lambda j:j.get("bytes",j.get("max_bytes",0)))
    results=[]; errors=[]
    with ThreadPoolExecutor(max_workers=plan["workers"]) as pool:
        pending={pool.submit(download,j):j for j in jobs}
        for future in as_completed(pending):
            job=pending[future]
            try:
                result=future.result(); results.append(result)
                print(json.dumps({"source_verified":job["family"]+"/"+job["name"],"bytes":result["source"]["bytes"]}),flush=True)
            except Exception as exc:
                errors.append({"job":job,"type":type(exc).__name__,"error":str(exc)})
                print(json.dumps({"source_failed":job["family"]+"/"+job["name"],"error":str(exc)}),flush=True)
            write(ROOT/"acquisition_progress.json",{"verified":results,"errors":errors,"requested":len(jobs)})
    write(ROOT/"acquisition_complete.json",{"verified":results,"errors":errors,"requested":len(jobs),"all_requested_verified":not errors})
    if errors: raise RuntimeError("Some source acquisitions failed; retain their logs and partials")


if __name__=="__main__": main()
