"""Acquire the official larger speech source on archival storage, not GPU disk."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from training.general_sources import write
from training.research_sources import fetch,file_sha


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--root",required=True); args=parser.parse_args(); root=Path(args.root); root.mkdir(parents=True,exist_ok=True)
    pin=root/"source_plan.json"
    if pin.exists(): plan=json.loads(pin.read_text())
    else:
        with urllib.request.urlopen("https://www.openslr.org/resources/12/md5sum.txt",timeout=30) as response: text=response.read().decode()
        (root/"official_md5sum.txt").write_text(text)
        md5={line.split()[1]:line.split()[0] for line in text.splitlines() if line.strip()}
        release=json.loads(Path("artifacts/general_world/diverse_v1/alignment_release.json").read_text())
        item=next(a for a in release["assets"] if a["name"]=="LibriSpeech-aligned-train-clean-360.tar.gz")
        url="https://www.openslr.org/resources/12/train-clean-360.tar.gz"
        with urllib.request.urlopen(urllib.request.Request(url,method="HEAD"),timeout=30) as response: size=int(response.headers["Content-Length"])
        plan={"files":[{"name":"train-clean-360.tar.gz","url":url,"bytes":size,"md5":md5["train-clean-360.tar.gz"]},
             {"name":item["name"],"url":item["browser_download_url"],"bytes":item["size"],"sha256":(item.get("digest") or "").removeprefix("sha256:") or None,"github_asset_id":item["id"]}],
             "license":"LibriSpeech CC BY4.0; alignment release provenance retained","purpose":"Larger speech source. Preselect256 previously unexposed people by official metadata;192 train,16 selection,16 calibration,32 final. Full archive retained for later scale work; actual subset hours audited.","source_sha256":file_sha(__file__)}
        write(pin,plan)
    results=[]
    for item in plan["files"]:
        path=root/"sources"/item["name"]; result=fetch(item["url"],path,item.get("sha256")); assert result["bytes"]==item["bytes"]
        if item.get("md5"):
            h=hashlib.md5()
            with path.open("rb") as stream:
                for block in iter(lambda:stream.read(8*1024**2),b""): h.update(block)
            assert h.hexdigest()==item["md5"]
        card={**item,"verified":True,"source":result}; write(path.with_suffix(path.suffix+".verified.json"),card); results.append(card)
    write(root/"acquisition_complete.json",{"verified":results,"all_requested_verified":True})


if __name__=="__main__": main()
