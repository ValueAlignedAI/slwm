"""Fetch a fixed external audiovisual falsification subset using public ZIP ranges.

Only selected full-AV RAVDESS members are fetched; ZIP CRC and file SHA256 are
verified. No external sample is used to fit a model or preprocessing statistic.
"""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import urllib.request
import zipfile
import zlib

from training.t2_prepare_cremad import sha256


class RangeFile(io.RawIOBase):
    """Seekable public HTTP byte-range reader with one-megabyte block caching."""
    def __init__(self,url: str,size: int):
        self.url,self.size,self.position=url,size,0
        self.blocks={}; self.block_size=1024*1024
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        return self.position
    def read(self,n=-1):
        n=self.size-self.position if n<0 else min(n,self.size-self.position)
        if n<=0: return b""
        pieces=[]
        while n:
            block=self.position//self.block_size; start=block*self.block_size
            if block not in self.blocks:
                end=min(start+self.block_size,self.size)-1
                request=urllib.request.Request(self.url,headers={"Range":f"bytes={start}-{end}"})
                with urllib.request.urlopen(request,timeout=60) as response:
                    if response.status!=206: raise RuntimeError("Server did not honor public ZIP range request")
                    payload=response.read()
                if len(payload)!=end-start+1: raise RuntimeError("Truncated ZIP range")
                self.blocks[block]=payload
            chunk=self.blocks[block]; offset=self.position-start; take=min(n,len(chunk)-offset)
            pieces.append(chunk[offset:offset+take]); self.position+=take; n-=take
        return b"".join(pieces)


def prepare(out_dir: str):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    metadata=json.load(urllib.request.urlopen("https://zenodo.org/api/records/1188976",timeout=30))
    (out/"source_record.json").write_text(json.dumps(metadata,indent=2))
    rows=[]; failures=[]
    for actor in (1,2,13,14):
        key=f"Video_Speech_Actor_{actor:02d}.zip"
        file=next(f for f in metadata["files"] if f["key"]==key)
        url=f"https://zenodo.org/records/1188976/files/{key}?download=1"
        reader=RangeFile(url,file["size"])
        with zipfile.ZipFile(reader) as archive:
            selected=[]
            for member in archive.infolist():
                name=Path(member.filename).name; fields=Path(name).stem.split("-")
                if len(fields)==7 and fields[0]=="01" and fields[1]=="01" and fields[3]=="01" and fields[5]=="01" and name.endswith(".mp4"):
                    selected.append(member)
            if len(selected)!=16: raise ValueError(f"Unexpected external selection for actor {actor}: {len(selected)}")
            for member in sorted(selected,key=lambda m:m.filename):
                name=Path(member.filename).name; path=out/"raw"/name; path.parent.mkdir(exist_ok=True)
                try:
                    if path.exists() and zlib.crc32(path.read_bytes())!=member.CRC:
                        path.unlink()
                    if not path.exists():
                        payload=archive.read(member)  # zipfile checks the complete member CRC.
                        tmp=path.with_suffix(".part"); tmp.write_bytes(payload); tmp.replace(path)
                    fields=path.stem.split("-")
                    rows.append({"sample_id":path.stem,"actor":f"RAVDESS-{actor:02d}","raw_path":str(path),"sha256":sha256(path),
                        "statement":fields[4],"transcript":"Kids are talking by the door." if fields[4]=="01" else "Dogs are sitting by the door.",
                        "emotion_code":fields[2],"zip_member_crc32":member.CRC,"source_archive":key,"source_archive_declared_md5":file["checksum"],
                        "license":"CC BY-NC-SA 4.0; evaluation-only, not used for fitting"})
                    print(json.dumps({"external_downloaded":len(rows),"sample":path.stem}),flush=True)
                except Exception as exc: failures.append({"sample_id":path.stem,"error":repr(exc)})
        del reader
    card={"source":"https://zenodo.org/records/1188976","authors":"Livingstone & Russo (2018)","license":"CC BY-NC-SA 4.0",
        "purpose":"untouched external falsification; never fitting or model selection","selected":64,"usable":len(rows),"failures":failures,"records":rows}
    (out/"dataset_card.json").write_text(json.dumps(card,indent=2)+"\n")
    return card


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output",default="artifacts/t2_audio_visual/ravdess_external_v1")
    prepare(parser.parse_args().output)
