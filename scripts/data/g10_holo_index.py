"""Index official uncompressed HoloAssist tar files by verified HTTP byte ranges."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import tarfile
import requests

BASE = "https://hl2data.z5.web.core.windows.net/holoassist-data-release/"
ARCHIVES = {
    "video_compress": ('"0x8DC8C9DE268AE31"', 155285145600),
    "imu": ('"0x8DC8C9DD94B2BA2"', 4971059200),
}


def fetch(session, url, start, stop, etag):
    for attempt in range(5):
        try:
            res = session.get(url, headers={"Range": f"bytes={start}-{stop-1}", "If-Match": etag}, timeout=(12, 65))
            res.raise_for_status()
            if res.status_code != 206 or not res.headers.get("Content-Range", "").startswith(f"bytes {start}-{stop-1}/"):
                raise ValueError("Source did not honor the exact range")
            if res.headers.get("ETag") != etag or len(res.content) != stop-start:
                raise ValueError("Source generation or length changed")
            return res.content
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2**attempt)


def index(root, name):
    etag, total = ARCHIVES[name]
    url = BASE + name + ".tar"
    target = root / (name + "-index.json")
    checkpoint = root / (name + "-index.partial.json")
    if target.exists():
        return
    info = json.loads(checkpoint.read_text()) if checkpoint.exists() else {"offset": 0, "members": {}, "etag": etag, "bytes": total}
    if (info["etag"], info["bytes"]) != (etag, total):
        raise ValueError("Partial index refers to a different source")
    offset = info["offset"]
    session = requests.Session()
    buffer = b""
    origin = -65536
    def read(pos, size):
        nonlocal buffer, origin
        if not (origin <= pos and pos+size <= origin+len(buffer)):
            origin = pos
            buffer = fetch(session, url, pos, min(total,pos+max(8192,size)), etag)
        return buffer[pos-origin:pos-origin+size]
    pax = {}
    while offset+512 <= total:
        block = read(offset,512)
        if block == bytes(512):
            break
        member = tarfile.TarInfo.frombuf(block, encoding="utf8", errors="strict")
        data = offset+512
        size = int(pax.get("size", member.size))
        if member.type in (tarfile.XHDTYPE, tarfile.XGLTYPE):
            payload = read(data,member.size)
            pax = {}
            while payload:
                count = int(payload.split(b" ",1)[0])
                key,value = payload[:count].split(b" ",1)[1].rstrip(b"\n").split(b"=",1)
                pax[key.decode()] = value.decode()
                payload = payload[count:]
            size = member.size
        else:
            key = pax.get("path",member.name).lstrip("./")
            if member.isfile():
                info["members"][key] = [data,size]
            pax = {}
        offset = data + ((size+511)//512)*512
        if not pax and len(info["members"])%64==0:
            info["offset"]=offset;tmp=checkpoint.with_suffix(".new");tmp.write_text(json.dumps(info));tmp.replace(checkpoint)
    info["offset"]=offset
    tmp=target.with_suffix(".new");tmp.write_text(json.dumps(info,sort_keys=True));tmp.replace(target)
    print(name,len(info["members"]),"indexed",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path);p.add_argument("archive",choices=ARCHIVES)
    a=p.parse_args();index(a.root,a.archive)
