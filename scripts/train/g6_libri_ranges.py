"""Resume the official 360-hour release with parallel verified byte ranges on Fedora's data SSD."""
import argparse
import errno
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

URLS=("https://www.openslr.org/resources/12/train-clean-360.tar.gz",
      "https://openslr.trmal.net/resources/12/train-clean-360.tar.gz",
      "https://openslr.elda.org/resources/12/train-clean-360.tar.gz")
TOTAL=23049477885
EXPECTED="c0e676e450a7ff2f54aeade5171606fa"
BLOCK=64*1024**2


def acquire(root,jobs):
    sources=root/"sources"; archive=sources/"train-clean-360.tar.gz"
    if archive.exists():
        with archive.open("rb") as f:
            if hashlib.file_digest(f,"md5").hexdigest()!=EXPECTED: raise ValueError("Existing release MD5 mismatch")
        return archive
    prefix=sources/"train-clean-360.tar.gz.partial"
    prefix_size=prefix.stat().st_size if prefix.exists() else 0
    if not 0<=prefix_size<TOTAL or shutil.disk_usage(sources).free<2*TOTAL:
        raise ValueError("Invalid original prefix or insufficient room on the data SSD")
    parts=sources/("libri360-ranges-"+str(prefix_size))
    parts.mkdir(exist_ok=True)
    ranges=[(n,min(n+BLOCK,TOTAL)-1) for n in range(prefix_size,TOTAL,BLOCK)]
    def fetch(pair):
        lo,hi=pair; size=hi-lo+1
        path=parts/f"{lo}-{hi}.bin"
        if path.exists() and path.stat().st_size==size:
            return path
        part=path.with_suffix(".partial"); head=path.with_suffix(".headers")
        last=None
        for url in URLS:
            result=subprocess.run(["curl","--silent","--show-error","--location","--fail","--connect-timeout","8","--max-time","420",
                                   "--retry","2","--retry-delay","5","--retry-all-errors","--max-filesize",str(size),
                                   "--range",f"{lo}-{hi}","--dump-header",head,"--output",part,url])
            headers=head.read_text(errors="replace").lower() if head.exists() else ""
            if result.returncode==0 and re.search(r"^http/\S+ 206\b",headers,re.M) and f"content-range: bytes {lo}-{hi}/{TOTAL}" in headers and part.stat().st_size==size:
                break
            last=(url,result.returncode)
            if re.search(r"^http/\S+ 429\b",headers,re.M):
                raise RuntimeError("Official mirror requested rate backoff; verified pieces retained")
        else:
            raise RuntimeError(f"No official mirror verified bytes {lo}-{hi}: {last}")
        part.replace(path); head.unlink(missing_ok=True)
        return path
    with ThreadPoolExecutor(max_workers=jobs) as io:
        for i,_ in enumerate(io.map(fetch,ranges),1):
            if i%12==0 or i==len(ranges): print("verified ranges",i,"/",len(ranges),flush=True)
    if prefix.exists() and prefix.stat().st_size!=prefix_size:
        raise ValueError("Original download changed while range workers ran")
    # A reboot can leave logical holes inside an otherwise correctly sized compressed prefix.
    # Recover only those bytes on a private copy and retain the original until the full publisher MD5 passes.
    gaps=[]
    effective_prefix=prefix
    if prefix_size:
        with prefix.open("rb") as pf:
            at=0
            while at<prefix_size:
                try: start=os.lseek(pf.fileno(),at,os.SEEK_HOLE)
                except OSError as ex:
                    if ex.errno in (errno.EINVAL,errno.ENOTSUP,errno.ENXIO): break
                    raise
                if start>=prefix_size: break
                try: end=os.lseek(pf.fileno(),start,os.SEEK_DATA)
                except OSError as ex:
                    if ex.errno!=errno.ENXIO: raise
                    end=prefix_size
                for left in range(start,end,BLOCK):
                    gaps.append((left,min(left+BLOCK,end)-1))
                at=end
        if gaps:
            effective_prefix=sources/"libri360-prefix.repair"
            subprocess.run(["cp","--reflink=auto","--",prefix,effective_prefix],check=True)
            with effective_prefix.open("r+b") as fixed:
                for left,right in gaps:
                    part=fetch((left,right))
                    fixed.seek(left)
                    with part.open("rb") as reader: shutil.copyfileobj(reader,fixed,1024**2)
                fixed.flush(); os.fsync(fixed.fileno())
            print("Reacquired sparse prefix intervals",gaps,flush=True)
    pending=archive.with_suffix(".assembling")
    md5=hashlib.md5(); sha=hashlib.sha256()
    with pending.open("wb") as out:
        for path in ([effective_prefix] if prefix.exists() else [])+[parts/f"{l}-{h}.bin" for l,h in ranges]:
            with path.open("rb") as reader:
                while chunk:=reader.read(8*1024**2):
                    out.write(chunk); md5.update(chunk); sha.update(chunk)
        out.flush(); os.fsync(out.fileno())
    if pending.stat().st_size!=TOTAL or md5.hexdigest()!=EXPECTED:
        raise ValueError("Full original release checksum failed; original prefix retained")
    pending.replace(archive)
    (sources/(archive.name+".sha256")).write_text(sha.hexdigest()+"  "+archive.name+"\n")
    if gaps:
        proof={"publisher_md5":md5.hexdigest(),"archive_sha256":sha.hexdigest(),"repaired_intervals":gaps,
               "bytes_reacquired":sum(hi-lo+1 for lo,hi in gaps)}
        (sources/"libri360-prefix-repair.json").write_text(json.dumps(proof,indent=2,sort_keys=True)+"\n")
        effective_prefix.unlink()
    for l,h in ranges+gaps: (parts/f"{l}-{h}.bin").unlink()
    parts.rmdir()
    prefix.unlink(missing_ok=True)
    print("Verified release",TOTAL,sha.hexdigest(),flush=True)
    return archive


def index(root,archive,old_rows):
    existing=json.loads(old_rows.read_bytes())
    used={r["id"] for r in existing if r["role"]=="train"}
    held={r["group"].removeprefix("LIBRI-") for r in existing if r["role"]!="train"}
    train_speakers={r["group"].removeprefix("LIBRI-") for r in existing if r["role"]=="train"}
    speakers=set(); accepted_speakers=set(); audio=set(); transcripts={}
    with tarfile.open(archive,mode="r|gz") as tar:
        for member in tar:
            if not member.isfile(): continue
            name=Path(member.name)
            if name.suffix==".flac":
                audio.add(name.stem)
            elif name.name.endswith(".trans.txt"):
                with tar.extractfile(member) as f:
                    for line in f:
                        uid,text=line.decode("utf8").rstrip().split(" ",1)
                        if uid in transcripts: raise ValueError("Duplicate utterance")
                        transcripts[uid]=text
    if audio!=set(transcripts):
        raise ValueError("Audio and transcript inventories differ")
    dst=root/"libri360-eligible.jsonl.gz"; temp=dst.with_suffix(".partial"); overlaps=0; held_rows=0; n=0
    with gzip.open(temp,"wt",encoding="utf8") as out:
        for uid,text in sorted(transcripts.items()):
            sp,chapter,_=uid.split("-")
            speakers.add(sp)
            if uid in used: overlaps+=1; continue
            if sp in held: held_rows+=1; continue
            accepted_speakers.add(sp); n+=1
            path=f"LibriSpeech/train-clean-360/{sp}/{chapter}/{uid}.flac"
            out.write(json.dumps({"id":uid,"speaker":sp,"member":path,"transcript":text},ensure_ascii=False)+"\n")
    temp.replace(dst)
    card={"release":"LibriSpeech train-clean-360","official_md5":EXPECTED,"archive_bytes":TOTAL,
          "paired_utterances":len(transcripts),"speakers":len(speakers),"existing_train_id_overlap":overlaps,
          "removed_for_existing_held_speaker":held_rows,"eligible_utterances":n,"eligible_speakers":len(accepted_speakers),
          "eligible_speakers_absent_from_previous_training":len(accepted_speakers-train_speakers),
          "use":"Staged for future training; excluded from locked G5/G6.","source":"https://openslr.org/12/"}
    (root/"libri360-card.json").write_text(json.dumps(card,indent=2,sort_keys=True)+"\n")
    print(json.dumps(card),flush=True)


if __name__=="__main__":
    q=argparse.ArgumentParser(description=__doc__); q.add_argument("root",type=Path); q.add_argument("--rows",type=Path,required=True)
    q.add_argument("--jobs",type=int,default=6); a=q.parse_args()
    if subprocess.check_output(["findmnt","-nro","TARGET","-T",a.root],text=True).strip()!="/mnt/slwm-data":
        raise SystemExit("Require the mounted Fedora study data volume")
    index(a.root,acquire(a.root,a.jobs),a.rows)
