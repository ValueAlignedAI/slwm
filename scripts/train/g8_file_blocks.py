"""Create bounded, lossless file-API packets for an immutable G8 optimizer generation."""
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import zlib
import numpy as np
from scripts.train.g6_artifacts import G8_TAG,metadata_files,stage_name

CHUNK=16*2**20
PACKETS=Path("/content/g8-packets")


def frame(raw):
    n=len(raw)//4*4
    shuffled=np.frombuffer(raw[:n],np.uint8).reshape(-1,4).T.copy().tobytes()+raw[n:]
    return zlib.compress(shuffled,1)


def unframe(blob,n):
    obj=zlib.decompressobj();raw=obj.decompress(blob,n+1)
    if len(raw)!=n or not obj.eof or obj.unused_data or obj.unconsumed_tail:raise ValueError("Invalid bounded optimizer packet")
    aligned=n//4*4
    return np.frombuffer(raw[:aligned],np.uint8).reshape(4,-1).T.copy().tobytes()+raw[aligned:]


def build(tag,step,expected):
    if not G8_TAG.fullmatch(tag):raise ValueError("Unregistered G8 transport source")
    relative=stage_name(tag,step);stage=Path("/content/g8-out")/relative
    done=json.loads((stage/"complete.json").read_bytes())
    if done["sha256"]["resume.npz"]!=expected or done["run"]!=tag or done["step"]!=step:raise ValueError("Transport source changed")
    dst=PACKETS/relative
    if (dst/"manifest.json").is_file():
        if json.loads((dst/"manifest.json").read_bytes())["header"]["sha"]!=expected:raise ValueError("Conflicting file generation")
        return
    dst.parent.mkdir(exist_ok=True,parents=True)
    with tempfile.TemporaryDirectory(dir=dst.parent,prefix=".packing-") as work:
        folder=Path(work)/"ready";folder.mkdir();chunks=[];digest=hashlib.sha256();pos=0;total=0
        with (stage/"resume.npz").open("rb") as f:
            while raw:=f.read(CHUNK):
                data=frame(raw);leaf=f"{len(chunks):05d}.bin";(folder/leaf).write_bytes(data);digest.update(raw)
                chunks.append({"file":leaf,"offset":pos,"bytes":len(raw),"compressed":len(data),"sha":hashlib.sha256(data).hexdigest(),"raw_sha":hashlib.sha256(raw).hexdigest()})
                pos+=len(raw);total+=len(data)
        if digest.hexdigest()!=expected:raise ValueError("Optimizer hash failed during packetization")
        meta={}
        for k in metadata_files(tag):
            data=(stage/k).read_bytes()
            if len(data)>8*2**20:raise ValueError("Unexpected metadata size")
            meta[k]=base64.b64encode(data).decode()
        doc={"version":1,"header":{"run":tag,"step":step,"sha":expected,"bytes":pos,"metadata":meta},"chunks":chunks,"wire":total}
        (folder/"manifest.json").write_text(json.dumps(doc,separators=(",",":"))+"\n");folder.rename(dst)
    print("G8 lossless packet ready",tag,step,pos,total,flush=True)
