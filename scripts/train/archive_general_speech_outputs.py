"""Mac archival of fully scored, immutable G1 speech outputs; metadata stays GPU-side."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

HOST="db@192.168.1.200"
REMOTE="/home/db/dev/slwm"
BASE="experiments/multimodal/t2/general-transfer-20261106/speech/evaluation"


def remote(code):
    command="cd "+shlex.quote(REMOTE)+" && /home/db/venvs/vllm/bin/python -c "+shlex.quote(code)
    return subprocess.check_output(["ssh","-o","BatchMode=yes",HOST,command],text=True)


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(4*1024**2),b""):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--checkout",type=Path,required=True);args=parser.parse_args();deadline=time.monotonic()+48*3600
    proof_path=args.checkout/"artifacts/general_world/diverse_v1/speech_media_archival.json"
    proof=json.loads(proof_path.read_text()) if proof_path.exists() else {"runs":{}}
    while time.monotonic()<deadline:
        status=json.loads(remote("from pathlib import Path; import json; root=Path("+repr(BASE)+"); print(json.dumps({'complete':[p.parent.name for p in root.glob('G1-*/complete.json')],'finished':Path('experiments/multimodal/t2/general-transfer-20261106/speech/summary.json').exists()}))"))
        for name in status["complete"]:
            if name in proof["runs"]:continue
            assert name.startswith("G1-") and name[3:].isdigit();relative=BASE+"/"+name
            inventory=json.loads(remote("from pathlib import Path; import json,hashlib; root=Path("+repr(relative)+"); print(json.dumps([{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for p in root.rglob('*.flac')]))"))
            destination=args.checkout/relative;destination.mkdir(parents=True,exist_ok=True)
            subprocess.run(["rsync","-a","--partial","-e","ssh -o BatchMode=yes",HOST+":"+REMOTE+"/"+relative+"/",str(destination)+"/"],check=True)
            for item in inventory:
                path=destination/item["path"];assert path.stat().st_size==item["bytes"] and sha(path)==item["sha256"]
            record={"verified_mac_path":str(destination),"files":inventory,"bytes":sum(r["bytes"] for r in inventory),"complete_marker_verified":True}
            code="""from pathlib import Path
import json,hashlib
root=Path(ROOT_VALUE);record=json.loads(RECORD_VALUE)
assert (root/'complete.json').exists()
for item in record['files']:
 p=root/item['path'];assert p.stat().st_size==item['bytes'];assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==item['sha256']
(root/'media_archived.json').write_text(json.dumps(record,indent=2)+'\\n')
for item in record['files']:(root/item['path']).unlink()
""".replace("ROOT_VALUE",repr(relative)).replace("RECORD_VALUE",repr(json.dumps(record)))
            remote(code);proof["runs"][name]=record;proof_path.parent.mkdir(parents=True,exist_ok=True);proof_path.write_text(json.dumps(proof,indent=2)+"\n")
            print(json.dumps({"archived_scored_speech_run":name,"bytes":record["bytes"]}),flush=True)
        if status["finished"] and set(status["complete"])<=set(proof["runs"]):return
        time.sleep(60)


if __name__=="__main__":main()
