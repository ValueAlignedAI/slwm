"""Mac archival of redundant G1 transports, with byte checks before removal.

Prepared observations and all metadata stay on Fedora. Only the three explicitly
listed task-owned transport directories can be archived by this utility.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HOST="db@192.168.1.200"
REMOTE="/home/db/dev/slwm"
ROOT="artifacts/general_world/diverse_v1"


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024**2),b""):h.update(chunk)
    return h.hexdigest()


def remote(code):
    return subprocess.check_output(["ssh","-o","BatchMode=yes",HOST,"cd "+shlex.quote(REMOTE)+" && /home/db/venvs/vllm/bin/python -c "+shlex.quote(code)],text=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--checkout",type=Path,required=True);parser.add_argument("--family",choices=("clotho","droid","droid_expanded"),required=True);args=parser.parse_args()
    relative=ROOT+"/"+args.family+"/sources";destination=args.checkout/relative;destination.mkdir(parents=True,exist_ok=True)
    inventory=json.loads(remote("from pathlib import Path; import hashlib,json; p=Path("+repr(relative)+"); print(json.dumps([{\"name\":f.name,\"bytes\":f.stat().st_size,\"sha256\":hashlib.file_digest(f.open('rb'),'sha256').hexdigest()} for f in sorted(p.iterdir()) if f.is_file() and not f.name.endswith('.json')]))"))
    subprocess.run(["rsync","-a","--partial","-e","ssh -o BatchMode=yes",HOST+":"+REMOTE+"/"+relative+"/",str(destination)+"/"],check=True)
    for item in inventory:
        p=destination/item["name"];assert p.stat().st_size==item["bytes"] and sha(p)==item["sha256"],str(p)
    proof={"family":args.family,"verified_mac_root":str(destination),"files":inventory,"bytes":sum(i["bytes"] for i in inventory),"prepared_observations_retained":True}
    local=args.checkout/ROOT/(args.family+"_transport_archival.json");local.write_text(json.dumps(proof,indent=2)+"\n")
    # Recheck every remote byte immediately before deleting only those exact
    # files. The other copy and immutable source/verification cards remain.
    code="""from pathlib import Path
import json,hashlib
proof=json.loads(PAYLOAD)
root=Path(DIRECTORY)
for item in proof['files']:
 p=root/item['name'];assert p.parent==root and p.stat().st_size==item['bytes'];assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==item['sha256']
Path(CARD).write_text(json.dumps(proof,indent=2)+'\\n')
for item in proof['files']:(root/item['name']).unlink()
print(json.dumps({'archived_bytes':proof['bytes'],'family':proof['family']}))
""".replace("PAYLOAD",repr(json.dumps(proof))).replace("DIRECTORY",repr(relative)).replace("CARD",repr(ROOT+"/"+args.family+"_transport_archival.json"))
    print(remote(code),flush=True)


if __name__=="__main__":main()
