"""Keep immutable study milestones on the Mac while Fedora retains resumes.

Only tokens_*.pt and position_*.pt in the two declared study roots are eligible.
Mutable best/resume weights and fixed endpoints are never removed by this tool.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import time

HOST = "db@192.168.1.200"
REMOTE = "/home/db/dev/slwm"
ROOTS = {"experiments/text/t1/500m-20261106":"tokens_*.pt",
         "experiments/multimodal/t2/general-transfer-20261106":"position_*.pt"}
PROOF = "artifacts/general_world/diverse_v1/checkpoint_archival.json"


def remote(code):
    command = "cd "+shlex.quote(REMOTE)+" && /home/db/venvs/vllm/bin/python -c "+shlex.quote(code)
    return subprocess.check_output(["ssh","-o","BatchMode=yes",HOST,command],text=True)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b""):h.update(chunk)
    return h.hexdigest()


def pass_once(checkout):
    proof_path = checkout/PROOF
    proof = json.loads(proof_path.read_text()) if proof_path.exists() else {"files":{},"method":"SHA256 both hosts before removal; immutable milestones only"}
    code = """from pathlib import Path
import json,time
roots=ROOT_VALUES
files=[]
for root,pattern in roots.items():
 for p in Path(root).rglob(pattern):
  s=p.stat()
  if time.time()-s.st_mtime>30:files.append({'path':str(p),'bytes':s.st_size})
print(json.dumps({'files':files,'finished':Path('experiments/text/t1/500m-20261106/summary.json').exists() and Path('experiments/multimodal/t2/general-transfer-20261106/training_complete.json').exists()}))
""".replace("ROOT_VALUES",repr(ROOTS))
    inventory = json.loads(remote(code));pending = [r for r in inventory["files"] if r["path"] not in proof["files"]]
    if not pending:return inventory["finished"]
    for item in pending:
        p=Path(item["path"])
        assert not p.is_absolute() and ".." not in p.parts
    names=[r["path"] for r in pending]
    hashes=json.loads(remote("import hashlib,json; from pathlib import Path; print(json.dumps({name:hashlib.file_digest(Path(name).open('rb'),'sha256').hexdigest() for name in "+repr(names)+"}))"))
    with tempfile.NamedTemporaryFile(mode="w",prefix="slwm-milestones-",delete=False) as listing:
        listing.write("\n".join(names)+"\n");listing_path=Path(listing.name)
    try:
        subprocess.run(["rsync","-a","--partial","--files-from="+str(listing_path),"-e","ssh -o BatchMode=yes",HOST+":"+REMOTE+"/",str(checkout)+"/"],check=True)
    finally:listing_path.unlink()
    for item in pending:
        p=checkout/item["path"];assert p.stat().st_size==item["bytes"] and sha(p)==hashes[item["path"]]
        proof["files"][item["path"]]={**item,"sha256":hashes[item["path"]],"verified_mac_path":str(p),"remote_transport_removed":True}
    proof["archived_bytes"]=sum(r["bytes"] for r in proof["files"].values())
    proof_path.parent.mkdir(parents=True,exist_ok=True);temporary=proof_path.with_suffix(".json.tmp");temporary.write_text(json.dumps(proof,indent=2)+"\n");temporary.replace(proof_path)
    code="""from pathlib import Path
import hashlib,json
proof=json.loads(PROOF_VALUE)
names=NAMES_VALUE
for name in names:
 p=Path(name);assert p.stat().st_size==proof['files'][name]['bytes'];assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==proof['files'][name]['sha256']
path=Path(PROOF_PATH);temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(proof,indent=2)+'\\n');temporary.replace(path)
for name in names:Path(name).unlink()
print(json.dumps({'archived_milestones':len(names),'total_archived_bytes':proof['archived_bytes']}))
""".replace("PROOF_VALUE",repr(json.dumps(proof))).replace("NAMES_VALUE",repr(names)).replace("PROOF_PATH",repr(PROOF))
    print(remote(code),flush=True)
    return inventory["finished"]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--checkout",type=Path,required=True);parser.add_argument("--watch",action="store_true");args=parser.parse_args()
    deadline=time.monotonic()+48*3600
    while True:
        finished=pass_once(args.checkout)
        if finished or not args.watch or time.monotonic()>deadline:return
        time.sleep(60)


if __name__=="__main__":main()
