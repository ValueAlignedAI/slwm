"""Verify finished study artifacts on the Mac and publish only curated results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

HOST="db@192.168.1.200"
REMOTE="/home/db/dev/slwm"
BRANCH="training/general-transfer-20261106"
STUDIES={"text":("experiments/text/t1/500m-20261106","report_complete.json"),"world":("experiments/multimodal/t2/general-transfer-20261106","study_complete.json")}


def remote(code):
    return subprocess.check_output(["ssh","-o","BatchMode=yes",HOST,"cd "+shlex.quote(REMOTE)+" && /home/db/venvs/vllm/bin/python -c "+shlex.quote(code)],text=True)


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b""):h.update(chunk)
    return h.hexdigest()


def mirror(checkout,relative):
    destination=checkout/relative;destination.mkdir(parents=True,exist_ok=True)
    subprocess.run(["rsync","-a","--partial","--exclude=resume.pt","-e","ssh -o BatchMode=yes",HOST+":"+REMOTE+"/"+relative+"/",str(destination)+"/"],check=True)
    return destination


def prepared_delivery(checkout):
    """Copy reproducible caches after both trained studies, keeping8GiB free."""
    parent=checkout/"artifacts/general_world/diverse_v1";parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(["rsync","-a","-e","ssh -o BatchMode=yes","--include=*/","--include=*.json","--include=*.csv","--include=*.txt","--exclude=*",HOST+":"+REMOTE+"/artifacts/general_world/diverse_v1/",str(parent)+"/"],check=True)
    (parent/"prepared").mkdir(exist_ok=True)
    subprocess.run(["scp","-q",HOST+":"+REMOTE+"/artifacts/general_world/diverse_v1/prepared/normalizers.npz",str(parent/"prepared/normalizers.npz")],check=True)
    contract=json.loads((parent/"prepared/dataset_contract.json").read_text());assert sha(parent/"prepared/normalizers.npz")==contract["normalizers_sha256"]
    results={}
    for family in ("coco","clotho","speech","ami","pamap2","mhealth","droid_expanded"):
        relative="artifacts/general_world/diverse_v1/prepared/"+family
        needed=int(remote("from pathlib import Path; print(sum(p.stat().st_size for p in Path("+repr(relative)+").rglob('*') if p.is_file()))").strip())
        free=shutil.disk_usage(checkout).free
        reserve_gib=int(os.environ.get("SLWM_PREPARED_DATA_RESERVE_GIB","8"))
        if free-needed<reserve_gib*1024**3:
            results[family]={"copied":False,"remote_root":REMOTE+"/"+relative,"bytes":needed,"reason":"Trained artifacts take priority; cache retained on Fedora, source/hash metadata preserved locally; configured reserve protects ongoing studies"}
            continue
        mirror(checkout,relative);proof=verify(checkout,relative)
        results[family]={"copied":True,**{k:v for k,v in proof.items() if k!="files"}}
    mirror(checkout,"artifacts/decoders/vocos-mel-24khz")
    (parent/"prepared_delivery.json").write_text(json.dumps({"families":results,"trained_studies_delivered_first":True},indent=2)+"\n")
    return results


def verify(checkout,relative):
    code="""from pathlib import Path
import hashlib,json
root=Path(ROOT_VALUE)
files={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('delivery_verification.json','resume.pt') and not p.name.endswith('.tmp')}
print(json.dumps(files))
""".replace("ROOT_VALUE",repr(relative))
    files=json.loads(remote(code))
    for name,item in files.items():
        path=checkout/relative/name
        assert path.stat().st_size==item["bytes"] and sha(path)==item["sha256"],str(path)
    result={"remote_root":REMOTE+"/"+relative,"mac_root":str(checkout/relative),"verified_files":len(files),"verified_bytes":sum(v["bytes"] for v in files.values()),"files":files,"scope":"Inference checkpoints/results verified locally; resumable optimizer/EMA states remain on Fedora and are identified by complete.json; separately archived milestones/media have byte proofs"}
    (checkout/relative/"delivery_verification.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def publish(checkout,kind,relative,verification):
    current=subprocess.check_output(["git","branch","--show-current"],cwd=checkout,text=True).strip()
    assert current==BRANCH,"Research branch changed; retain artifacts and review publication manually"
    root=checkout/relative
    if kind=="text":
        names=["REPORT.md","findings.json","summary.json","selection.json","source_hashes.json","report_complete.json","scaling.png"]
        paths=[root/name for name in names]
        for folder in sorted(root.glob("EXP-*")):
            paths.extend(folder/name for name in ("complete.json","test.json","samples.json","curves.json","config.json"))
    else:
        paths=[root/name for name in ("REPORT.md","findings.json","gates.json","study_complete.json","evaluation_lock.json","evaluation_source_hashes.json","source_hashes.json")]
        paths += [root/"speech/summary.json",root/"speech/calibration.json",root/"alignment/summary.json",root/"readouts/evaluation_summary.json"]
        paths += [root/"selection_geometry/diagnostic.json",root/"angular_pilot_v2/summary.json"]
        for folder in sorted(root.glob("G1-*")):paths.extend(folder/name for name in ("complete.json","curves.json"))
        paths.extend(root.glob("operational/G1-*/summary.json"));paths.extend(root.glob("consequences/G1-*/summary.json"))
    card=root/"delivery_summary.json";card.write_text(json.dumps({k:v for k,v in verification.items() if k!="files"},indent=2)+"\n");paths.append(card)
    selected=[str(p.relative_to(checkout)) for p in paths if p.exists()]
    subprocess.run(["git","add","-f","--",*selected],cwd=checkout,check=True)
    # --only excludes any unrelated staged user change from these commits.
    changed=subprocess.run(["git","diff","--cached","--quiet","--",*selected],cwd=checkout).returncode!=0
    if changed:subprocess.run(["git","commit","--only","-m","research: publish verified "+kind+" study outcomes","--",*selected],cwd=checkout,check=True)
    env=dict(os.environ,GIT_SSH_COMMAND="ssh -o BatchMode=yes -o UserKnownHostsFile=/private/tmp/slwm_github_known_hosts -o StrictHostKeyChecking=yes")
    subprocess.run(["git","push","origin",BRANCH],cwd=checkout,env=env,check=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--checkout",type=Path,required=True);args=parser.parse_args();checkout=args.checkout
    state_path=checkout/"artifacts/general_world/diverse_v1/research_delivery.json";state=json.loads(state_path.read_text()) if state_path.exists() else {"delivered":{}}
    deadline=time.monotonic()+48*3600
    while time.monotonic()<deadline:
        status=json.loads(remote("from pathlib import Path; import json; print(json.dumps({name:Path(root,marker).exists() for name,(root,marker) in "+repr(STUDIES)+".items()}))"))
        for kind,ready in status.items():
            if not ready or kind in state["delivered"]:continue
            relative,_=STUDIES[kind];mirror(checkout,relative);verification=verify(checkout,relative)
            publish(checkout,kind,relative,verification);state["delivered"][kind]={k:v for k,v in verification.items() if k!="files"};state_path.write_text(json.dumps(state,indent=2)+"\n")
            print(json.dumps({"verified_and_published":kind,"files":verification["verified_files"],"bytes":verification["verified_bytes"]}),flush=True)
        if len(state["delivered"])==len(STUDIES):
            if not state.get("prepared_delivery"):
                state["prepared_delivery"]=prepared_delivery(checkout);state_path.write_text(json.dumps(state,indent=2)+"\n")
            return
        time.sleep(60)


if __name__=="__main__":main()
