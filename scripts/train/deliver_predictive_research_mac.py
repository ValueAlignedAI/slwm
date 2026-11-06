"""Verify G2 inference artifacts locally, archive immutable weights, push findings.

Optimizer/EMA resumes stay on Fedora. Removal requires matching SHA256/size on
both machines and is restricted to completed old parents or G2 milestones.
"""
import argparse,fcntl,hashlib,json,os,shlex,shutil,subprocess,tempfile,time,traceback
from pathlib import Path

HOST='db@192.168.1.200'
REMOTE='/home/db/dev/slwm-g2'
ROOT='experiments/multimodal/t3/predictive-bottleneck-20261106'
BRANCH='training/predictive-bottleneck-20261106'
PROOF='artifacts/predictive_world/g2/delivery.json'


def remote(code):
    return subprocess.check_output(['ssh','-o','BatchMode=yes',HOST,'cd '+shlex.quote(REMOTE)+' && /home/db/venvs/vllm/bin/python -c '+shlex.quote(code)],text=True,timeout=600)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def copy_names(checkout,names,base=REMOTE):
    if not names:return
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
    with tempfile.NamedTemporaryFile(mode='w',prefix='slwm-g2-',delete=False) as stream:
        stream.write('\n'.join(names)+'\n');listing=Path(stream.name)
    try:subprocess.run(['rsync','-aH','--partial','--files-from='+str(listing),'-e','ssh -o BatchMode=yes',HOST+':'+base+'/',str(checkout)+'/'],check=True,timeout=1800)
    finally:listing.unlink()


def archive(checkout,proof):
    names=json.loads(remote("""from pathlib import Path
import json,time
root=Path(ROOT_VALUE)
names=[str(p) for p in root.glob('G2-*/position_*.pt') if time.time()-p.stat().st_mtime>30]
print(json.dumps(names))
""".replace('ROOT_VALUE',repr(ROOT))))
    names=[n for n in names if n not in proof['archived_milestones']]
    if names:
        hashes=json.loads(remote("import pathlib,hashlib,json; print(json.dumps({n:{'bytes':pathlib.Path(n).stat().st_size,'sha256':hashlib.file_digest(pathlib.Path(n).open('rb'),'sha256').hexdigest()} for n in "+repr(names)+"}))"))
        assert shutil.disk_usage(checkout).free>sum(r['bytes'] for r in hashes.values())+8*1024**3,'Keep local checkpoint reserve'
        copy_names(checkout,names)
        for name,item in hashes.items():
            assert (checkout/name).stat().st_size==item['bytes'] and sha(checkout/name)==item['sha256']
            proof['archived_milestones'][name]={**item,'mac_path':str(checkout/name),'verified_unix':time.time()}
        save(checkout/PROOF,proof)
        # Final verify is in the same remote operation as unlink; a failed copy
        # or interrupted verification can never remove the only retained copy.
        remote("""from pathlib import Path
import json,hashlib
items=ITEMS_VALUE
for name,item in items.items():
 p=Path(name);assert p.stat().st_size==item['bytes'];assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==item['sha256']
for name in items:Path(name).unlink()
print(json.dumps({'archived':len(items)}))
""".replace('ITEMS_VALUE',repr(hashes)))
    # These parents are needed by the active500M study until its entire final
    # summary exists. Existing Mac originals must match before unlink.
    if not proof.get('old_text_parents_archived'):
        names=json.loads(remote("""from pathlib import Path
import json
root=Path('/home/db/dev/slwm')
names=[str(p.relative_to(root)) for p in (root/'experiments/text/t1/extended-throughput-20261105').rglob('*.pt')] if (root/'experiments/text/t1/500m-20261106/summary.json').exists() else []
print(json.dumps(names))
"""))
        old=Path('/Users/db/dev/slwm');eligible=[n for n in names if (old/n).exists()]
        if eligible:
            hashes=json.loads(remote("from pathlib import Path; import hashlib,json; root=Path('/home/db/dev/slwm'); print(json.dumps({n:{'bytes':(root/n).stat().st_size,'sha256':hashlib.file_digest((root/n).open('rb'),'sha256').hexdigest()} for n in "+repr(eligible)+"}))"))
            for name,item in hashes.items():assert (old/name).stat().st_size==item['bytes'] and sha(old/name)==item['sha256']
            proof['old_text_parents_archived']={n:{**v,'mac_path':str(old/n)} for n,v in hashes.items()};save(checkout/PROOF,proof)
            remote("""from pathlib import Path
import hashlib
root=Path('/home/db/dev/slwm');assert (root/'experiments/text/t1/500m-20261106/summary.json').exists()
items=ITEMS_VALUE
for name,item in items.items():
 p=root/name;assert p.stat().st_size==item['bytes'] and hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==item['sha256']
for name in items:(root/name).unlink()
""".replace('ITEMS_VALUE',repr(hashes)))


def publish(checkout,paths,message):
    assert subprocess.check_output(['git','branch','--show-current'],cwd=checkout,text=True).strip()==BRANCH
    names=[str(p.relative_to(checkout)) for p in paths if p.is_file()]
    if not names:return
    subprocess.run(['git','add','-f','--',*names],cwd=checkout,check=True)
    if subprocess.run(['git','diff','--cached','--quiet','--',*names],cwd=checkout).returncode:
        subprocess.run(['git','commit','--only','-m',message,'--',*names],cwd=checkout,check=True)
    env=dict(os.environ,GIT_SSH_COMMAND='ssh -o BatchMode=yes -o UserKnownHostsFile='+shlex.quote(str(checkout/'artifacts/predictive_world/g2/github_known_hosts'))+' -o StrictHostKeyChecking=yes')
    subprocess.run(['git','push','origin',BRANCH],cwd=checkout,env=env,check=True,timeout=180)


def final_delivery(checkout,proof):
    destination=checkout/ROOT;destination.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-aH','--partial','--exclude=resume.pt','--exclude=*.tmp','--exclude=*.lock','-e','ssh -o BatchMode=yes',HOST+':'+REMOTE+'/'+ROOT+'/',str(destination)+'/'],check=True,timeout=3600)
    files=json.loads(remote("""from pathlib import Path
import hashlib,json
root=Path(ROOT_VALUE)
files={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for p in root.rglob('*') if p.is_file() and p.name!='resume.pt' and not p.name.endswith(('.tmp','.lock'))}
print(json.dumps(files))
""".replace('ROOT_VALUE',repr(ROOT))))
    for name,item in files.items():assert (destination/name).stat().st_size==item['bytes'] and sha(destination/name)==item['sha256'],name
    proof['final_verification']={'remote_root':REMOTE+'/'+ROOT,'mac_root':str(destination),'files':files,'verified_unix':time.time(),'resume_policy':'Optimizer/EMA checkpoints retained on Fedora; every inference/result byte verified on Mac'}
    # Small dataset contracts plus the independently withheld sensor source.
    relative='artifacts/predictive_world/g2';data=checkout/relative;data.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-aH','--exclude=acquisition*source/','--exclude=*.log','--exclude=sources/','-e','ssh -o BatchMode=yes',HOST+':'+REMOTE+'/'+relative+'/',str(data)+'/'],check=True,timeout=600)
    metadata='artifacts/general_world/diverse_v1/droid_g2';target=checkout/metadata;target.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-a','--include=*/','--include=*.json','--exclude=*','-e','ssh -o BatchMode=yes',HOST+':'+REMOTE+'/'+metadata+'/',str(target)+'/'],check=True,timeout=600)
    save(checkout/PROOF,proof)
    card={k:v for k,v in proof['final_verification'].items() if k!='files'};card.update(verified_files=len(files),verified_bytes=sum(v['bytes'] for v in files.values()),archived_milestones=len(proof['archived_milestones']))
    save(destination/'delivery_summary.json',card)
    selected=[destination/n for n in ('REPORT.md','findings.json','gates.json','study_complete.json','evaluation_lock.json','source_hashes.json','environment.json','delivery_summary.json','readouts/evaluation_summary.json','consequences/summary.json','alignment/summary.json','speech_retention/summary.json','operational/summary.json','exports/actual_checkpoint_inference.json')]
    selected += list(destination.glob('G2-*/complete.json'))+list(destination.glob('G2-*/curves.json'))+list(destination.glob('operational/G2-*/summary.json'))
    publish(checkout,selected,'research: publish frozen predictive-state transfer and dynamics findings')
    proof['delivered_and_published']=True;save(checkout/PROOF,proof)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkout',type=Path,required=True);a=parser.parse_args();checkout=a.checkout.resolve();base=checkout/'artifacts/predictive_world/g2';base.mkdir(parents=True,exist_ok=True)
    lock=(base/'delivery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    proof=json.loads((checkout/PROOF).read_text()) if (checkout/PROOF).exists() else {'archived_milestones':{}}
    if proof.get('delivered_and_published'):return
    while True:
        try:
            archive(checkout,proof)
            ready=json.loads(remote("from pathlib import Path; import json; r=Path("+repr(ROOT)+"); print(json.dumps({'screen':(r/'screen_complete.json').exists(),'complete':(r/'study_complete.json').exists(),'status':json.loads((r/'pipeline_status.json').read_text()) if (r/'pipeline_status.json').exists() else {}}))"))
            save(base/'remote_status.json',ready)
            if ready['screen'] and not proof.get('screen_published'):
                names=[ROOT+'/'+n for n in ('screen_complete.json','source_hashes.json','environment.json','real_data_gradient_preflight.json')]
                copy_names(checkout,names);publish(checkout,[checkout/n for n in names],'research: record frozen large-model predictive-state training screen');proof['screen_published']=True;save(checkout/PROOF,proof)
            if ready['complete'] and ready['status'].get('phase')=='complete':final_delivery(checkout,proof);return
            if ready['status'].get('phase')=='failed':save(base/'needs_attention.json',ready['status'])
        except Exception as exc:
            save(base/'delivery_retry.json',{'unix':time.time(),'error':repr(exc),'traceback':traceback.format_exc()});print(traceback.format_exc(),flush=True)
        time.sleep(60)


if __name__=='__main__':main()
