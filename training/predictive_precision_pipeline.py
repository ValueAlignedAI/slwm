"""Resume frozen G2 evaluation with an explicit physical-cache precision erratum."""
import argparse,fcntl,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from training.general_sources import write
from training.research_sources import file_sha

CONFIG='configs/t3/predictive_world.json'


def prepare(cfg):
    from training.predictive_train import source_guard as core_guard
    from training.predictive_readouts_precision_v2 import source_guard,cache_split
    root=Path(cfg['root']);readouts=root/'readouts';core_guard(cfg);sources=source_guard(readouts)
    marker=readouts/'precision_erratum.json'
    if marker.exists():
        value=json.loads(marker.read_text());assert value['sources']==sources
        for item in value['caches']:assert file_sha(item['path'])==item['new_sha256']
        return
    assert not (root/'evaluation_lock.json').exists(),'Register correction before final observations'
    out=readouts/'precision_erratum_20261109';out.mkdir(exist_ok=True);temporary=out/'reextracted';records=[]
    recipe=json.loads(Path('configs/t3/predictive_readouts.json').read_text());run=next(r for r in cfg['runs'] if r['id']==recipe['shared_raw_text_edge'])
    for family in ('har','mhealth'):
        for role in ('train','selection'):
            folder=readouts/'features'/family/run['id']/role;p=folder/'invariant.npy';old=np.load(p,mmap_mode='r');old_nonfinite=int((~np.isfinite(old)).sum());old_sha=file_sha(p)
            backup=out/'before'/family/role;backup.mkdir(parents=True,exist_ok=True)
            for name in ('invariant.npy','complete.json'):shutil.copyfile(folder/name,backup/name)
            cache_split(cfg,recipe,temporary,run,family,role)
            fresh=temporary/'features'/family/run['id']/role
            assert file_sha(folder/'cases.json')==file_sha(fresh/'cases.json'),'Case identities changed'
            for target in folder.glob('target_*.npy'):assert file_sha(target)==file_sha(fresh/target.name),'Readout labels changed'
            values=np.load(fresh/'invariant.npy',mmap_mode='r');assert values.dtype==np.float32 and np.isfinite(values).all()
            assert values.shape==old.shape;del old
            staging=folder/'invariant.fp32.tmp';shutil.copyfile(fresh/'invariant.npy',staging);staging.replace(p)
            complete=json.loads((folder/'complete.json').read_text());complete['files']['invariant.npy']=file_sha(p);complete['physical_invariant_storage']='float32; precision erratum before final lock';write(folder/'complete.json',complete)
            records.append({'path':str(p),'old_sha256':old_sha,'old_nonfinite':old_nonfinite,'new_sha256':file_sha(p),'new_dtype':'float32','new_nonfinite':0,'feature_formula_changed':False,'examples_and_labels_unchanged':True})
    retired=[]
    for family in ('har','mhealth'):
        for p in (readouts/'heads'/family).glob('*/physical-invariants'):
            destination=out/'retired_heads'/family/p.parent.name/'physical-invariants';destination.parent.mkdir(parents=True,exist_ok=True)
            retired += [str(x) for x in p.rglob('complete.json')];shutil.move(str(p),str(destination))
    write(marker,{'sources':sources,'caches':records,'retired_completed_heads':retired,'trained_core_weights_modified':False,'final_lock_existed':False,'scope':'FP32 physical-control cache storage; unchanged trained cores, examples, labels, formulas, learned caches, head capacity, optimizer recipe and metrics','registered_unix':time.time()})
    print(json.dumps({'cache_precision_corrected':len(records),'refit_completed_invariant_heads':len(retired)}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');args=p.parse_args();cfg=json.loads(Path(CONFIG).read_text())
    if args.prepare:prepare(cfg);return
    root=Path(cfg['root']);logs=root/'pipeline_logs';lease=(root/'pipeline.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    phases_path=root/'pipeline_phases.json';phases=json.loads(phases_path.read_text());env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python',PYTHONUNBUFFERED='1')
    def state(phase,**extra):write(root/'pipeline_status.json',{'pid':os.getpid(),'phase':phase,'unix':time.time(),'configuration_sha256':file_sha(CONFIG),'evaluation_revision':'physical-cache-FP32-v2',**extra})
    def phase(name,module,arguments):
        if name in phases:return
        assert shutil.disk_usage('.').free>4*1024**3,'Retain device storage headroom'
        for attempt in range(1,4):
            state(name,attempt=attempt)
            with (logs/(name+'_precision_v2.log')).open('a') as stream:code=subprocess.run([sys.executable,'-m',module,*arguments],env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
            if code==0:break
            if attempt==3:raise RuntimeError(name+' failed; retained original and precision-v2 logs')
            time.sleep(20)
        phases[name]={'module':module,'args':list(arguments),'completed_unix':time.time(),'evaluation_revision':'physical-cache-FP32-v2','log_sha256':file_sha(logs/(name+'_precision_v2.log'))};write(phases_path,phases)
    try:
        assert (root/'training_complete.json').exists()
        for name,module,arguments in [
            ('precision_cache_repair','training.predictive_precision_pipeline',('--prepare',)),
            ('fit_frozen_readouts','training.predictive_readouts_precision_v2',('--stage','fit')),
            ('final_lock','training.predictive_eval_precision_v2',('--stage','freeze')),
            ('export_and_test_artifacts','training.predictive_report',('--stage','export')),
            ('unseen_task_evaluation','training.predictive_eval_precision_v2',('--stage','readouts')),
            ('action_consequences','training.predictive_dynamics_eval',()),
            ('independent_alignment','training.predictive_alignment',()),
            ('fresh_speaker_retention','training.predictive_speech_retention',()),
            ('ongoing_replay','training.predictive_live',()),
            ('report','training.predictive_report',('--stage','report'))]:phase(name,module,arguments)
        state('complete',all_registered_phases_attempted=True,precision_erratum_sha256=file_sha(root/'readouts/precision_erratum.json'))
    except BaseException as exc:state('failed',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
