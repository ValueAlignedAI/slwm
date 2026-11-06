"""Resumable G2 training -> frozen evaluation, serialized with older GPU work."""
import fcntl,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from training.general_sources import write
from training.research_sources import file_sha

CONFIG='configs/t3/predictive_world.json'


def main():
    cfg=json.loads(Path(CONFIG).read_text());root=Path(cfg['root']);root.mkdir(parents=True,exist_ok=True);logs=root/'pipeline_logs';logs.mkdir(exist_ok=True)
    lease=(root/'pipeline.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=root/'pipeline_status.json';done=root/'pipeline_phases.json';phases=json.loads(done.read_text()) if done.exists() else {}
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python',PYTHONUNBUFFERED='1')
    def state(phase,**extra):write(status,{'pid':os.getpid(),'phase':phase,'unix':time.time(),'configuration_sha256':file_sha(CONFIG),**extra})
    def phase(name,module,args=()):
        if name in phases:return
        assert shutil.disk_usage('.').free>6*1024**3,'Preserve disk headroom; archive only verified immutable artifacts'
        for attempt in range(1,4):
            state(name,attempt=attempt)
            with (logs/(name+'.log')).open('a') as stream:
                result=subprocess.run([sys.executable,'-m',module,*args],stdout=stream,stderr=subprocess.STDOUT,env=env)
            if result.returncode==0:break
            if attempt==3:raise RuntimeError(f'{name} failed after three identical-source attempts; see {logs/name}')
            time.sleep(30)
        phases[name]={'completed_unix':time.time(),'module':module,'args':list(args),'log_sha256':file_sha(logs/(name+'.log'))};write(done,phases)
    try:
        deadline=time.time()+12*3600
        while not (root/'long_context_preflight/complete.json').exists():
            state('waiting_long_context_hardware_check');assert time.time()<deadline,'Long-context preflight incomplete';time.sleep(30)
        acquisition=Path('artifacts/general_world/diverse_v1/droid_g2/acquisition_complete.json')
        if not acquisition.exists():phase('robot_data','training.predictive_robot')
        phase('decoder_audit','training.predictive_decoder_audit')
        phase('verified_native_clocks','training.predictive_reclock')
        phase('freeze_data','training.predictive_data')
        phase('real_data_gradients','training.predictive_train',('--stage','smoke'))
        phase('primary_screen','training.predictive_train',('--stage','screen'))
        # The screen shares bounded GPU leases. Give the two earlier committed
        # studies priority before the full12-run scale matrix enters the queue.
        older=Path('/home/db/dev/slwm');deadline=time.time()+72*3600
        while not all((older/p).exists() for p in ('experiments/text/t1/500m-20261106/summary.json','experiments/multimodal/t2/general-transfer-20261106/training_complete.json')):
            state('waiting_previous_core_weights_to_freeze',test_scores_read=False);assert time.time()<deadline,'Earlier core queue exceeded72h';time.sleep(60)
        for name,module,args in [
            ('scale_training','training.predictive_train',('--stage','train')),
            ('readout_features','training.predictive_readouts',('--stage','features')),
            ('fit_frozen_readouts','training.predictive_readouts',('--stage','fit')),
            ('final_lock','training.predictive_eval',('--stage','freeze')),
            ('export_and_test_artifacts','training.predictive_report',('--stage','export')),
            ('unseen_task_evaluation','training.predictive_eval',('--stage','readouts')),
            ('action_consequences','training.predictive_dynamics_eval',()),
            ('independent_alignment','training.predictive_alignment',()),
            ('fresh_speaker_retention','training.predictive_speech_retention',()),
            ('ongoing_replay','training.predictive_live',()),
            ('report','training.predictive_report',('--stage','report'))]:phase(name,module,args)
        state('complete',all_registered_phases_attempted=True)
    except BaseException as exc:
        state('failed',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
