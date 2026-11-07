"""Recover existing immutable studies sequentially after CUDA-memory failures."""
import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path

OLD=Path('/home/db/dev/slwm');NEW=Path('/home/db/dev/slwm-g2-final');BASE=OLD/'artifacts/predictive_world/g2/recovery_20261107';BASE.mkdir(parents=True,exist_ok=True)
ENV=dict(os.environ,PYTHONPATH='/home/db/dev/slwm-g2/.g2-deps:/home/db/dev/slwm/.research-venv/lib/python3.12/site-packages',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python')


def write(name,value):
 p=BASE/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)


def state(phase,**extra):write('status.json',{'pid':os.getpid(),'phase':phase,'unix':time.time(),'science_sources_modified':False,**extra})


def idle():
 deadline=time.time()+600
 while True:
  used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
  if used<1024:return
  assert time.time()<deadline,'Another GPU process is active; keep recovery serialized'
  time.sleep(10)


def phase(name,args,cwd):
 done=BASE/(name+'.json')
 if done.exists():return
 for attempt in range(1,4):
  idle();state(name,attempt=attempt);began=time.time()
  with (BASE/(name+'.log')).open('a') as stream:
   env=dict(ENV)
   if cwd==OLD:
    env.pop('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION',None);env['PYTHONPATH']='/home/db/dev/slwm/.research-venv/lib/python3.12/site-packages'
   p=subprocess.Popen(args,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT);write(name+'_pid.json',{'pid':p.pid,'args':args,'cwd':str(cwd),'started_unix':began});code=p.wait()
  # A failed child can release its file lock before its CUDA context finishes
  # teardown. Wait for the process and device cleanup before another heavy job.
  idle()
  if code==0:write(name+'.json',{'returncode':0,'wall_seconds':time.time()-began,'attempt':attempt,'completed_unix':time.time()});return
  if attempt==3:raise RuntimeError(name+' failed; logs and all existing states retained')
  time.sleep(20)


def main():
 lease=(BASE/'supervisor.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  export=NEW/'experiments/multimodal/t3/predictive-bottleneck-20261106/interim_inference_20261107/export_complete.json';deadline=time.time()+1800
  while not (export.exists() and (OLD/'experiments/text/t1/500m-20261106/interim_inference_20261107/manifest.json').exists()):
   state('waiting_inference_exports');assert time.time()<deadline,'Inference export did not complete';time.sleep(10)
  # All original configs, source guards, optimizers, RNG and example positions
  # stay unchanged. Only process scheduling and failure detection are new.
  phase('text_train',[sys.executable,'-m','training.text_500m','--stage','train'],OLD)
  phase('text_evaluate',[sys.executable,'-m','training.text_500m','--stage','evaluate'],OLD)
  phase('general_core_train',[sys.executable,'-m','training.general_train','--stage','train'],OLD)
  # The existing G2 supervisor wakes once these older cores freeze. Keep the
  # stopped G1 evaluator out of its GPU slot until G2's registered suite finishes.
  root=NEW/'experiments/multimodal/t3/predictive-bottleneck-20261106';deadline=time.time()+96*3600;restarts=0
  while not (root/'study_complete.json').exists():
   current=json.loads((root/'pipeline_status.json').read_text());state('g2_registered_pipeline',g2=current)
   pid=current['pid'];p=Path('/proc')/str(pid);alive=p.exists() and p.joinpath('stat').read_text().split()[2]!='Z'
   if current['phase']=='failed' or not alive:
    assert restarts<2,'G2 stopped repeatedly; preserve source recipe and require investigation'
    phase('g2_restart_'+str(restarts),[sys.executable,'-m','training.predictive_pipeline'],NEW);restarts+=1
   assert time.time()<deadline,'G2 exceeded96h recovery wait';time.sleep(30)
  phase('general_final_evaluation',['bash','scripts/train/run_general_evaluation_fedora.sh'],OLD)
  state('complete')
 except BaseException as exc:state('failed',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
