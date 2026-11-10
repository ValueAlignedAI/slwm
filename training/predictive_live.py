"""Over40s of paced stateful inference, same-head reset controls and stale rejection."""
import fcntl,gc,json,time
from pathlib import Path
import numpy as np
import tiktoken,torch

from exploration.predictive_infer import LiveWorld,load_world,load_actor,actor_proposal,weight_hash
from training.general_batch import tensor_tree
from training.general_data import CACHE,order
from training.general_sources import write
from training.predictive_data import PREPARED
from training.predictive_eval import register
from training.predictive_features import supplied_text
from training.predictive_readout_data import ReadoutData
from training.research_sources import file_sha


def main():
    torch.set_num_threads(4);cfg=json.loads(Path('configs/t3/predictive_world.json').read_text());root=Path(cfg['root']);register();assert (root/'evaluation_lock.json').exists()
    data=ReadoutData(cfg,'droid','test',True);candidates=[]
    for row in data.rows.values():
        if not row['physical_timestamps'] or row['steps']<64:continue
        clock=data.load(str(CACHE/cfg['robot_family']/row['id']/'clock.npy'))
        if clock[-9]-clock[31]>=41 and np.diff(clock).max()<=.25:candidates.append(row)
    out=root/'operational';out.mkdir(exist_ok=True)
    if not candidates:
        write(out/'summary.json',{'completed':False,'reason':'No frozen final robot episode covers40 native seconds after context warmup','no_physical_action_sent':True});return
    registered=json.loads((PREPARED/'evaluation_cases.json').read_text()).get('live_robot_id')
    row=next(r for r in candidates if r['id']==registered);clock=np.asarray(data.load(str(CACHE/cfg['robot_family']/row['id']/'clock.npy')));ticks=np.arange(411)*.1
    # Latest available packet, never the first future frame after a due time.
    indices=(np.searchsorted(clock,clock[31]+ticks,side='right')-1).clip(31,len(clock)-9);encoder=tiktoken.get_encoding('gpt2');results={}
    for run in cfg['runs'][:4]:
        folder=out/run['id'];folder.mkdir(exist_ok=True)
        if (folder/'summary.json').exists():results[run['id']]=json.loads((folder/'summary.json').read_text());continue
        with open('/tmp/slwm_research_gpu.lock','a') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX);model,_=load_world(root/run['id']/'endpoint.pt');heads,policy=load_actor(root/'exports'/run['id']/'actor.pt');assert policy['core_endpoint_sha256']==file_sha(root/run['id']/'endpoint.pt');loop=LiveWorld(model,heads);before=weight_hash(model);events=[];latency=[];start=time.perf_counter()
            for tick,index in enumerate(indices):
                due=start+ticks[tick];remaining=due-time.perf_counter()
                if remaining>0:time.sleep(remaining)
                # Case is from a preselected identity/length rule, never output quality.
                pipeline_start=time.perf_counter();case={'id':row['id'],'start':int(index)-31,'group':row['group'],'role':'test'};batch=data._robot([case],'provided')
                local=clock[index-31:index+9]-clock[index-31];batch['obs']['time']=local[:32][None].astype(np.float32);batch['obs']['time_unit']=np.zeros(1,np.int64);batch['obs'].update(supplied_text(batch['captions'],encoder));obs=tensor_tree(batch['obs'],'cuda')
                if tick and tick%97==0:
                    previous=loop.memory.clone();rejected=loop.observe(obs,loop.last_packet-.001)
                    assert rejected['status']=='stale_rejected' and torch.equal(previous,loop.memory)
                    events.append({'tick':tick,'stale_rejected':True})
                torch.cuda.synchronize();began=time.perf_counter();current=loop.observe(obs,float(clock[index]));torch.cuda.synchronize();compute_seconds=time.perf_counter()-began;seconds=time.perf_counter()-pipeline_start;latency.append(seconds);lateness=max(0,time.perf_counter()-due-.1)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):reset=model(obs)
                reset_command=actor_proposal(reset,heads);target=batch['target']
                events.append({'tick':tick,'source_seconds':float(clock[index]),'latency_seconds':seconds,'model_policy_seconds':compute_seconds,'deadline_lateness_seconds':lateness,'status':current['status'],'command':None if 'command' not in current else current['command'].tolist(),'reset_command':reset_command.tolist(),'target_command':target.tolist(),'persistent_command_absolute_error':None if 'command' not in current else float(np.abs(current['command']-target).mean()),'reset_command_absolute_error':float(np.abs(reset_command-target).mean())})
            after=weight_hash(model);assert before==after
            summary={'run':run['id'],'episode':row['id'],'group':row['group'],'ticks':len(indices),'replay_seconds':float(ticks[-1]),'source_span_seconds':float(clock[indices[-1]]-clock[indices[0]]),'consumer_hz':10,'native_median_hz':float(1/np.median(np.diff(clock))),'latency_p50_seconds':float(np.median(latency)),'latency_p95_seconds':float(np.quantile(latency,.95)),'stale_packets_rejected':sum(e.get('stale_rejected',False) for e in events),'weights_sha256_before':before,'weights_sha256_after':after,'same_weights':True,'memory_updates':loop.updates,'scope':'Persistent frozen state plus state-only actor; reset control is extra diagnostic computation. Recorded observations do not respond to proposals; no physical feedback/actuation.'}
            write(folder/'events.json',events);write(folder/'summary.json',summary);results[run['id']]=summary
            del model,heads,loop;gc.collect();torch.cuda.empty_cache()
    write(out/'summary.json',{'completed':True,'runs':results,'physical_actions_sent':0})


if __name__=='__main__':main()
