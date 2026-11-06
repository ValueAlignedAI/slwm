"""Frozen G2 consequences, moving strata and prefix controls; no fitting."""
import fcntl,gc,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import tiktoken,torch

from training.general_batch import tensor_tree
from training.general_consequences import state_errors,physical_state,images16
from training.general_data import CACHE,pose_delta
from training.general_sources import write
from training.predictive_data import PREPARED
from training.predictive_eval import register
from training.predictive_features import supplied_text
from training.predictive_readout_data import ReadoutData
from training.predictive_train import initialize
from training.research_sources import file_sha

HORIZONS=[1,4,8]
TAGS={'id','group','condition','kind','moving','unconditional','physical_clock'}


def controls(batch):
    last=np.concatenate([batch['anchors'],batch['anchor_grippers'][:,None]],1)
    persistence=np.repeat(last[:,None],3,1)
    commanded=np.concatenate([batch['commanded_poses'][:,np.asarray(HORIZONS)-1],batch['plans'][:,np.asarray(HORIZONS)-1,6:]],-1)
    dt=batch['obs']['time'][:,-1]-batch['obs']['time'][:,-2]
    elapsed=batch['future_time'][:,np.asarray(HORIZONS)-1]-batch['obs']['time'][:,-1:]
    ratio=elapsed/dt.clip(1e-8)[:,None];velocity=persistence.copy()
    velocity[...,:3]+=(last[:,:3]-batch['previous_poses'][:,:3])[:,None]*ratio[...,None]
    initial=Rotation.from_euler('xyz',last[:,3:6]);previous=Rotation.from_euler('xyz',batch['previous_poses'][:,3:6]);step=(previous.inv()*initial).as_rotvec()
    for j in range(3):velocity[:,j,3:6]=(initial*Rotation.from_rotvec(step*ratio[:,j,None])).as_euler('xyz')
    velocity[...,6]+=(last[:,6]-batch['previous_poses'][:,6])[:,None]*ratio
    return {'persistence':physical_state(persistence),'last-velocity':physical_state(velocity),'commanded-pose-as-state':physical_state(commanded)},ratio,elapsed


def fixed_plan(data,case):
    raw=data.load(str(CACHE/data.cfg['robot_family']/case['id']/'trajectory.npz'));at=case['start']+data.cfg['context']-1;q=data.cfg['horizon']
    anchor=np.repeat(raw['observed'][at:at+1],q,0)
    return np.concatenate([pose_delta(anchor,raw['commanded'][at:at+q]),raw['action'][at:at+q,6:]],1)


def paired(rows,baseline,key):
    by_id={r['id']:r for r in baseline};common=[r for r in rows if r['id'] in by_id]
    groups=sorted({r['group'] for r in common})
    if not groups:return {'groups':0,'gain_95ci':None,'candidate_error':None,'baseline_error':None}
    a=np.array([np.mean([r[key] for r in common if r['group']==g],0) for g in groups]);b=np.array([np.mean([by_id[r['id']][key] for r in common if r['group']==g],0) for g in groups])
    rng=np.random.default_rng(7391);draw=(b-a)[rng.integers(0,len(groups),(10000,len(groups)))].mean(1)
    return {'groups':len(groups),'candidate_error':a.mean(0).tolist(),'baseline_error':b.mean(0).tolist(),'gain':(b-a).mean(0).tolist(),'gain_95ci':np.quantile(draw,[.025,.975],axis=0).tolist()}


def subset(row,stratum):
    return {'unconditional-all':row['unconditional'],'unconditional-physical':row['unconditional'] and row['physical_clock'],'moving-physical':row['moving'] and row['physical_clock'],'unconditional-ordinal':row['unconditional'] and not row['physical_clock'],'moving-ordinal':row['moving'] and not row['physical_clock']}[stratum]


def evaluate_run(cfg,run,root):
    out=root/'consequences'/run['id'];out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    data=ReadoutData(cfg,'droid','test',True);registered=json.loads((PREPARED/'evaluation_cases.json').read_text())['cases']['droid'];data.cases=[]
    for row in registered:
        if row['role']!='test':continue
        uniform=set(row['starts']);starts=sorted(uniform|set(row['moving_starts']));moving=set(data.rows[row['id']]['moving_starts'])
        data.cases.extend({**row,'start':s,'unconditional':s in uniform,'moving':s in moving} for s in starts)
    stats=data.stats;all_plans=np.stack([fixed_plan(data,case) for case in data.cases]);wrong=np.roll(all_plans,max(1,len(data)//2),0)
    encoder=tiktoken.get_encoding('gpt2');records=[];saved_predictions=[];started=time.perf_counter()
    with open('/tmp/slwm_research_gpu.lock','a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX);saved=torch.load(root/run['id']/'endpoint.pt',map_location='cpu',weights_only=False,mmap=True)
        model=initialize(cfg,run).cuda().eval().requires_grad_(False);model.load_state_dict(saved['model']);del saved
        for start in range(0,len(data),16):
            stop=min(start+16,len(data));cases=data.cases[start:stop];batch=data.batch(range(start,stop));batch['obs'].update(supplied_text(batch['captions'],encoder));obs=tensor_tree(batch['obs'],'cuda')
            plans=(batch['plans']-stats['action_mean'])/stats['action_scale'];wrong_plan=(wrong[start:stop]-stats['action_mean'])/stats['action_scale'];prefix=plans.copy();prefix[:,:-1]=np.roll(prefix[:,:-1],3,axis=1)
            truth=batch['future_poses'][:,np.asarray(HORIZONS)-1];frames,last,previous=images16(data,batch,HORIZONS);pose,ratio,elapsed=controls(batch)
            pictures={'last-frame':np.repeat(last[:,None],3,1),'last-image-velocity':last[:,None]+(last-previous)[:,None]*ratio[...,None,None,None,None]}
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                state=model(obs);times=torch.from_numpy(batch['future_time']).cuda()
                for condition,plan in [('actual-plan',plans),('no-plan',None),('shuffled-plan',wrong_plan),('prefix-shuffled-final-fixed',prefix)]:
                    actions=None if plan is None else torch.from_numpy(plan.astype(np.float32)).cuda()
                    query=model.query(state,times,obs['time_unit'],2,actions)
                    # Direct state decoding: do not reintroduce G1 persistence.
                    normalized=model.propose(query,'numeric')[...,33:43]
                    physical=normalized.cpu().numpy()*stats['numeric_scale'][33:43]+stats['numeric_mean'][33:43]
                    pose[condition]=physical[:,np.asarray(HORIZONS)-1]
                    views=[]
                    for view in range(3):
                        q=model.query(state,times,obs['time_unit'],1,actions,view=view)
                        views.append(model.propose(q,'image').cpu().numpy()[:,np.asarray(HORIZONS)-1])
                    pictures[condition]=np.stack(views,2)
            for condition,value in pose.items():
                errors=state_errors(value,truth);errors['normalized_state_mse']=(((value-physical_state(truth))/stats['numeric_scale'][33:43])**2).mean(-1)
                for i,key in enumerate(batch['ids']):records.append({'id':key,'group':batch['groups'][i],'condition':condition,'kind':'pose','moving':cases[i]['moving'],'unconditional':cases[i]['unconditional'],'physical_clock':cases[i]['physical_timestamps'],**{k:v[i].tolist() for k,v in errors.items()}})
            duration=np.diff(np.concatenate([np.zeros((len(cases),1)),elapsed],1),axis=1).clip(1e-8)
            target_motion=np.diff(np.concatenate([last[:,None],frames],1),axis=1)/duration[...,None,None,None,None]
            for condition,value in pictures.items():
                pixel=((value-frames)**2).mean((2,3,4,5));motion=np.diff(np.concatenate([last[:,None],value],1),axis=1)/duration[...,None,None,None,None];error=((motion-target_motion)**2).mean((2,3,4,5))
                for i,key in enumerate(batch['ids']):records.append({'id':key,'group':batch['groups'][i],'condition':condition,'kind':'image','moving':cases[i]['moving'],'unconditional':cases[i]['unconditional'],'physical_clock':cases[i]['physical_timestamps'],'pixel_mse':pixel[i].tolist(),'motion_velocity_mse':error[i].tolist()})
            if start==0:np.savez_compressed(out/'fixed_first16_predictions.npz',truth=truth,frames=frames,**{'pose_'+k:v for k,v in pose.items()},**{'image_'+k:v for k,v in pictures.items()})
            saved_predictions.append(np.stack([pose[k] for k in ('actual-plan','no-plan','shuffled-plan','prefix-shuffled-final-fixed')]))
            write(out/'progress.json',{'processed':stop,'total':len(data),'no_fitting':True})
        del model;gc.collect();torch.cuda.empty_cache()
    write(out/'records.json',records);np.savez_compressed(out/'all_pose_predictions.npz',values=np.concatenate(saved_predictions,1))
    summaries={};effects={}
    for stratum in ('unconditional-all','unconditional-physical','moving-physical','unconditional-ordinal','moving-ordinal'):
        for kind in ('pose','image'):
            candidates=[r for r in records if r['kind']==kind and subset(r,stratum)];conditions=sorted({r['condition'] for r in candidates})
            for condition in conditions:
                rows=[r for r in candidates if r['condition']==condition];keys=set(rows[0])-TAGS
                # Motion velocities in different clock units are never pooled.
                keys-= {'motion_velocity_mse'} if stratum=='unconditional-all' else set()
                summaries[stratum+'/'+kind+'/'+condition]={k:np.mean([r[k] for r in rows],0).tolist() for k in keys}
                if condition=='actual-plan':continue
                reference=[r for r in candidates if r['condition']=='actual-plan']
                for key in keys:effects[stratum+'/'+kind+'/'+key+'/'+condition]=paired(reference,rows,key)
    write(out/'summary.json',{'horizons_steps':HORIZONS,'metrics':summaries,'paired_effects':effects,'groups':len({r['group'] for r in records}),'seconds':time.perf_counter()-started,'scope':'Actual recorded consequences; plan perturbations have no observed counterfactual outcomes','velocity_units':'Per second in physical-clock strata; per ordinal step otherwise; never pooled','rotation_invalid_penalty_rad':float(np.pi)})
    write(out/'complete.json',{'run':run['id'],'cases':len(data),'new_optimizer_updates':0,'source_sha256':file_sha(__file__)})


def main():
    torch.set_num_threads(4);cfg=json.loads(Path('configs/t3/predictive_world.json').read_text());root=Path(cfg['root']);register();assert (root/'evaluation_lock.json').exists()
    for run in cfg['runs']:evaluate_run(cfg,run,root)
    effects={}
    for full,point in [('G2-201','G2-211'),('G2-202','G2-212')]:
        a=json.loads((root/'consequences'/full/'records.json').read_text());b=json.loads((root/'consequences'/point/'records.json').read_text())
        keep=lambda r:r['kind']=='pose' and r['condition']=='actual-plan' and subset(r,'moving-physical')
        effect=paired([r for r in a if keep(r)],[r for r in b if keep(r)],'normalized_state_mse')
        effect['passes_five_percent_moving_dynamics']=bool(effect['groups']>=5 and effect['gain_95ci'][0][-1]>0 and (effect['baseline_error'][-1]-effect['candidate_error'][-1])/max(1e-12,effect['baseline_error'][-1])>=.05)
        effects[full+'_vs_'+point]=effect
    write(root/'consequences/summary.json',{'history_vs_point':effects,'one_core_seed_mechanism_comparison':True,'scope':'Eighth-step moving physical-clock normalized pose MSE; all component errors retained'})


if __name__=='__main__':main()
