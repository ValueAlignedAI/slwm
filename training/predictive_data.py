"""G2-only data contract, robot clocks and training-only normalization."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from training.general_batch import Sources as LegacySources,base_observation,normalize,rotation6d,numeric_baseline,tensor_tree,ROBOT_STATE,ROBOT_ACTION
from training.general_data import ROOT,CACHE,order,pose_delta
from training.general_dataset import Moments
from training.general_sources import write
from training.research_sources import file_sha

CONFIG='configs/t3/predictive_world.json'
PREPARED=Path('artifacts/predictive_world/g2')


def moving_starts(trajectory,context=32,horizon=8):
    n=len(trajectory['observed'])-context-horizon+1
    if n<=0:return np.empty(0,np.int64)
    starts=np.arange(max(0,n));a=trajectory['observed'][starts+context-1];b=trajectory['observed'][starts+context+horizon-1]
    translation=np.linalg.norm(a[:,:3]-b[:,:3],axis=1)
    rotation=(Rotation.from_euler('xyz',a[:,3:]).inv()*Rotation.from_euler('xyz',b[:,3:])).magnitude()
    grip=np.abs(trajectory['gripper'][starts+context+horizon-1,0]-trajectory['gripper'][starts+context-1,0])
    return starts[(translation>=.01)|(rotation>=.1)|(grip>=.1)]


class Sources(LegacySources):
    def __init__(self,cfg,role='train'):
        super().__init__(cfg,role);self.stats=dict(np.load(cfg['normalizers'],allow_pickle=False))
        self.rows['droid']=[r for r in json.loads((PREPARED/'robot_rows.json').read_text()) if r['role']==role]
        self.long=None;self.motion_cache={}
        if role=='train' and cfg['context']==32 and 64 in cfg.get('training_contexts',[]):
            self.long=Sources({**cfg,'context':64,'batch_temporal':cfg['batch_temporal']//2},role)

    def make(self,family,position,seed=201,device='cpu',batch=None):
        if self.long is not None and position//self.cfg['accumulation']%4==3:
            return self.long.make(family,position,seed,device,batch)
        if family!='droid':return super().make(family,position,seed,device,batch)
        rng=np.random.default_rng(seed*1000003+position);t=self.cfg['context'];q=self.cfg['horizon'];b=batch or self.cfg['batch_temporal'];rows=self.pick(family,rng,b)
        values=[];futures=[];pictures=[];future_pictures=[];plans=[];captions=[];clocks=[];units=[];starts=[]
        for row in rows:
            folder=CACHE/self.robot_family/row['id'];trajectory=self.load(str(folder/'trajectory.npz'));state=rotation6d(trajectory['observed'],trajectory['gripper'])
            key=(row['id'],t)
            if key not in self.motion_cache:self.motion_cache[key]=moving_starts(trajectory,t,q)
            choices=self.motion_cache[key]
            start=int(rng.choice(choices)) if self.role=='train' and len(choices) and rng.random()<.5 else int(rng.integers(len(state)-t-q+1));starts.append(start)
            numeric=np.full((t+q,64),np.nan,np.float32);numeric[:,ROBOT_STATE]=normalize(state[start:start+t+q],self.stats['numeric_mean'][ROBOT_STATE],self.stats['numeric_scale'][ROBOT_STATE])
            indices=start+np.arange(t);previous=(indices-1).clip(0)
            past=np.concatenate([pose_delta(trajectory['observed'][indices],trajectory['commanded'][previous]),trajectory['action'][previous,6:]],1)
            numeric[:t,ROBOT_ACTION]=normalize(past,self.stats['action_mean'],self.stats['action_scale']);numeric[:t][indices==0,ROBOT_ACTION]=np.nan
            values.append(numeric[:t]);futures.append(numeric[t:])
            anchor=np.repeat(trajectory['observed'][start+t-1:start+t],q,axis=0)
            plan=np.concatenate([pose_delta(anchor,trajectory['commanded'][start+t-1:start+t+q-1]),trajectory['action'][start+t-1:start+t+q-1,6:]],1)
            plans.append(normalize(plan,self.stats['action_mean'],self.stats['action_scale']))
            image=np.asarray(self.load(str(folder/'rgb.npy'))[start:start+t+q],np.float32).transpose(0,1,4,2,3)/127.5-1;pictures.append(image[:t]);future_pictures.append(image[t:])
            captions.append(row['captions'][int(rng.integers(len(row['captions'])))] if row['captions'] else '')
            clock=np.asarray(self.load(str(folder/'clock.npy'))[start:start+t+q]);clocks.append((clock-clock[0]).astype(np.float32));units.append(0 if row['physical_timestamps'] else 1)
        original=np.stack(values);obs=base_observation(b,t,1.,True);obs.update(time=np.stack(clocks)[:,:t],time_unit=np.asarray(units,np.int64),numeric=np.nan_to_num(original),numeric_present=np.isfinite(original),images=np.stack(pictures),image_present=np.ones((b,t,3),bool))
        hidden=int(rng.integers(3));cross=bool(rng.random()<.5)
        if cross:obs['image_present'][:,:,hidden]=False;obs['images'][:,:,hidden]=0
        text=self.text_observation(captions);obs.update(text=text['text'],text_present=text['text_present'])
        targets=np.stack(futures);future=base_observation(b,q,1.,True);future.update(time=np.stack(clocks)[:,t:],time_unit=np.asarray(units,np.int64),numeric=np.nan_to_num(targets),numeric_present=np.isfinite(targets),images=np.stack(future_pictures),image_present=np.ones((b,q,3),bool))
        picture=np.stack(pictures)[:,:,hidden].reshape(b,t,3,16,4,16,4).mean((4,6));future_picture=np.stack(future_pictures)[:,:,hidden].reshape(b,q,3,16,4,16,4).mean((4,6))
        result={'family':'droid','ids':[r['id'] for r in rows],'groups':[r['group'] for r in rows],'starts':starts,'captions':captions,'caption_hashes':[order(c.strip().lower()) for c in captions],'companion':self.text_observation(captions),'obs':obs,'query_time':obs['time'].copy(),'numeric_target':np.nan_to_num(original),'numeric_target_present':np.isfinite(original),'numeric_hidden':np.zeros_like(original,bool),'numeric_baseline':np.nan_to_num(original).copy(),'picture_target':picture,'picture_hidden':cross,'picture_baseline':np.zeros_like(picture) if cross else picture.copy(),'hidden_view':hidden,'future_obs':future,'future_time':future['time'].copy(),'future_numeric':np.nan_to_num(targets),'future_numeric_present':np.isfinite(targets),'future_picture':future_picture,'planned_actions':np.stack(plans).astype(np.float32),'physical_observed_seconds':float(sum(c[t-1]-c[0] for c,u in zip(clocks,units) if u==0))}
        return tensor_tree(result,device)


def freeze():
    cfg=json.loads(Path(CONFIG).read_text());PREPARED.mkdir(parents=True,exist_ok=True)
    marker=Path(cfg['dataset_manifest'])
    if marker.exists():
        saved=json.loads(marker.read_text());assert saved['configuration_sha256']==file_sha(CONFIG);return saved
    acquisition=ROOT/cfg['robot_family']/'acquisition_complete.json';assert acquisition.exists()
    path=ROOT/cfg['robot_family']/'raw_metadata/scene_role_plan.json';plan=json.loads(path.read_text());rows=plan['rows'];assert sum(r['role']=='train' for r in rows)>=400
    previous=dict(np.load(CACHE/'normalizers.npz'));numeric=Moments(10);action=Moments(7);cases=[];live=[];hours={role:0. for role in ('train','selection','calibration','test')}
    assert sum(r['role']=='train' and r['steps']>=40 for r in rows)>=400
    assert sum(r['role']=='train' and r['steps']>=72 for r in rows)>=300
    for row in rows:
        assert all(file_sha(CACHE/cfg['robot_family']/row['id']/name)==digest for name,digest in row['files'].items()),'Robot cache changed after acquisition'
        if row['physical_timestamps']:hours[row['role']]+=row['clock_audit']['seconds']/3600
        data=np.load(CACHE/cfg['robot_family']/row['id']/'trajectory.npz');moving=moving_starts(data,cfg['context'],cfg['horizon']);row['moving_starts']=moving.tolist()
        if row['role']=='train':numeric.add(rotation6d(data['observed'],data['gripper']));action.add(data['action'])
        maximum=row['steps']-cfg['context']-cfg['horizon']+1
        if maximum<=0:continue
        rng=np.random.default_rng(int(order(row['id'])[:16],16));starts=rng.choice(maximum,min(32,maximum),replace=False)
        moving_selected=rng.choice(moving,min(32,len(moving)),replace=False)
        cases.append({'id':row['id'],'role':row['role'],'group':row['group'],'starts':sorted(starts.tolist()),'moving_starts':sorted(moving_selected.tolist()),'physical_timestamps':row['physical_timestamps']})
        if row['role']=='test' and row['physical_timestamps'] and row['steps']>=64:
            clock=np.load(CACHE/cfg['robot_family']/row['id']/'clock.npy')
            if clock[-9]-clock[31]>=41 and np.diff(clock).max()<=.25:live.append(row['id'])
    nm,ns=numeric.finish();am,asc=action.finish();previous['numeric_mean'][ROBOT_STATE]=nm;previous['numeric_scale'][ROBOT_STATE]=ns;previous['numeric_mean'][ROBOT_ACTION]=am;previous['numeric_scale'][ROBOT_ACTION]=asc;previous.update(action_mean=am,action_scale=asc)
    np.savez(cfg['normalizers'],**previous)
    # Own metadata includes moving starts; G1 rows/contracts stay immutable.
    write(PREPARED/'robot_rows.json',rows)
    assert len({r['group'] for r in cases if r['role']=='test'})>=5
    old_cases=json.loads((CACHE/'evaluation_cases.json').read_text());write(PREPARED/'evaluation_cases.json',{'cases':{'droid':cases,'mhealth':old_cases['cases']['mhealth']},'live_robot_id':min(live,key=order) if live else None,'moving_definition':'At eighth future step: translation>=.01m OR rotation>=.1rad OR gripper change>=.1; fixed before fitting','source_sha256':file_sha(__file__)})
    har=PREPARED/'har/complete.json';assert har.exists()
    audit=PREPARED/'decoder_recovery_audit.json';assert json.loads(audit.read_text())['passed']
    result={'configuration_sha256':file_sha(CONFIG),'old_frozen_cache_hashes_sha256':file_sha(CACHE/'cache_hashes.json'),'robot_acquisition_sha256':file_sha(acquisition),'robot_roles_sha256':file_sha(path),'robot_rows_sha256':file_sha(PREPARED/'robot_rows.json'),'normalizers_sha256':file_sha(cfg['normalizers']),'evaluation_cases_sha256':file_sha(PREPARED/'evaluation_cases.json'),'har_sha256':file_sha(har),'training_robot_episodes':sum(r['role']=='train' for r in rows),'training_robot_steps':sum(r['steps'] for r in rows if r['role']=='train'),'training_moving_windows':sum(len(r['moving_starts']) for r in rows if r['role']=='train'),'role_counts':{role:sum(r['role']==role for r in rows) for role in ('train','selection','calibration','test')},'clock_verified_counts':{role:sum(r['role']==role and r['physical_timestamps'] for r in rows) for role in ('train','selection','calibration','test')},'normalization_scope':'Existing nonrobot training-only moments; robot/action moments recomputed from G2 training groups only','core_never_receives_har_or_mhealth_labels':True}
    result.update(unique_robot_physical_hours=hours,robot_group_counts=plan['group_counts'],decoder_recovery_audit_sha256=file_sha(audit),robot_source_plan_sha256=file_sha(ROOT/cfg['robot_family']/'source_plan.json'),eligible_training_32=sum(r['role']=='train' and r['steps']>=40 for r in rows),eligible_training_64=sum(r['role']=='train' and r['steps']>=72 for r in rows))
    write(marker,result);return result


if __name__=='__main__':print(json.dumps(freeze()))
