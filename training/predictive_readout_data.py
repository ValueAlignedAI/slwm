"""Core-unseen task labels, new robot roles and true packet clocks."""
import json
import numpy as np
from training.general_readout_data import ReadoutData as Base,COMMON
from training.general_batch import base_observation
from training.general_data import CACHE
from training.predictive_data import PREPARED


class ReadoutData(Base):
    def __init__(self,cfg,family,role,allow_final=False):
        if role=='test' and not allow_final:raise RuntimeError('Final observations remain locked')
        self.cfg,self.family,self.role=cfg,family,role
        self.stats=np.load(cfg['normalizers'],allow_pickle=False)
        if family=='har':
            rows=json.loads((PREPARED/'har/rows.json').read_text());self.rows={r['id']:r for r in rows if r['role']==role};self.cases=list(self.rows.values());return
        manifest=json.loads((PREPARED/'evaluation_cases.json').read_text())
        self.cases=[{**r,'start':start} for r in manifest['cases'][family] if r['role']==role for start in r['starts']]
        path=PREPARED/'robot_rows.json' if family=='droid' else CACHE/family/'rows.json'
        self.rows={r['id']:r for r in json.loads(path.read_text()) if r['role']==role}

    def batch(self,indices,goal_control='provided'):
        cases=[self.cases[int(i)] for i in indices]
        if self.family=='har':return self._har(cases)
        if self.family=='mhealth':return self._sensor(cases)
        result=super()._robot(cases,goal_control);clocks=[];units=[]
        for case in cases:
            row=self.rows[case['id']];start=case['start'];n=self.cfg['context']+self.cfg['horizon']
            clock=np.asarray(self.load(str(CACHE/self.cfg['robot_family']/row['id']/'clock.npy'))[start:start+n]);clocks.append((clock-clock[0]).astype(np.float32));units.append(0 if row['physical_timestamps'] else 1)
        result['obs']['time']=np.stack(clocks)[:,:self.cfg['context']];result['obs']['time_unit']=np.asarray(units,np.int64)
        result['future_time']=np.stack(clocks)[:,self.cfg['context']:]
        return result

    def _har(self,cases):
        indices=[r['index'] for r in cases];raw=np.asarray(self.load(str(PREPARED/'har/signals.npy'))[indices],np.float32)
        b,t,_=raw.shape;numeric=np.zeros((b,t,64),np.float32);present=np.zeros_like(numeric,bool)
        numeric[:,:,:6]=(raw-self.stats['numeric_mean'][:6])/self.stats['numeric_scale'][:6];present[:,:,:6]=True
        obs=base_observation(b,t,.08);obs.update(numeric=numeric,numeric_present=present)
        physical=np.zeros((b,t,18),np.float32);physical[:,:,:6]=raw;known=np.zeros_like(physical,bool);known[:,:,:6]=True
        return {'obs':obs,'activity':np.asarray(self.load(str(PREPARED/'har/labels.npy'))[indices],np.int64),'future':np.zeros((b,8,18),np.float32),'future_present':np.zeros((b,8,18),bool),'last_physical':physical[:,-1],'previous_physical':physical[:,-2],'last_present':known[:,-1],'groups':[r['person'] for r in cases],'ids':[r['id'] for r in cases],'common':np.zeros((b,0),np.float32)}
