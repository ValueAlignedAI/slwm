"""Official UCI HAR signals: completely withheld from G2 core pretraining."""
import io,json,zipfile
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from training.general_data import order
from training.general_sources import write
from training.research_sources import fetch,file_sha

ROOT=Path('artifacts/predictive_world/g2/har')
URL='https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'complete.json').exists():return
    path=ROOT/'source.zip';receipt=fetch(URL,path)
    archive=zipfile.ZipFile(path)
    if not any('Inertial Signals/total_acc_x_train.txt' in n for n in archive.namelist()):
        inner=next(n for n in archive.namelist() if n.endswith('.zip'));archive=zipfile.ZipFile(io.BytesIO(archive.read(inner)))
    def read(suffix):
        name=next(n for n in archive.namelist() if n.endswith(suffix))
        return np.loadtxt(io.BytesIO(archive.read(name)))
    values=[];labels=[];people=[];splits=[];ids=[]
    for role in ('train','test'):
        raw=np.stack([read(role+'/Inertial Signals/'+kind+'_'+axis+'_'+role+'.txt') for kind in ('total_acc','body_gyro') for axis in 'xyz'],-1).astype(np.float32)
        raw[:,:,:3]*=9.80665
        # Cover the whole labeled2.56s window, filtering before downsampling.
        signal=resample_poly(raw,1,4,axis=1).astype(np.float32);assert signal.shape[1:]==(32,6)
        values.append(signal);labels.extend(read(role+'/y_'+role+'.txt').astype(int).tolist());people.extend(read(role+'/subject_'+role+'.txt').astype(int).tolist())
        splits.extend([role]*len(raw));ids.extend(['HAR-'+role+'-'+str(i) for i in range(len(raw))])
    people=np.asarray(people);train_people=sorted(set(people[np.asarray(splits)=='train']),key=order);test_people=sorted(set(people[np.asarray(splits)=='test']))
    assert len(train_people)==21 and len(test_people)==9 and not set(train_people)&set(test_people)
    roles={int(p):'train' if i<16 else 'selection' for i,p in enumerate(train_people)};roles.update({int(p):'test' for p in test_people})
    np.save(ROOT/'signals.npy',np.concatenate(values));np.save(ROOT/'labels.npy',np.asarray(labels,np.int64));np.save(ROOT/'people.npy',people.astype(np.int64))
    rows=[{'id':key,'index':i,'person':'HAR-person-'+str(person),'role':roles[int(person)]} for i,(key,person) in enumerate(zip(ids,people))]
    write(ROOT/'rows.json',rows)
    write(ROOT/'complete.json',{'source':receipt,'source_url':URL,'license':'CC BY4.0, UCI dataset240; Anguita et al./Reyes-Ortiz et al.2013','core_training_examples':0,'labels':6,'source_hz':50,'model_hz':12.5,'source_context':128,'model_context':32,'acceleration':'total acceleration, g converted to m/s²','gyro':'source body angular velocity, rad/s','placement':'waist phone mapped to first six scalar adapter channels; no claim it was a wrist sensor','overlap':'Published128-sample windows overlap50%; all windows of a person stay in one role; bootstrap people, not windows','role_people':{r:sorted(p for p,v in roles.items() if v==r) for r in ('train','selection','test')},'files':{p.name:file_sha(p) for p in ROOT.iterdir() if p.suffix in ('.npy','.json')},'preparer_sha256':file_sha(__file__)})


if __name__=='__main__':main()
