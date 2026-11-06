"""Fresh-core-speaker phonetic retention; no vocoder/ASR intelligibility claim."""
import fcntl,gc,json
from pathlib import Path
import numpy as np
import torch
from training.general_losses import without
from training.general_sources import write
from training.predictive_data import Sources
from training.predictive_eval import register
from training.predictive_train import initialize


def main():
    torch.set_num_threads(4);cfg=json.loads(Path('configs/t3/predictive_world.json').read_text());root=Path(cfg['root']);register();assert (root/'evaluation_lock.json').exists();source=Sources(cfg,'test');out=root/'speech_retention';out.mkdir(exist_ok=True)
    vocabulary=json.loads((Path(cfg['data_root'])/'speech/phone_vocabulary.json').read_text());silence=[i for i,s in enumerate(vocabulary) if s.strip().lower() in ('sil','sp','spn','<sil>','<blank>','<unk>')];results={}
    for run in cfg['runs']:
        path=out/(run['id']+'.json')
        if path.exists():results[run['id']]=json.loads(path.read_text());continue
        cases=[]
        with open('/tmp/slwm_research_gpu.lock','a') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX);saved=torch.load(root/run['id']/'endpoint.pt',map_location='cpu',weights_only=False,mmap=True);model=initialize(cfg,run).cuda().eval().requires_grad_(False);model.load_state_dict(saved['model']);del saved
            for index in range(64):
                batch=source.make('speech',23000000+index,9001,'cuda',16);obs=without(batch['obs'],'text')
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    state=model(obs);query=model.query(state,batch['query_time'],obs['time_unit'],0);phones=model.propose(query,'phones').argmax(-1);mel=model.propose(query,'audio')
                target=batch['phones'];valid=(target>=0)&~torch.isin(target,torch.tensor(silence,device='cuda'))
                hidden=batch['mel_hidden'].reshape(len(target),cfg['context'],4,100).any(-1)&valid
                for i,key in enumerate(batch['ids']):
                    if batch['modes'][i]!='repair':continue
                    count=int(hidden[i].sum());errors=int(((phones[i]!=target[i])&hidden[i]).sum())
                    cases.append({'id':batch.get('alignment_ids',batch['ids'])[i]+'@'+str(index)+':'+str(i),'group':batch['groups'][i],'nonsilent_hidden_phones':count,'phone_errors':errors,'direct_hidden_mel_mse':float(((mel[i]-batch['mel_target'][i]).square()*batch['mel_hidden'][i]).sum()/batch['mel_hidden'][i].sum().clamp_min(1))})
            del model;gc.collect();torch.cuda.empty_cache()
        errors=sum(r['phone_errors'] for r in cases);count=sum(r['nonsilent_hidden_phones'] for r in cases)
        value={'run':run['id'],'cases':cases,'people':len({r['group'] for r in cases}),'nonsilent_hidden_phone_error':errors/max(1,count),'phone_frames':count,'scope':'Audio-only repaired-context phone classification on core-unexposed voices; no waveform intelligibility claim and no external decoder'}
        write(path,value);results[run['id']]=value
    effects={}
    for a,b in [('G2-201','G2-202'),('G2-203','G2-204'),('G2-201','G2-205'),('G2-203','G2-207')]:
        left={r['id']:r for r in results[a]['cases']};right={r['id']:r for r in results[b]['cases']};assert left.keys()==right.keys()
        groups=sorted({r['group'] for r in left.values()});delta=[]
        for group in groups:
            keys=[k for k,r in left.items() if r['group']==group];count=sum(left[k]['nonsilent_hidden_phones'] for k in keys)
            assert count==sum(right[k]['nonsilent_hidden_phones'] for k in keys)
            if count:delta.append(sum(right[k]['phone_errors']-left[k]['phone_errors'] for k in keys)/count)
        rng=np.random.default_rng(7391);values=np.asarray(delta);draw=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
        effects[a+'_vs_'+b]={'people':len(values),'equal_person_phone_error_gain':float(values.mean()),'gain_95ci':np.quantile(draw,[.025,.975]).tolist()}
    write(out/'summary.json',{'runs':{name:{k:v for k,v in value.items() if k!='cases'} for name,value in results.items()},'paired_person_effects':effects})


if __name__=='__main__':main()
