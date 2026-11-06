"""Independent-modal G2 retrieval; category agreement is not a physical event."""
import fcntl,gc,json
from pathlib import Path
import numpy as np
import torch
from training.general_alignment_eval import observations,semantic_rows,group_gain,normalize
from training.general_batch import tensor_tree
from training.general_data import CACHE
from training.general_sources import write
from training.predictive_data import Sources
from training.predictive_eval import register
from training.predictive_train import initialize


def edge(model,obs):
    if 'audio' in obs:return torch.nn.functional.normalize(model.audio_edge(obs['audio']).mean((1,2)).float(),dim=-1)
    if 'images' in obs:
        x=obs['images'];b,t,v,c,h,w=x.shape
        return torch.nn.functional.normalize(model.vision_edge(x.reshape(b*t*v,c,h,w)).mean((2,3)).reshape(b,t*v,-1).mean(1).float(),dim=-1)
    x=model.text_projection(model.text_edge(obs['text']));mask=obs['text_present']
    return torch.nn.functional.normalize(((x*mask[...,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None]).float(),dim=-1)


def encode(model,source,family,rows):
    a=[];b=[]
    for start in range(0,len(rows),64):
        obs=source.text_observation(rows[start:start+64]) if family=='text' else observations(source,family,rows[start:start+64])
        obs=tensor_tree(obs,'cuda')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):a.append(model(obs)['embedding'].cpu().numpy());b.append(edge(model,obs).cpu().numpy())
    return np.concatenate(a),np.concatenate(b)


def caption_scores(values,captions,rows):
    strings=[s.strip().lower() for r in rows for s in r['captions']];owner=np.repeat(np.arange(len(rows)),[len(r['captions']) for r in rows])
    positive=owner[None,:]==np.arange(len(rows))[:,None];by_text={}
    for record,row in enumerate(rows):
        for text in row['captions']:by_text.setdefault(text.strip().lower(),set()).add(record)
    for i,text in enumerate(strings):positive[list(by_text[text]),i]=True
    scores=values@captions.T;rank=np.argsort(-scores,axis=1,kind='stable');reverse=scores.argmax(0)
    return {'signal_to_text_recall1':float(positive[np.arange(len(rows)),rank[:,0]].mean()),'signal_to_text_recall5':float(np.take_along_axis(positive,rank[:,:5],1).any(1).mean()),'text_to_signal_recall1':float(positive[reverse,np.arange(len(strings))].mean()),'uniform_signal_to_text_chance':float(positive.mean()),'records':len(rows),'captions':len(strings),'duplicate_caption_texts_are_multi_positive':True}


def main():
    torch.set_num_threads(4);cfg=json.loads(Path('configs/t3/predictive_world.json').read_text());root=Path(cfg['root']);register();assert (root/'evaluation_lock.json').exists();source=Sources(cfg,'test')
    out=root/'alignment';out.mkdir(exist_ok=True);semantic=json.loads((CACHE/'evaluation_cases.json').read_text())['semantic'];classes=list(semantic['aliases']);embeddings={};retrieval={}
    runs=list(cfg['runs'])
    for original in cfg['runs']:
        if original['id'] in ('G2-201','G2-202','G2-205','G2-206'):runs.append({**original,'id':'RND-'+original['id'].split('-')[-1],'untrained':True})
    for run in runs:
        name=run['id'];path=out/(name+'.npz')
        if path.exists():embeddings[name]=dict(np.load(path));retrieval[name]=json.loads((out/(name+'_captions.json')).read_text());continue
        with open('/tmp/slwm_research_gpu.lock','a') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX);saved=torch.load(root/name/'endpoint.pt',map_location='cpu',weights_only=False,mmap=True);model=initialize(cfg,run).cuda().eval().requires_grad_(False);model.load_state_dict(saved['model']);del saved
            current={};scores={};current['prototype'],current['edge_prototype']=encode(model,source,'text',classes)
            for family in ('coco','clotho'):
                rows=source.rows[family];current[family],current['edge_'+family]=encode(model,source,family,rows)
                captions,_=encode(model,source,'text',[s for r in rows for s in r['captions']]);scores[family]=caption_scores(current[family],captions,rows)
            np.savez_compressed(path,**current);write(out/(name+'_captions.json'),scores);embeddings[name]=current;retrieval[name]=scores
            del model;gc.collect();torch.cuda.empty_cache()
    labels={r['id']:r['labels'] for r in semantic['coco'] if r['role']=='test'};images=[{'id':r['id'],'labels':labels.get(r['id'],[])} for r in source.rows['coco']]
    metadata={r['id']:r for r in semantic['clotho'] if r['role']=='test'};indices=[i for i,r in enumerate(source.rows['clotho']) if r['id'] in metadata];queries=[metadata[source.rows['clotho'][i]['id']] for i in indices]
    cases={};summaries={}
    for name,current in embeddings.items():
        raw=current['clotho'][indices]@current['coco'].T;rows=semantic_rows(raw,queries,images)
        def pivot(values):
            scores=values@current['edge_prototype'].T;return normalize(scores-scores.mean(1,keepdims=True))
        bridge=semantic_rows(pivot(current['edge_clotho'][indices])@pivot(current['edge_coco']).T,queries,images)
        cases[name]=rows;cases[name+'-edge-bridge']=bridge
        per_class={label:{'queries':sum(label in r['labels'] for r in rows),'hit1':float(np.mean([r['class_hit1'][label] for r in rows if label in r['labels']]))} for label in classes if any(label in r['labels'] for r in rows)}
        summaries[name]={'hit1':float(np.mean([r['hit1'] for r in rows])),'hit5':float(np.mean([r['hit5'] for r in rows])),'macro_class_hit1':float(np.mean([v['hit1'] for v in per_class.values()])),'per_class':per_class,'edge_bridge_hit1':float(np.mean([r['hit1'] for r in bridge])),'gain_over_edge_bridge':group_gain(rows,bridge)}
    write(out/'semantic_cases.json',cases);write(out/'summary.json',{'caption_retrieval':retrieval,'sound_to_image':summaries,'queries':len(queries),'images':len(images),'fixed_categories':classes,'scope':'Unpaired semantic category retrieval, not matching physical events; edge text bridge is diagnostic, not a separately trained specialist'})


if __name__=='__main__':main()
