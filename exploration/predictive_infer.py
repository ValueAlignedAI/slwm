"""Load actual G2 checkpoints and frozen state-only actors for ongoing inference."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch

from models.predictive_world import PredictiveWorld
from training.general_batch import tensor_tree
from training.predictive_features import project
from training.predictive_readouts import Readout
from training.research_sources import file_sha


def load_world(path,device='cuda'):
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True);cfg=saved['config'];run=saved['run']
    expected=saved.get('sources',{}).get('models/predictive_world.py')
    if expected and expected!=file_sha('models/predictive_world.py'):raise RuntimeError('Use the checkpoint source snapshot or its recorded Git version')
    model=PredictiveWorld(width=run['width'],layers=run['layers'],variant=run['variant'],heads=cfg['heads'],slots=cfg['slots'],memory=cfg['memory'],phone_count=saved.get('phone_count',79),seed=run['seed'],bypass=run.get('bypass',False),action_history=run.get('action_history',True))
    model.load_state_dict(saved['model']);return model.to(device).eval().requires_grad_(False),saved


def load_actor(bundle):
    saved=torch.load(bundle,map_location='cpu',weights_only=False);heads=[]
    for item in saved['heads']:
        model=Readout(item['inputs'],item['outputs'],item['hidden']);model.load_state_dict(item['model']);model.eval().requires_grad_(False)
        assert not len(item['projection']['common_mean']),'Primary G2 actor must have no direct raw-input bypass'
        heads.append((model,item['projection']))
    return heads,saved


def actor_proposal(state,heads):
    # Match the saved feature-cache quantization exactly.
    value=state['shared'].float().cpu().numpy().astype(np.float16).astype(np.float32);outputs=[]
    for model,fitted in heads:
        x=project(value,fitted).astype(np.float32)
        with torch.inference_mode():prediction=model(torch.from_numpy(x)).numpy()
        outputs.append(prediction*fitted['target_scale']+fitted['target_mean'])
    return np.mean(outputs,axis=0)


def weight_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class LiveWorld:
    """Persistent observation cycle; proposals never actuate a physical device."""
    def __init__(self,model,heads=None):
        self.model=model;self.heads=heads;self.memory=None;self.last_packet=None;self.updates=0

    def observe(self,received_window,packet_time):
        if self.last_packet is not None and packet_time<=self.last_packet:return {'status':'stale_rejected','updates':self.updates}
        self.last_packet=packet_time
        present=[received_window[k][:,-1] for k in ('numeric_present','audio_present','image_present') if k in received_window]
        if not present and 'text_present' in received_window:present=[received_window['text_present']]
        if not any(bool(v.any()) for v in present):return {'status':'missing_observation','updates':self.updates}
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=next(self.model.parameters()).is_cuda):
            state=self.model(received_window) if self.memory is None else self.model.stream_step(received_window,self.memory)
        self.memory=state['memory'];self.last_packet=packet_time;self.updates+=1
        result={'status':'internal_state_updated','updates':self.updates,'state':state}
        if self.heads:result.update(status='internal_command_proposal',command=actor_proposal(state,self.heads))
        return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True,type=Path);parser.add_argument('--observations',required=True,type=Path);parser.add_argument('--actor',type=Path);parser.add_argument('--output',required=True,type=Path);parser.add_argument('--device',default='cuda');parser.add_argument('--future',type=Path,help='NPZ: increasing time[B,Q] and optional physical body-frame plans[B,Q,7]');parser.add_argument('--family',choices=('numeric','image','audio'),default='numeric');args=parser.parse_args()
    model,saved=load_world(args.checkpoint,args.device);values=dict(np.load(args.observations,allow_pickle=False));obs=tensor_tree(values,args.device)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=args.device.startswith('cuda')):state=model(obs)
    result={'state':state['shared'].cpu().numpy(),'embedding':state['embedding'].cpu().numpy()}
    if args.future:
        future=dict(np.load(args.future,allow_pickle=False));times=torch.as_tensor(future['time'],device=args.device);plan=None
        if 'plans' in future:
            stats=saved['normalizers'];plan=torch.as_tensor((future['plans']-stats['action_mean'])/stats['action_scale'],device=args.device,dtype=torch.float32)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=args.device.startswith('cuda')):
            query=model.query(state,times,obs['time_unit'],{'audio':0,'image':1,'numeric':2}[args.family],plan,view=0 if args.family=='image' else None)
            result['future_'+args.family]=model.propose(query,args.family).cpu().numpy()
        result['future_time']=future['time']
    if args.actor:
        heads,policy=load_actor(args.actor);assert policy['core_endpoint_sha256']==file_sha(args.checkpoint)
        result['command']=actor_proposal(state,heads)
    args.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.output,**result)
    print(json.dumps({'output':str(args.output),'status':'internal_command_proposal' if args.actor else 'internal_world_state','command_units':['body_translation_m','rotation_vector_rad','gripper'],'physical_action_sent':False}))


if __name__=='__main__':main()
