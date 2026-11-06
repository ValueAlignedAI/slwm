"""G2: all proposals pass through a recurrent, fixed-capacity world state."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.slwm_general import GeneralWorld


def join_observations(past, future):
    """Join observed[B,T,...] and teacher-only future[B,Q,...]; erase missing edges."""
    b,t=past['time'].shape;q=future['time'].shape[1];result={}
    if not torch.equal(past['time_unit'],future['time_unit']):raise ValueError('Teacher continuation changes clock units')
    for key in set(past)|set(future):
        if key in ('time_unit','text','text_present'):
            if key in past:result[key]=past[key]
            continue
        a=past.get(key);z=future.get(key);sample=a if a is not None else z
        fill=1 if key in ('pixel_present','patch_present','image_weight') else 0
        if a is None:a=sample.new_full((b,t,*sample.shape[2:]),fill)
        if z is None:z=sample.new_full((b,q,*sample.shape[2:]),fill)
        result[key]=torch.cat((a,z),1)
    return result


class PredictiveWorld(GeneralWorld):
    """Typed observations -> state[B,M]; forecasts -> states[B,Q,M].

    The main path exposes no encoder field to output queries. A registered bypass
    control adds field attention; a point-dynamics control resets each future
    transition to the same received state. Neither control adds parameters.
    """
    def __init__(self,*,memory=512,action_history=True,bypass=False,**kwargs):
        super().__init__(**kwargs)
        self.memory_width=memory;self.action_history=action_history;self.bypass=bypass
        # The inherited field encoder needs no separate trainable pooled probe.
        self.embedding=nn.Identity();self.future_embedding=nn.Identity()
        del self.action
        seed=kwargs.get('seed',101)
        with torch.random.fork_rng():
            torch.manual_seed(seed+33000)
            self.history_input=nn.Sequential(nn.Linear(self.slots*self.width,memory),nn.SiLU())
            self.history=nn.GRU(memory,memory,batch_first=True)
            self.state_norm=nn.LayerNorm(memory)
            self.state_slots=nn.Linear(memory,self.slots*self.width)
            self.transition=nn.GRU(self.width,memory,batch_first=True)
            self.transition_interval=nn.Linear(3,self.width)
            self.world_embedding=nn.Linear(memory,128)

    def _field(self,obs):
        return super().forward(obs)['field']

    def _state(self,obs,field,posterior,hidden):
        shared=self.state_norm(posterior.float())
        raw=self.world_embedding(shared[:,-1]).float()
        result={'shared':shared[:,-1],'posterior':shared,'memory':hidden,
                'raw_embedding':raw,'embedding':F.normalize(raw,dim=-1),
                'sequence_embedding':F.normalize(self.world_embedding(shared).float(),dim=-1),
                'last':shared[:,-1],'valid':obs['valid'],'time_origin':obs['time'][:,:1],
                'last_time':obs['time'][:,-1:],'time_unit':obs['time_unit']}
        if self.bypass:result['field']=field
        return result

    def forward(self,obs,memory=None):
        """Received[B,T,...] plus optional hidden[1,B,M] -> compact posterior."""
        if not bool(obs['valid'].all()):
            raise ValueError('G2 training requires contiguous received packets; erase missing modalities with their masks')
        field=self._field(obs);b,t,k,d=field.shape
        inputs=self.history_input(field.reshape(b,t,k*d))
        posterior,hidden=self.history(inputs,memory)
        return self._state(obs,field,posterior,hidden)

    def stream_step(self,received_window,memory=None):
        """Consume only the newest encoded packet; never feed a window twice to GRU."""
        if not bool(received_window['valid'].all()):raise ValueError('Use received packets only; represent missing modalities with their masks')
        field=self._field(received_window);b,t,k,d=field.shape
        inputs=self.history_input(field[:,-1:].reshape(b,1,k*d))
        posterior,hidden=self.history(inputs,memory)
        return self._state(received_window,field,posterior,hidden)

    def forecast_states(self,state,times,time_unit,planned_actions=None):
        """Past hidden[1,B,M] + supplied commands[B,Q,7] -> causal priors[B,Q,M]."""
        b,q=times.shape
        supplied=planned_actions is not None
        if planned_actions is None:planned_actions=times.new_zeros((b,q,7))
        delta=times-torch.cat((state['last_time'],times[:,:-1]),1)
        if bool((delta<=0).any()):raise ValueError('Future rollout timestamps must be increasing after the received state')
        interval=torch.stack((delta.log1p(),time_unit[:,None].expand_as(delta).to(delta.dtype),torch.full_like(delta,float(supplied))),-1)
        inputs=self.action_edge(planned_actions)+self.clock_features(times,time_unit,state['time_origin'])+self.transition_interval(interval)
        if self.action_history:
            prior,_=self.transition(inputs,state['memory'])
        else:
            initial=state['memory'][:, :, None].expand(-1,-1,q,-1).reshape(1,b*q,self.memory_width).contiguous()
            prior,_=self.transition(inputs.reshape(b*q,1,self.width),initial)
            prior=prior.reshape(b,q,self.memory_width)
        return self.state_norm(prior.float())

    def query(self,state,times,time_unit,kind,planned_actions=None,view=None):
        """Every decoder query reads compact posterior/prior, never raw observations."""
        future=times>state['last_time']
        if bool(future.any()) and not bool(future.all()):raise ValueError('Separate reconstruction and future queries')
        shared=self.forecast_states(state,times,time_unit,planned_actions) if bool(future.all()) else state['shared'][:,None].expand(-1,times.shape[1],-1)
        b,q=times.shape
        slots=self.state_slots(shared).reshape(b*q,self.slots,self.width)
        query=self.clock_features(times,time_unit,state['time_origin'])+self.query_kind.weight[kind]
        if view is not None:query=query+self.view.weight[view]
        decoded=self.query_attention(self.query_norm(query).reshape(b*q,1,self.width),slots,slots,need_weights=False)[0].reshape(b,q,self.width)
        result=query+decoded
        if self.bypass:
            field=state['field'].flatten(1,2)
            result=result+self.query_attention(self.query_norm(query),field,field,key_padding_mask=~state['valid'].repeat_interleave(self.slots,1),need_weights=False)[0]
        return result
