"""Direct predictions from the mandatory G2 state; no raw residual shortcuts."""
import math
import torch
import torch.nn.functional as F

from models.predictive_world import join_observations
from training.general_losses import masked_mean,pair_loss,ctc_loss,without


def spread(values):
    """FP32 variance and off-diagonal covariance SUM/D, not diluted mean/D²."""
    with torch.autocast(values.device.type,enabled=False):
        x=values.float();x=x-x.mean(0)
        variance=F.relu(.5-torch.sqrt(x.square().mean(0)+1e-4)).mean()
        covariance=x.T@x/max(1,len(x)-1)
        off=covariance-torch.diag_embed(covariance.diagonal())
        return variance,off.square().sum()/x.shape[1]


def participation(values):
    with torch.autocast(values.device.type,enabled=False):
        x=values.detach().float();x=x-x.mean(0);gram=x@x.T
        return float(gram.diagonal().sum().square()/gram.square().sum().clamp_min(1e-20))


def loss(model,teacher,batch,run,cfg,position,validation=False):
    obs=batch['obs'];parts={};metrics={};weights=cfg['loss_weights'];family=batch['family']
    # No target/reference is passed to the online observation encoder.
    state=model(obs);query=model.query(state,batch['query_time'],obs['time_unit'],0 if 'mel_target' in batch else 2)
    if 'mel_target' in batch:
        prediction=model.propose(query,'audio')
        error=F.smooth_l1_loss(prediction,batch['mel_target'],reduction='none')
        parts['reconstruction']=.9*masked_mean(error,batch['mel_hidden'])+.1*error.mean()
        metrics['audio_reconstruction']=float(error.detach().mean())
        if 'phones' in batch:
            phone=model.propose(query,'phones');valid=batch['phones']>=0
            parts['phones']=masked_mean(F.cross_entropy(phone.reshape(-1,model.phone_count),batch['phones'].clamp_min(0).reshape(-1),reduction='none').reshape_as(valid),valid)
            parts['ctc']=ctc_loss(model.propose(query,'characters'),batch['reference'],batch['lexical_eligible'])
    if 'numeric_target' in batch:
        prediction=model.propose(query,'numeric');error=F.smooth_l1_loss(prediction,batch['numeric_target'],reduction='none')
        parts['reconstruction']=.9*masked_mean(error,batch['numeric_hidden'])+.1*masked_mean(error,batch['numeric_target_present'])
    if 'picture_target' in batch:
        picture=model.query(state,batch['query_time'],obs['time_unit'],1,view=batch.get('hidden_view',0))
        parts['picture']=F.mse_loss(model.propose(picture,'image'),batch['picture_target'])
    var,cov=spread(state['shared']);parts['state_variance']=var;parts['state_covariance']=cov
    metrics['state_participation']=participation(state['shared'])
    if 'companion' in batch:
        signal=model(without(obs,'text')) if 'text' in obs else state
        text=model(batch['companion']);eligible=torch.tensor([bool(t.strip()) for t in batch['captions']],device=query.device)
        if family=='ami':eligible=eligible&batch['correspondence_eligible']
        parts['alignment']=pair_loss(signal['embedding'],text['embedding'],batch.get('alignment_ids',batch['ids']),batch['caption_hashes'],eligible)
        if int(eligible.sum())>1:
            a=signal['embedding'][eligible];b=text['embedding'][eligible]
            va,ca=spread(a*math.sqrt(a.shape[-1]));vb,cb=spread(b*math.sqrt(b.shape[-1]))
            parts['alignment_variance']=(va+vb)*.5;parts['alignment_covariance']=(ca+cb)*.5
            parts['alignment_mean']=(a.mean(0).square().sum()+b.mean(0).square().sum())*.5
            identifiers=batch.get('alignment_ids',batch['ids']);indices=eligible.nonzero().flatten().tolist()
            positive=torch.tensor([[identifiers[i]==identifiers[j] or batch['caption_hashes'][i]==batch['caption_hashes'][j] for j in indices] for i in indices],device=query.device)
            chosen=(a@b.T).argmax(1)
            metrics['paired_hit1']=float(positive[torch.arange(len(a),device=query.device),chosen].float().mean())
            metrics['paired_uniform_chance']=float(positive.float().mean())
            metrics['alignment_participation']=participation(a)
    if family=='ami':
        audio=model(without(obs,'images','text'));video=model(without(obs,'audio','text'))
        wrong=model({**without(obs,'audio','text'),'images':batch['wrong_images']})
        positive=model.correspondence_logits(audio['embedding'],video['embedding'])
        negative=model.correspondence_logits(audio['embedding'],wrong['embedding'])
        error=(F.binary_cross_entropy_with_logits(positive,torch.ones_like(positive),reduction='none')+F.binary_cross_entropy_with_logits(negative,torch.zeros_like(negative),reduction='none'))*.5
        parts['correspondence']=masked_mean(error,batch['correspondence_eligible'])
    if 'future_obs' in batch:
        times=batch['future_time'];actions=batch.get('planned_actions')
        if family=='droid' and not validation and position%10==0:
            actions=None;metrics['plan_dropped']=1.
        future=model.query(state,times,obs['time_unit'],0 if 'future_mel' in batch else 2,actions)
        if 'future_mel' in batch:
            error=(model.propose(future,'audio')-batch['future_mel']).square()
            parts['future']=error.mean();risk=error.detach().mean(-1)
        else:
            error=(model.propose(future,'numeric')-batch['future_numeric']).square();mask=batch['future_numeric_present']
            parts['future']=masked_mean(error,mask);risk=(error.detach()*mask).sum(-1)/mask.sum(-1).clamp_min(1)
        if 'future_picture' in batch:
            picture=model.query(state,times,obs['time_unit'],1,actions,view=batch['hidden_view'])
            parts['future_picture']=F.mse_loss(model.propose(picture,'image'),batch['future_picture'])
        parts['risk']=F.smooth_l1_loss(F.softplus(model.risk(future).float()[...,0]),risk)
        with torch.no_grad():
            # Causal teacher continuation: targets include received past and
            # genuinely observed future; target states never enter online forward.
            target=teacher(join_observations(obs,batch['future_obs']))['posterior'][:,-times.shape[1]:]
        prior=model.forecast_states(state,times,obs['time_unit'],actions)
        parts['latent']=F.smooth_l1_loss(prior,target)
    objective=sum(weights[name]*value for name,value in parts.items())
    metrics.update({name:float(value.detach()) for name,value in parts.items()})
    metrics['objective']=float(objective.detach())
    metrics['selection']=sum(weights[name]*float(value.detach()) for name,value in parts.items() if name not in ('latent','risk','state_variance','state_covariance','alignment_variance','alignment_covariance','alignment_mean'))
    return objective,metrics
