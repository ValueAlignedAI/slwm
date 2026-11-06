"""G1 observation-only forward paths and explicit predictive/alignment losses."""
import math
import numpy as np
import torch
import torch.nn.functional as F


def masked_mean(values,mask):
    weights=mask.to(values.dtype)
    return (values*weights).sum()/weights.sum().clamp_min(1)


def pair_loss(a,b,ids,captions,eligible=None,shuffle=False):
    """Symmetric multi-positive contrastive loss; duplicates are not negatives."""
    device=a.device;n=len(a)
    positive=torch.tensor([[ids[i]==ids[j] or captions[i]==captions[j] for j in range(n)] for i in range(n)],device=device,dtype=torch.bool)
    valid=torch.ones(n,device=device,dtype=torch.bool) if eligible is None else eligible
    indices=valid.nonzero().flatten()
    if len(indices)<2:return a.sum()*0
    a=a[indices];b=b[indices];positive=positive[indices][:,indices]
    if shuffle:
        # A nonzero circular shift is a fixed, reproducible wrong-pair control.
        b=b.roll(1,0)
    scores=a@b.T/.1
    forward=torch.logsumexp(scores,1)-torch.logsumexp(scores.masked_fill(~positive,-torch.inf),1)
    reverse=torch.logsumexp(scores,0)-torch.logsumexp(scores.masked_fill(~positive,-torch.inf),0)
    return (forward.mean()+reverse.mean())*.5


def without(obs,*kinds):
    result={**obs}
    for kind in kinds:
        for key in {"audio":("audio","audio_present"),"images":("images","image_present","pixel_present","patch_present","image_weight"),"text":("text","text_present"),"numeric":("numeric","numeric_present")}[kind]:result.pop(key,None)
    return result


def ctc_loss(logits,reference,eligible):
    alphabet=" 'abcdefghijklmnopqrstuvwxyz";targets=[];lengths=[];indices=[];steps=logits.shape[1]
    eligible=eligible.detach().cpu().tolist() if isinstance(eligible,torch.Tensor) else eligible
    for i,text in enumerate(reference):
        ids=[alphabet.index(c)+1 for c in text.lower() if c in alphabet]
        required=len(ids)+sum(a==b for a,b in zip(ids,ids[1:]))
        if bool(eligible[i]) and ids and required<=steps:targets+=ids;lengths.append(len(ids));indices.append(i)
    if not indices:return logits.sum()*0
    selected=logits[indices].log_softmax(-1).transpose(0,1).contiguous()
    return F.ctc_loss(selected,torch.tensor(targets,device=logits.device,dtype=torch.long),torch.full((len(indices),),steps,dtype=torch.long),torch.tensor(lengths,dtype=torch.long),blank=0,reduction="mean",zero_infinity=True)


def batch_loss(model,teacher,batch,run,cfg,position,validation=False):
    """Return differentiable scalar and detached metrics. Never pass targets to online forward."""
    family=batch["family"];obs=batch["obs"];weights=cfg["loss_weights"];parts={};auxiliary={};zero=next(model.parameters()).sum()*0
    state=None
    if family=="ami":
        a=model(without(obs,"images","text"));v=model(without(obs,"audio","text"));wrong_obs={**without(obs,"audio","text"),"images":batch["wrong_images"]};wrong=model(wrong_obs)
        good_logits=model.correspondence_logits(a["sequence_embedding"],v["sequence_embedding"]).mean(1);bad_logits=model.correspondence_logits(a["sequence_embedding"],wrong["sequence_embedding"]).mean(1);eligible=batch["correspondence_eligible"]
        parts["correspondence"]=masked_mean(F.binary_cross_entropy_with_logits(good_logits,torch.ones_like(good_logits),reduction="none")+F.binary_cross_entropy_with_logits(bad_logits,torch.zeros_like(bad_logits),reduction="none"),eligible)*.5
        parts["av_alignment"]=pair_loss(a["embedding"],v["embedding"],batch["alignment_ids"],batch["caption_hashes"],eligible,run.get("shuffle_alignment",False) and not validation)
        if "distillation" in weights:
            teacher_audio=without(obs,"images","text");teacher_audio["audio"]=batch["mel_target"].reshape(*obs["audio"].shape);teacher_audio["audio_present"]=torch.ones_like(obs["audio_present"])
            with torch.no_grad():
                target_state=teacher(teacher_audio);target_query=teacher.query(target_state,batch["query_time"],obs["time_unit"],0);probability=teacher.propose(target_query,"phones").softmax(-1);confidence,unit=probability.max(-1)
            video_query=model.query(v,batch["query_time"],obs["time_unit"],0);prediction=model.propose(video_query,"phones");keep=(confidence>=cfg["phone_teacher_min_confidence"])&eligible[:,None,None]
            error=F.cross_entropy(prediction.reshape(-1,model.phone_count),unit.reshape(-1),reduction="none").reshape_as(keep);parts["distillation"]=masked_mean(error,keep);auxiliary["phone_teacher_coverage"]=float(keep.float().mean())
        rng=np.random.default_rng(7919+position);conditions=rng.choice(3,len(good_logits),p=[.5,.25,.25]);conditions=torch.tensor(conditions,device=good_logits.device)
        # Offset labels never enter observations. The correspondence model sees
        # claimed-aligned clocks for both actual and mistimed video.
        chosen=torch.where((conditions==1)[:,None,None,None,None,None],batch["wrong_images"],obs["images"])
        confidence=torch.where(conditions==1,bad_logits,good_logits).sigmoid().detach()
        fused={**obs,"images":chosen,"image_present":obs["image_present"]&(conditions!=2)[:,None,None],"image_weight":confidence[:,None,None].expand_as(obs["image_present"])}
        state=model(fused);signal=a
    else:
        state=model(obs);signal=state
    query=model.query(state,batch["query_time"],obs["time_unit"],0 if "mel_target" in batch else 2)
    if "mel_target" in batch:
        prediction=batch["mel_baseline"]+4*model.propose(query,"audio").tanh();parts["masked"]=masked_mean(F.smooth_l1_loss(prediction,batch["mel_target"],reduction="none"),batch["mel_hidden"])
        if "phones" in batch:
            phone=model.propose(query,"phones");valid=batch["phones"]>=0
            parts["phones"]=masked_mean(F.cross_entropy(phone.reshape(-1,model.phone_count),batch["phones"].clamp_min(0).reshape(-1),reduction="none").reshape_as(valid),valid)
            parts["ctc"]=ctc_loss(model.propose(query,"characters"),batch["reference"],batch["lexical_eligible"])
    if "numeric_target" in batch:
        prediction=batch["numeric_baseline"]+model.propose(query,"numeric");parts["numeric_masked"]=masked_mean(F.smooth_l1_loss(prediction,batch["numeric_target"],reduction="none"),batch["numeric_hidden"])
    if "picture_target" in batch:
        picture_query=model.query(state,batch["query_time"],obs["time_unit"],1,view=batch.get("hidden_view",0));base=batch.get("picture_baseline",torch.zeros_like(batch["picture_target"]))
        prediction=base+2*model.propose(picture_query,"image").tanh()
        parts["picture_masked"]=F.mse_loss(prediction,batch["picture_target"]) if batch.get("picture_hidden",True) else zero
    if "companion" in batch:
        # A supplied goal/transcript must not leak into its own alignment anchor.
        if "text" in obs and family!="ami":
            supplied=obs["text_present"].any(-1);indices=supplied.nonzero().flatten()
            if len(indices):
                sub={k:v[indices] for k,v in without(obs,"text").items()};encoded=model(sub)
                signal={**state,"embedding":state["embedding"].index_copy(0,indices,encoded["embedding"])}
        text=model(batch["companion"]);eligible=torch.tensor([bool(t.strip()) for t in batch["captions"]],device=query.device)
        if family=="ami":eligible=eligible&batch["correspondence_eligible"]
        parts["alignment"]=pair_loss(signal["embedding"],text["embedding"],batch.get("alignment_ids",batch["ids"]),batch["caption_hashes"],eligible,run.get("shuffle_alignment",False) and not validation)
    parts["variance"]=F.relu(.5-state["raw_embedding"].float().std(0,unbiased=False)).mean()
    if "future_obs" in batch and (not run.get("no_future_loss") or validation):
        future=model.query(state,batch["future_time"],obs["time_unit"],0 if "future_mel" in batch else 2,batch.get("planned_actions"))
        if "future_mel" in batch:
            # Persistence uses the final actually received fine mel packet.
            base=batch["mel_baseline"][:,-1:];prediction=base+model.propose(future,"audio");error=(prediction-batch["future_mel"]).square();parts["future"]=error.mean();risk_target=error.detach().mean(-1)
        else:
            base=batch["numeric_baseline"][:,-1:];prediction=base+model.propose(future,"numeric");error=(prediction-batch["future_numeric"]).square();mask=batch["future_numeric_present"];parts["future"]=masked_mean(error,mask);risk_target=(error.detach()*mask).sum(-1)/mask.sum(-1).clamp_min(1)
        if "future_picture" in batch:
            picture=model.query(state,batch["future_time"],obs["time_unit"],1,batch.get("planned_actions"),view=batch["hidden_view"]);base=batch["picture_baseline"][:,-1:];prediction=base+2*model.propose(picture,"image").tanh()
            parts["future_picture"]=F.mse_loss(prediction,batch["future_picture"])
        parts["risk"]=F.smooth_l1_loss(F.softplus(model.risk(future).float()[...,0]),risk_target)
        with torch.no_grad():target=teacher(batch["future_obs"])["sequence_embedding"]
        latent=F.normalize(model.future_embedding(future).float(),dim=-1);parts["latent"]=(1-(latent*target).sum(-1)).mean()
    coefficient={"numeric_masked":"masked","picture_masked":"masked","future_picture":"future","av_alignment":"alignment"}
    total=sum(weights[coefficient.get(name,name)]*value for name,value in parts.items())
    metrics={name:float(value.detach()) for name,value in parts.items()};metrics.update(auxiliary);metrics["objective"]=float(total.detach())
    # Selection excludes the moving-target/variance/risk terms. All included
    # metrics have fixed data targets, not a drifting EMA-label scale.
    metrics["selection"]=sum(float(value.detach())*weights[coefficient.get(name,name)] for name,value in parts.items() if name not in ("latent","variance","risk","distillation"))
    return total,metrics
