"""The original model field with a fused tied-embedding CE training head."""
import torch
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss


def hidden(model,input_ids):
    """Long[B,T] -> original final normalized field[B,T,D], before vocabulary projection."""
    positions=torch.arange(input_ids.shape[1],device=input_ids.device)[None]
    field=model.token_embedding(input_ids)+model.position_embedding(positions)
    for block in model.blocks: field=block(field)
    return model.ln_f(field)


def fused_loss(model,input_ids,targets):
    """Identical tied vocabulary weights and targets, avoiding the full [B,T,V] array."""
    field=hidden(model,input_ids)
    return LigerFusedLinearCrossEntropyLoss(accum_dtype=torch.float32)(model.token_embedding.weight,field.reshape(-1,field.shape[-1]),targets.reshape(-1))
