"""Language-initialized, causal shared state for text, images, sound and telemetry."""
import torch
from torch import nn
from torch.nn import functional as F


class Adapter(nn.Module):
    """Residual update [B,T,D] -> [B,T,D], initialized to zero."""
    def __init__(self, width, rank):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(F.gelu(self.down(x)))


class SharedUpdate(nn.Module):
    """A dense adapter or dropless learned top-one dispatch per causal position."""
    def __init__(self, width, rank, experts):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.experts = nn.ModuleList([Adapter(width,rank) for _ in range(experts)])
        self.router = nn.Linear(width,experts,bias=False) if experts>1 else None

    def forward(self, x, valid):
        z = self.norm(x)
        if self.router is None:
            return x+self.experts[0](z), x.new_zeros(()), None
        flat = z.flatten(0,1)
        prob = self.router(flat).float().softmax(-1)
        index = prob.argmax(-1)
        out = torch.zeros_like(flat)
        counts = []
        for i,expert in enumerate(self.experts):
            selected = (index==i).nonzero().flatten()
            # Empty tensors are supported. Exactly one expert receives each token.
            value = expert(flat.index_select(0,selected))
            value = value * prob[selected,i,None].to(value.dtype)
            out = out.index_copy(0,selected,value.to(out.dtype))
            counts.append(valid.flatten().index_select(0,selected).sum())
        counts = torch.stack(counts)
        weight = valid.flatten().float()
        mass = (prob*weight[:,None]).sum(0)/weight.sum().clamp_min(1)
        balance = len(self.experts) * (mass*counts/counts.sum().clamp_min(1)).sum()
        return x+out.reshape_as(x), balance, counts.detach()


def pool(z, lengths):
    """Right-padded [B,T,D] -> valid-prefix average [B,D]."""
    mask = torch.arange(z.shape[1],device=z.device)[None] < lengths[:,None]
    return (z*mask[...,None]).sum(1)/lengths.clamp_min(1)[:,None]


class JointWorld(nn.Module):
    """Typed observations -> a single ordered [B,T,D] field shared by all heads."""
    def __init__(self, parent, profile, seed):
        super().__init__()
        self.core = parent.requires_grad_(False)
        d = parent.token_embedding.embedding_dim
        self.width = d
        with torch.random.fork_rng():
            torch.manual_seed(seed+5449)
            self.edges = nn.ModuleDict({k:nn.Linear(n,d) for k,n in {"image":192,"audio":100,"numeric":128}.items()})
            self.ingress = nn.LayerNorm(d)
            self.current = nn.ModuleDict({k:nn.Linear(d,n) for k,n in {"image":192,"audio":100,"numeric":64}.items()})
            self.future = nn.ModuleDict({k:nn.Linear(d,n) for k,n in {"image":192,"audio":100,"numeric":64}.items()})
            self.align = nn.Linear(d,192,bias=False)
            self.updates = nn.ModuleList([SharedUpdate(d,profile["rank"],profile["experts"]) for _ in parent.blocks])

    def field(self, batch, lengths=None):
        if "text" in batch:
            z = self.core.token_embedding(batch["text"])
        else:
            edges = []
            for key, edge in self.edges.items():
                if key not in batch:
                    continue
                values = batch[key]
                if key=="numeric":
                    mask = batch["present"]
                    values = torch.cat([values.masked_fill(~mask,0),mask.to(values.dtype)],-1)
                edges.append(edge(values))
            z = .03*self.ingress(torch.stack(edges).sum(0)/(len(edges)**.5))
        clock = torch.arange(z.shape[1],device=z.device)
        if lengths is None:
            lengths = torch.full((len(z),),z.shape[1],device=z.device)
        valid = clock[None] < lengths[:,None]
        z = z+self.core.position_embedding(clock)
        aux, counts = z.new_zeros(()), []
        for block, update in zip(self.core.blocks,self.updates):
            z, cost, count = update(block(z), valid)
            aux = aux+cost
            if count is not None:
                counts.append(count)
        return self.core.ln_f(z), aux/len(self.updates), counts

    def accounting(self):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        dormant = sum(sum(p.numel() for e in layer.experts[1:] for p in e.parameters()) for layer in self.updates)
        return {"total":total, "trainable":train, "nonexpert_plus_one_expert_per_layer":total-dormant,
                "trainable_nonexpert_plus_one_expert_per_layer":train-dormant}

    def delta(self):
        return {n:p.detach().cpu() for n,p in self.named_parameters() if p.requires_grad}

    def apply_delta(self, values):
        expected = {n for n,p in self.named_parameters() if p.requires_grad}
        if expected != set(values):
            raise ValueError("Incomplete joint weights")
        self.load_state_dict(values,strict=False)


def contrastive(left,right,positive):
    """Symmetric multi-positive loss [B,D],[B,D],[B,B] -> scalar."""
    scores = F.normalize(left.float(),dim=-1) @ F.normalize(right.float(),dim=-1).T / .07
    def direction(s,p):
        return (s.logsumexp(1)-s.masked_fill(~p,-torch.inf).logsumexp(1)).mean()
    return .5*(direction(scores,positive)+direction(scores.T,positive.T))


def objective(model,batch):
    state, balance, counts = model.field(batch)
    metrics = {}
    if "next_text" in batch:
        logits = F.linear(state,model.core.token_embedding.weight)
        loss = F.cross_entropy(logits.flatten(0,1),batch["next_text"].flatten())
        metrics["next_token_nll"] = loss.detach()
    else:
        losses = []
        for k in model.future:
            if k not in batch:
                continue
            target = batch[k]
            mask = batch.get("present") if k=="numeric" else torch.ones_like(target,dtype=torch.bool)
            cur = F.smooth_l1_loss(model.current[k](state).float(),target,reduction="none")
            nxt = F.smooth_l1_loss(model.future[k](state[:,:-1]).float(),target[:,1:],reduction="none")
            current = (cur*mask).sum()/mask.sum().clamp_min(1)
            future = (nxt*mask[:,1:]).sum()/mask[:,1:].sum().clamp_min(1)
            losses.append(.5*current+future)
            metrics[k+"_next_huber"] = future.detach()
        loss = torch.stack(losses).mean()
        if "caption" in batch:
            lengths = batch["lengths"]
            ids = (lengths>0).nonzero().flatten()
            if len(ids)>1:
                texts, aux, _ = model.field({"text":batch["caption"][ids]},lengths[ids])
                sig = model.align(state[ids].mean(1))
                words = model.align(pool(texts,lengths[ids]))
                cap, rec = batch["caption_keys"][ids],batch["record_keys"][ids]
                term = contrastive(sig,words,(cap[:,None]==cap[None])|(rec[:,None]==rec[None]))
                loss = loss+term
                balance = (balance+aux)*.5
                metrics["paired_nce"] = term.detach()
    return loss+.01*balance, metrics, counts
