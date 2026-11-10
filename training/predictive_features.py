"""Fixed observations -> frozen representation/raw/edge controls; no targets."""
import numpy as np
import torch
import torch.nn.functional as F

from training.general_readout_data import COMMON, ROBOT_CHANNELS


def supplied_text(captions, encoder):
    encoded = [encoder.encode(s, allowed_special=set(), disallowed_special=())[:64] for s in captions]
    length = max(1, max(map(len, encoded), default=1))
    tokens = np.zeros((len(captions), length), np.int64)
    present = np.zeros_like(tokens, bool)
    for i, ids in enumerate(encoded):
        tokens[i, :len(ids)] = ids
        present[i, :len(ids)] = True
    return {"text":tokens, "text_present":present}


def temporal_summary(values):
    """Received[B,T,...] -> mean/std/last/linear-slope features[B,D]."""
    clock = torch.linspace(-1, 1, values.shape[1], device=values.device, dtype=values.dtype)
    shape = [1, len(clock)]+[1]*(values.ndim-2)
    slope = (values*clock.reshape(shape)).mean(1)/clock.square().mean().clamp_min(.01)
    return torch.cat([values.mean(1).flatten(1), values.float().std(1, unbiased=False).flatten(1), values[:, -1].flatten(1), slope.flatten(1)], 1)


def frozen_features(model, obs, family, include_raw=False, stats=None):
    """Only observations enter this function, making the actor boundary explicit."""
    state=model(obs)
    result = {"shared":state["shared"].float(),"aligned":state["embedding"].float()}
    channels = COMMON if family in ("mhealth","har") else ROBOT_CHANNELS
    numeric = obs["numeric"][:, :, channels]
    present = obs["numeric_present"][:, :, channels]
    values = numeric.masked_fill(~present, 0)
    edge = model.numeric_edge(values[..., None])+model.channel.weight[channels][None,None]
    edge = edge.masked_fill(~present[..., None], 0).sum(2)/present.sum(2).clamp_min(1)[..., None]
    edge_parts = [temporal_summary(edge)]
    raw_parts = [values.flatten(1), present.float().flatten(1)]
    if family == "droid":
        images = obs["images"]
        b,t,v,c,h,w = images.shape
        learned = model.vision_edge(images.reshape(b*t*v,c,h,w)).mean((2,3)).reshape(b,t,v,-1)
        edge_parts.append(temporal_summary(learned))
        if include_raw:
            raw_parts.append(F.adaptive_avg_pool2d(images.reshape(b*t*v,c,h,w),8).reshape(b,-1))
        tokens = model.text_projection(model.text_edge(obs["text"]))
        mask = obs["text_present"]
        text = (tokens*mask[..., None]).sum(1)/mask.sum(1).clamp_min(1)[:, None]
        edge_parts.append(text); raw_parts.append(text)
    result["edge"] = torch.cat(edge_parts, 1).float()
    if include_raw:
        result["raw"] = torch.cat(raw_parts, 1).float()
        if family in ("mhealth","har"):
            # Six named3-axis vectors; missing chest gyro stays explicitly absent.
            if stats is None:raise ValueError("Physical sensor magnitudes require the training normalizer")
            mean=torch.as_tensor(stats["numeric_mean"][COMMON],device=values.device)
            scale=torch.as_tensor(stats["numeric_scale"][COMMON],device=values.device)
            physical=(values*scale+mean).masked_fill(~present,0)
            xyz = physical.reshape(len(values),values.shape[1],6,3)
            vector_present = present.reshape(len(values),values.shape[1],6,3).all(-1)
            magnitude = xyz.norm(dim=-1).masked_fill(~vector_present,0)
            spectrum = torch.fft.rfft(magnitude.float(),dim=1).abs().square()
            bands = [spectrum[:, :2].mean(1), spectrum[:, 2:5].mean(1), spectrum[:, 5:].mean(1)]
            result["invariant"] = torch.cat([magnitude.flatten(1),vector_present.float().flatten(1),temporal_summary(magnitude),*bands],1)
    return {name:value.float().cpu().numpy() for name,value in result.items()}


def fit_projection(values, width=128, seed=7391):
    """Training-only standardization and bounded randomized SVD; no test access."""
    x = np.asarray(values,np.float32)
    mean = x.mean(0); scale = x.std(0).clip(.01)
    x = (x-mean)/scale
    rank = min(width, len(x)-1, x.shape[1])
    # Bounded Gaussian range finder with reorthogonalized power iterations.
    # NumPy is already pinned for G1; no shared training dependency is changed.
    rng=np.random.default_rng(seed);dimension=min(rank+16,*x.shape)
    omega=rng.standard_normal((x.shape[1],dimension),dtype=np.float32)
    sample=x@omega
    for _ in range(3):
        left=np.linalg.qr(sample,mode="reduced")[0]
        right=np.linalg.qr(x.T@left,mode="reduced")[0]
        sample=x@right
    left=np.linalg.qr(sample,mode="reduced")[0]
    _,_,vectors=np.linalg.svd(left.T@x,full_matrices=False)
    basis = np.zeros((x.shape[1],width),np.float32);basis[:,:rank] = vectors[:rank].T
    return {"mean":mean,"scale":scale,"basis":basis,"fitted_coefficients":mean.size+scale.size+basis.size}


def project(values, fitted):
    return ((np.asarray(values,np.float32)-fitted["mean"])/fitted["scale"])@fitted["basis"]
