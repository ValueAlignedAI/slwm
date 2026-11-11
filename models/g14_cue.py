"""Full-utterance audio/visual slot recognition with training-only temporal supervision."""
from functools import lru_cache
import jax
import jax.numpy as jnp
import numpy as np
from data.g14_grid import SLOTS,T

def initial(seed,d=96):
    rng=np.random.default_rng(seed)
    def mat(a,b,g=1):return jnp.asarray((rng.standard_normal((a,b))*g/np.sqrt(a)).astype("f4"))
    p=dict(audio=mat(128,d),lip=mat(192,d,.7),position=mat(T,d,.12),attention=mat(d,3,.35),time_bias=jnp.zeros((T,3)))
    for i in range(3):
        p[f"local{i}"]=mat(3*d,d);p[f"gate{i}"]=mat(d,d,.6);p[f"out{i}"]=mat(d,d,.65)
    for i,v in enumerate(SLOTS):p[f"class{i}"]=mat(d,len(v),.25);p[f"bias{i}"]=jnp.zeros(len(v))
    return p

def norm(x):
    x=x-x.mean(-1,keepdims=True);return x*jax.lax.rsqrt(jnp.mean(x*x,-1,keepdims=True)+1e-4)

def forward(p,a,l,length,arm,mode="normal"):
    a=a.astype("f4");l=l.astype("f4")/255.
    if mode=="no_audio":a=jnp.zeros_like(a)
    if mode=="no_lips" or arm!="audiovisual":l=jnp.zeros_like(l)
    if mode=="shifted_lips":
        l=jnp.concatenate((jnp.zeros_like(l[:,:12]),l[:,:-12]),1)
    if mode=="noisy": # Deterministic second-stream masking is performed by the evaluator, not by class labels.
        a=a
    def edge(x):return jnp.concatenate((x,jnp.diff(x,axis=1,prepend=x[:,:1])),axis=-1)
    x=edge(a)@p["audio"]+edge(l)@p["lip"]+p["position"][None]
    for i,dilation in enumerate((1,3,6)):
        left=jnp.concatenate((jnp.zeros_like(x[:,:dilation]),x[:,:-dilation]),1)
        right=jnp.concatenate((x[:,dilation:],jnp.zeros_like(x[:,:dilation])),1)
        v=jax.nn.gelu(jnp.concatenate((left,x,right),-1)@p[f"local{i}"])
        x=norm(x+jax.nn.sigmoid(x@p[f"gate{i}"])*(v@p[f"out{i}"]))
    mask=jnp.arange(T)[None,:]<length[:,None]
    scores=x@p["attention"]+p["time_bias"][None]
    attention=jax.nn.softmax(jnp.where(mask[:,:,None],scores,-1e6),axis=1)
    slots=jnp.einsum("bts,btd->bsd",attention,x)
    logits=tuple(slots[:,i]@p[f"class{i}"]+p[f"bias{i}"] for i in range(3))
    return logits,attention

def objective(p,row,arm,t,seed):
    keys=jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed),t),4)
    a=row["a"];l=row["l"]
    # Whole-utterance gain and contiguous band masking reduce speaker/channel shortcuts.
    gain=jax.random.uniform(keys[0],(len(a),1,1),minval=-.07,maxval=.07)
    f=jax.random.randint(keys[1],(len(a),1,1),0,65);hz=jnp.arange(64)[None,None,:]
    a=jnp.clip(a+gain,0,1)*((hz<f)|(hz>=f+4))
    if arm=="audiovisual":
        a=a*(jax.random.uniform(keys[2],(len(a),1,1))>.12)
        l=l*(jax.random.uniform(keys[3],(len(a),1,1))>.12)
    logits,att=forward(p,a,l,row["length"],arm)
    loss=sum(-jnp.take_along_axis(jax.nn.log_softmax(out),row["y"][:,i,None],axis=1).mean() for i,out in enumerate(logits))
    if arm!="unaligned":
        support=row["alignment"].transpose(0,2,1);clean=support/jnp.maximum(support.sum(1,keepdims=True),1)
        loss-=.18*(clean*jnp.log(jnp.maximum(att,1e-7))).sum(1).mean()
    return loss
