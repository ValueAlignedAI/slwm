"""A compact spectrogram codec producing a reusable direction field and calibrated word proposals."""
import jax
import jax.numpy as jnp
import numpy as np


def init(seed):
    rng=np.random.default_rng(seed);p={}
    for i,(channels,out) in enumerate(((1,24),(24,40),(40,64))):
        k=5 if i==0 else 3
        p["c"+str(i)]=jnp.asarray((rng.standard_normal((k,k,channels,out))*np.sqrt(2/(k*k*channels))).astype("f4"))
        p["b"+str(i)]=jnp.zeros(out)
    p["field"]=jnp.asarray((rng.standard_normal((320,128))/np.sqrt(320)).astype("f4"))
    p["word"]=jnp.asarray((rng.standard_normal((128,4))*.03).astype("f4"));p["bias"]=jnp.zeros(4)
    return p


def predict(p,x):
    x=x[:,:,:,None].astype(jnp.float32)
    for i in range(3):
        x=jax.lax.conv_general_dilated(x,p["c"+str(i)],(2,2),"SAME",dimension_numbers=("NHWC","HWIO","NHWC"))+p["b"+str(i)]
        mu=x.mean((1,2),keepdims=True);var=((x-mu)**2).mean((1,2),keepdims=True)
        x=jax.nn.gelu((x-mu)*jax.lax.rsqrt(var+1e-4))
    z=jnp.tanh(x.mean(1).reshape(len(x),320)@p["field"])
    return z,z@p["word"]+p["bias"]


def augment(x,key):
    keys=jax.random.split(key,4);n=len(x)
    roll=jax.random.randint(keys[0],(n,1),-5,6);indices=jnp.arange(x.shape[1])[None,:]-roll
    shifted=jnp.take_along_axis(x,jnp.clip(indices,0,x.shape[1]-1)[:,:,None],axis=1)*((indices>=0)&(indices<x.shape[1]))[:,:,None]
    t=jax.random.randint(keys[1],(n,1,1),0,65);f=jax.random.randint(keys[2],(n,1,1),0,41)
    clock=jnp.arange(64)[None,:,None];hz=jnp.arange(40)[None,None,:]
    keep=(~((clock>=t)&(clock<t+6)))&(~((hz>=f)&(hz<f+3)))
    return jnp.clip(shifted*jnp.asarray(keep)+jax.random.uniform(keys[3],(n,1,1),minval=-.055,maxval=.055),0,1)
