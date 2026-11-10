"""Export the four synchronized model inputs for a previously reported synthetic mission."""
import json
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from models.g9_tabletop import initial,camera,observation,advance

def main(trace,out):
    d=json.loads(trace.read_text());state=initial(jnp.array([d["index"]]),d["mission_seed"])
    first=state["goal"];board=camera(state["stations"],state["tint"]);controls=jnp.asarray(d["actions"]["hybrid"],jnp.int32)
    def step(st,item):
        t,act=item;st={**st,"goal":(first+t//64)%3}
        obs=observation(st,board,t);obs["plan"]=act[None]
        return advance(st,act[None],t,stall=True),obs
    _,data=jax.lax.scan(step,state,(jnp.arange(len(controls)),controls))
    np.savez_compressed(out,**{k:np.asarray(v[:,0]) for k,v in data.items()})

if __name__=="__main__":main(Path(sys.argv[1]),Path(sys.argv[2]))
