"""Verify the batched physical inference entry point on development fixtures outside G8 confirmation."""
from pathlib import Path
import sys
import jax
import numpy as np
from training.g5_batches import JointData
from models.g5_residual import world
from training.g8_models import resolve,load_model,G8_DATA
from training.g8_pushing import predictions,Geometry,observed_batch
from training.g8_live import ScenePool,rollout
from scripts.train.g6_artifacts import roots
from training.g7_units import put


def main():
    sys.path.append(str(Path.home()/"dev/slwm-g5/.sim/lib/python3.12/site-packages"))
    old=G8_DATA.parent/"g5";record=resolve("hybrid",511,"parent");spec,p,cal,stats=load_model(record)
    original=JointData(old,"dev",verify=False,stats=stats);sim=original.packs[2];indices=np.repeat(np.arange(4),3);branches=np.tile(np.arange(3),4)
    b=original.gather(np.full(12,8),indices,branches);b["source"][:,[1,2]]=1
    observed=observed_batch(sim["n"][:4],sim["v"][:4],sim["plan"][:4],"joint")
    for name in b:
        if name!="outcome":np.testing.assert_array_equal(observed[name],b[name])
    original_y=np.asarray(jax.jit(lambda params,prior,q:world(params,spec,q,q["source"],prior)[2][:,-1]*64)(p,cal,b)).reshape(4,3,2)
    actual=predictions(spec,p,cal,sim["n"][:4],sim["v"][:4],sim["plan"][:4])
    np.testing.assert_allclose(actual,original_y,rtol=0,atol=1e-5)
    output=G8_DATA/"development-smoke";output.mkdir(exist_ok=True)
    prior=Geometry(old)
    for task in ("pusht","puck_barrier"):
        pool=ScenePool(task,1155000,1155004,workers=2)
        try:
            rollout(pool,"hybrid-development",spec,p,cal,prior,output,record["weights"])
            # Raw and trained controllers receive the same initial state and goal.
            rollout(pool,"first_proposal",None,None,None,prior,output,"fixed-first_proposal")
        finally:pool.close()
        with np.load(output/"live"/task/"hybrid-development/trajectory.npz") as a,np.load(output/"live"/task/"first_proposal/trajectory.npz") as b:
            np.testing.assert_array_equal(a["goals"],b["goals"]);np.testing.assert_array_equal(a["object_xy"][:,0],b["object_xy"][:,0])
            assert a["executed"].shape==(4,24,16,2) and np.isfinite(a["object_xy"]).all()
    put(output/"verified.json",{"G7_batched_forecast_max_abs_difference_px":float(np.abs(actual-original_y).max()),"excluded_from_confirmation":[1155000,1155004],"devices":str(jax.devices()),"two_live_tasks_completed":True})
    print("Normal entry points, phase and goal pairing verified on development starts",flush=True)


if __name__=="__main__":main()
