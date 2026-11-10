"""Separate simulator feasibility diagnostic after the first locked terminal policies were scored."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from training.g7_reacher import environment,setup,actions,advance,H

ENV=None


def evaluate(job):
    import mujoco as mj
    global ENV
    if ENV is None:ENV=environment()
    seed,which,depth=job
    e=ENV;rng,goals=setup(e,seed);goal=goals[which];initial=rng.uniform(-.8,.8,2).astype(np.float32)
    advance(e,initial,False)
    state_type=mj.mjtState.mjSTATE_INTEGRATION
    def snapshot():
        s=np.empty(mj.mj_stateSize(e.model,state_type));mj.mj_getState(e.model,e.data,s,state_type);return s
    def restore(s):
        mj.mj_setState(e.model,e.data,s,state_type);mj.mj_forward(e.model,e.data)
    def bank(t):return actions(np.random.default_rng([seed,t,8119]))
    distances=[]
    for t in range(24):
        proposals=bank(t);next_proposals=bank(t+1);base=snapshot();values=[]
        for candidate in proposals:
            restore(base);tip=advance(e,candidate,False)
            immediate=np.linalg.norm(tip-goal)
            if depth==1 or t==23:values.append(immediate);continue
            s=snapshot();future=[]
            for other in next_proposals:
                restore(s);future.append(np.linalg.norm(advance(e,other,False)-goal))
            values.append(.2*immediate+min(future))
        restore(base);p=advance(e,proposals[np.argmin(values)],False);distances.append(float(np.linalg.norm(p-goal)*100))
    return seed,which,depth,distances[-1],min(distances)


def main(a):
    import json
    jobs=[(s,i,d) for s in list(range(860000,860064))+list(range(880000,880064)) for i in (0,1) for d in (1,2)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(evaluate,jobs,chunksize=2))
    summary={}
    for start in (860000,880000):
        for d in (1,2):
            rows=[r for r in results if start<=r[0]<start+64 and r[2]==d]
            dist=np.array([r[3] for r in rows]);best=np.array([r[4] for r in rows])
            summary[f"{start}-lookahead{d}"]={"starts":64,"episodes":len(rows),"final_success":float((dist<3).mean()),"final_cm":float(dist.mean()),"ever_success":float((best<3).mean())}
    doc={"design":"Post hoc mechanism diagnostic. Exact simulator and target; depth two sees the next exogenous proposal bank. Both held and newly reserved start ranges.","rows":results,"summary":summary}
    tmp=a.out.with_suffix(".partial");tmp.write_text(json.dumps(doc,sort_keys=True)+"\n");tmp.replace(a.out)
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);p.add_argument("--workers",type=int,default=4)
    main(p.parse_args())
