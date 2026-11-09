"""Causal visibility, true structured parameter counts, streaming equivalence and sampling coverage."""
import numpy as np
import pytest
import json
import time

jax=pytest.importorskip("jax")
pytest.importorskip("optax")
import jax.numpy as jnp
from models.g4_shared import Spec, effective_denominators, encode, initialize, objective
from training.g4_train import Sampler
from training.g4_checkpoint_budget import admit,backlog,parameters


def signals():
    rng=np.random.default_rng(81)
    return {"a":rng.integers(0,256,(2,64,100),dtype=np.uint8),"v":rng.integers(0,256,(2,64,256),dtype=np.uint8),
            "n":rng.normal(size=(2,64,17)).astype(np.float32),"x":rng.integers(3,258,(2,64),dtype=np.uint16),
            "ok":np.ones((2,64),np.uint8),"family":np.zeros(2,np.uint8)}


@pytest.mark.parametrize("core,tied",[("signal",False),("gpt2",False),("grouped",False),("signal",True)])
def test_causal_field_has_no_hidden_input_and_carries_across_blocks(core,tied):
    spec=Spec(core,128,2,"float32",4,tied)
    p=initialize(spec,81); batch=signals(); src=jnp.zeros(2,jnp.int32)
    stats=jnp.stack([jnp.zeros(17),jnp.ones(17)])
    full,_=encode(p,spec,batch,src,stats)
    corrupt=dict(batch,v=255-batch["v"],n=-batch["n"],x=np.zeros_like(batch["x"]))
    hidden,_=encode(p,spec,corrupt,src,stats)
    np.testing.assert_array_equal(full,hidden)
    changed=batch["a"].copy(); changed[:,33:]=255-changed[:,33:]
    future,_=encode(p,spec,dict(batch,a=changed),src,stats)
    np.testing.assert_allclose(full[:,:33],future[:,:33],atol=2e-5)
    def half(start,end):
        return {k:(v[:,start:end] if k!="family" else v) for k,v in batch.items()}
    left,carry=encode(p,spec,half(0,32),src,stats)
    right,_=encode(p,spec,half(32,64),src,stats,memory=carry,offset=32)
    np.testing.assert_allclose(np.concatenate([left,right],axis=1),full,atol=3e-4,rtol=1e-4)


def test_action_is_delayed_and_dense_baselines_match_declared_parameter_budgets():
    def count(spec):return sum(x.size for x in initialize(spec,81).values())
    assert count(Spec("signal",128,2))==count(Spec("gpt2",128,2))==count(Spec("signal",128,2,tied=True))
    assert count(Spec("grouped",128,2))==count(Spec("gpt2",128,2,expansion=1))
    assert count(Spec("signal",128,2))==parameters(128,2)
    assert count(Spec("grouped",128,2))==parameters(128,2,groups=4)
    spec=Spec("signal",128,1,"float32"); p=initialize(spec,81); b=signals()
    kind=jnp.full((2,),2); stats=jnp.stack([jnp.zeros(17),jnp.ones(17)])
    z,_=encode(p,spec,b,kind,stats)
    n=b["n"].copy(); n[:,15,10:]+=5
    changed,_=encode(p,spec,dict(b,n=n),kind,stats)
    np.testing.assert_allclose(z[:,:16],changed[:,:16],atol=1e-5)
    assert np.max(np.abs(z[:,16]-changed[:,16]))>1e-3


def test_sampler_keeps_exact_exposure_and_replays_from_an_update_number():
    class Data:
        host={"family":np.repeat(np.arange(7),256),"has":np.tile([[1,1,0,1]],(7*256,1))}
        uid=np.random.default_rng(9).permutation(7*256)
        pools=[np.arange(i*256,(i+1)*256) for i in range(7)]
    data=Data(); mix=[4,3,3,1,3,2,0]
    one=Sampler(data,256,19,mix); two=Sampler(data,256,19,mix)
    many,_=one.block(0,12); ids,views=two.sample(11)
    np.testing.assert_array_equal(many[-1],ids)
    np.testing.assert_array_equal(np.bincount(data.host["family"][ids],minlength=7),np.array(mix)*16)
    assert np.all(views[:,0]!=views[:,1])
    quarter=Sampler(data,256,19,mix,.25)
    assert all(len(p)==64 for p in quarter.pools)


@pytest.mark.parametrize("policy",["cross","motion"])
def test_effective_batch_denominators_match_unsplit_loss_and_readout_gradient(policy):
    base=signals()
    batch={k:jnp.asarray(np.tile(v,(4,)+(1,)*(v.ndim-1))) for k,v in base.items()}
    batch["family"]=jnp.array([4,4,4,5,0,0,0,0])
    batch["has"]=jnp.array([[1,1,1,1]]*3+[[0,0,0,1]]+[[1,1,0,1]]*4)
    batch["cap"]=jnp.arange(8)
    views=jnp.array([[2,1,0,0],[2,1,0,1],[1,2,0,0],[3,3,0,0],[0,1,0,0],[1,0,0,0],[1,0,0,1],[0,1,0,0]])
    spec=Spec("signal",64,1,"float32"); p=initialize(spec,91)
    stats=jnp.stack([jnp.zeros(17),jnp.ones(17)])
    def score(readout,b,v,den):
        value,metrics=objective(dict(p,readout=readout),spec,b,v,stats,policy,2,den)
        return value,metrics
    fn=jax.jit(jax.value_and_grad(score,has_aux=True))
    (_,whole),dw=fn(p["readout"],batch,views,None)
    den=effective_denominators(batch,jnp.arange(8),views,2,policy)
    partial=[]
    for sl in (slice(0,4),slice(4,8)):
        (_,m),g=fn(p["readout"],jax.tree.map(lambda v:v[sl],batch),views[sl],den)
        partial.append((m,g))
    np.testing.assert_allclose((partial[0][0]+partial[1][0])/2,whole,rtol=2e-5,atol=1e-5)
    np.testing.assert_allclose((partial[0][1]+partial[1][1])/2,dw,rtol=3e-4,atol=1e-5)


def test_budget_waits_for_the_exact_full_checkpoint(tmp_path,monkeypatch):
    run=tmp_path/"g4-test"; run.mkdir()
    (run/"resume.npz").write_bytes(b"optimizer")
    (run/"complete.json").write_text(json.dumps({"step":2,"weight_file_sha256":"ab"*32}))
    spool=tmp_path/"spool"; refs=spool/"receipts"/run.name; refs.mkdir(parents=True)
    receipt=refs/"000000000002-example.json"
    receipt.write_text(json.dumps({"files":{"model.npz":{"sha256":"ab"*32}}}))
    assert backlog(tmp_path,spool)==9
    assert not admit(tmp_path,spool,time.time()+1,5,100,tmp_path/"status.json")
    receipt.write_text(json.dumps({"files":{"resume.npz":{"sha256":"ab"*32}}}))
    assert backlog(tmp_path,spool)==0
    assert admit(tmp_path,spool,time.time()+300,5,100,tmp_path/"status.json")
