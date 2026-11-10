"""Regression checks for causal masks, fair history and action-history identification."""
import numpy as np
import pytest

jax=pytest.importorskip("jax")
import jax.numpy as jnp
from models.g9_shared import Spec,encode,parameters
from models.g9_tabletop import sim_batch
from evals.g9_live import live,paired_histories
from scripts.train.g9_horizon import compatibility


def test_hidden_sensor_and_future_do_not_enter_shared_state():
    spec=Spec("gpt2",128,1,128,"gelu","float32")
    p=parameters(spec,21)
    r=sim_batch(jnp.arange(3),24)
    keep=jnp.ones((3,4),bool).at[:,1].set(False)
    z,_=encode(p,spec,r,keep)
    altered={**r,"v":255-r["v"],"a":r["a"].at[:,17:].set(255-r["a"][:,17:])}
    changed,_=encode(p,spec,altered,keep)
    np.testing.assert_allclose(np.asarray(z[:,:17]),np.asarray(changed[:,:17]),atol=1e-6)
    assert z.shape==(3,32,128)


def test_history_is_identifiable_only_with_the_issued_action():
    spec=Spec("hybrid",128,1,128,"gelu","float32")
    out=paired_histories(parameters(spec,29),spec,b=8)
    assert bool(out["sensor_identical"]) and float(out["opposite"])==1
    assert np.max(out["reset"])<=.5


def test_generalized_attention_reproduces_native_window():
    compatibility()


def test_raw_controller_works_on_changed_dynamics():
    spec=Spec("gpt2",128,1,128,"gelu","float32")
    signal=live(None,spec,"raw",b=8,steps=64,seed=19)
    assert signal[:,:,4].all()
    assert signal[:,:,0].any(0).all()


def test_portable_per_channel_quantization_round_trip(tmp_path):
    from models.g9_portable import pack,load
    rng=np.random.default_rng(55)
    matrix=rng.normal(size=(512,128)).astype(np.float32)
    source={"projection":matrix,"bias":np.zeros(128,np.float32)}
    file=tmp_path/"field.npz";pack(file,source,{"field":128},"parent")
    manifest,decoded=load(file)
    assert manifest["parent_sha256"]=="parent"
    error=np.max(abs(decoded["projection"]-matrix),axis=0)
    assert np.all(error<=np.max(abs(matrix),axis=0)/14+1e-6)
    np.testing.assert_array_equal(decoded["bias"],source["bias"])
