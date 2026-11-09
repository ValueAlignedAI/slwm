import numpy as np
from training.predictive_cache_precision import feature_storage_dtype


def test_large_physical_power_is_preserved_without_changing_learned_cache_precision():
    values=np.array([[70000.,200000.,3.]],np.float32)
    stored=values.astype(feature_storage_dtype('invariant'))
    assert stored.dtype==np.float32 and np.array_equal(stored,values)
    assert np.isfinite(stored).all()
    shared=np.array([[.12,2.,-1.]],np.float32)
    assert np.array_equal(shared.astype(feature_storage_dtype('shared')),shared.astype(np.float16))
