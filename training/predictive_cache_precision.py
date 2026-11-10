"""Post-training cache storage erratum, registered before final observations."""
import numpy as np

def feature_storage_dtype(name):
    """Physical FFT-power invariants retain FP32; learned/raw caches keep FP16."""
    return np.float32 if name.split('__')[0]=='invariant' else np.float16
