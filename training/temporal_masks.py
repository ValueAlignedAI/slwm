"""Stable integer-sample observation masks shared by acoustic experiments."""
import numpy as np


def mask_support(frames,gap_start,gap_end,rate=24000):
    """Sample gap -> observed STFT frames[F], conservatively including1024 FFT support."""
    centers=np.arange(frames)*256
    return (centers+513<=int(round(gap_start*rate)))|(centers-512>=int(round(gap_end*rate)))
