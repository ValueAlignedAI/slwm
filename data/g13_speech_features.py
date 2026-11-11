"""Causal log-mel windows for a completed microphone word."""

import numpy as np


def features(pcm):
    data = np.zeros(16000, np.float32)
    n = min(len(pcm), 16000)
    data[:n] = pcm[:n]
    idx = np.arange(64)[:, None] * 244 + np.arange(400)
    f = np.fft.rfft(data[idx] * np.hanning(400), n=512)
    p = np.abs(f) ** 2
    hz = np.fft.rfftfreq(512, 1 / 16000)
    edges = 700 * (
        10
        ** (
            np.linspace(
                2595 * np.log10(1 + 60 / 700), 2595 * np.log10(1 + 7700 / 700), 42
            )
            / 2595
        )
        - 1
    )
    a, b, c = edges[:-2], edges[1:-1], edges[2:]
    mel = np.maximum(
        0, np.minimum((hz[:, None] - a) / (b - a), (c - hz[:, None]) / (c - b))
    )
    log = np.log10(np.maximum(p @ mel, 1e-7))
    return np.clip((log + 5.5) / 7.0, 0, 1).astype(np.float16)
