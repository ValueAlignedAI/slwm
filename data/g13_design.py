"""Causal signal features shared by training and portable action readouts."""

import numpy as np


def raw_x(r):
    obj = r["geo"].astype("f4")
    pixel = r["px"][:, -1].reshape(-1, 10, 16, 3).astype("f4")
    image = pixel.reshape(-1, 2, 5, 4, 4, 3).mean((2, 4)).reshape(len(pixel), -1) / 255
    return np.concatenate(
        (
            image,
            r["goal"].astype("f4") / 127,
            r["bind"].reshape(len(pixel), -1),
            r["kin"].reshape(len(pixel), -1),
            obj[:, -1].reshape(len(pixel), -1),
            (obj[:, -1] - obj[:, 0]).reshape(len(pixel), -1),
        ),
        1,
    )


def action_design(r):
    hist = r["kin"][:, :, 2:].reshape(len(r["kin"]), -1) / 0.05
    return np.concatenate(
        [
            raw_x(r),
            hist,
            np.tanh(hist),
            np.linalg.norm(r["kin"][:, :, 2:4], axis=2) / 0.05,
        ],
        1,
    )
