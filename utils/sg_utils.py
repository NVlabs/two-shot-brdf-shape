# -----------------------------------------------------------------------
# Copyright (c) 2020, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
#
# Official Implementation of the CVPR2020 Paper
# Two-shot Spatially-varying BRDF and Shape Estimation
# Mark Boss, Varun Jampani, Kihwan Kim, Hendrik P. A. Lensch, Jan Kautz
# -----------------------------------------------------------------------

import os
from typing import List, Union

import cv2
import numpy as np
import pyexr
from tqdm import tqdm


def magnitude(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), 1e-12))


def dot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sum(x * y, axis=-1, keepdims=True)


def normalize(x: np.ndarray) -> np.ndarray:
    return x / magnitude(x)


def setup_axis_sharpness(num_sgs) -> np.ndarray:
    axis = []
    inc = np.pi * (3.0 - np.sqrt(5.0))
    off = 2.0 / num_sgs
    for k in range(num_sgs):
        y = k * off - 1.0 + (off / 2.0)
        r = np.sqrt(1.0 - y * y)
        phi = k * inc
        axis.append(normalize(np.array([np.cos(phi) * r, np.sin(phi) * r, y])))

    minDp = 1.0
    for a in axis:
        h = normalize(a + axis[0])
        minDp = min(minDp, dot(h, axis[0]))

    sharpness = (np.log(0.65) * num_sgs) / (minDp - 1.0)

    axis = np.stack(axis, 0)  # Shape: num_sgs, 3
    sharpnessNp = np.ones((num_sgs, 1)) * sharpness
    return np.concatenate([axis, sharpnessNp], -1)


def _evaluate(sg: np.ndarray, d: np.ndarray) -> np.ndarray:
    assert sg.shape[-1] == 7
    assert d.shape[-1] == 3
    assert len(sg.shape) == len(d.shape)

    s_amplitude = sg[..., 0:3]
    s_axis = sg[..., 3:6]
    s_sharpness = sg[..., 6:7]

    cosAngle = dot(d, s_axis)
    return s_amplitude * np.exp(s_sharpness * (cosAngle - 1.0))


def visualize_fit(
    output_paths: Union[List[str], str],
    shape,
    sgs: np.ndarray,
    scale_down_by: float = 1,
):
    assert len(sgs.shape) == 2
    assert sgs.shape[-1] == 7
    assert len(shape) >= 2

    shape = (
        shape[0],
        shape[1],
        3,
    )

    output = np.zeros(shape)

    us, vs = np.meshgrid(np.linspace(0, 1, shape[1]), np.linspace(0, 1, shape[0]))  # OK

    uvs = np.dstack([us, vs])
    # q   f

    theta = 2.0 * np.pi * uvs[..., 0] - (np.pi / 2)
    phi = np.pi * uvs[..., 1]

    d = np.dstack(
        [np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)]
    )

    for i in range(sgs.shape[0]):
        sg = sgs[i : i + 1]
        output = output + _evaluate(sg, d)

    if isinstance(output_paths, str):
        output_paths = [output_paths]

    for path in output_paths:
        curOut = output.copy()

        is_output_exr = os.path.splitext(path)[1] == ".exr"
        is_output_hdr = os.path.splitext(path)[1] == ".hdr"

        if is_output_exr:
            pyexr.write(path, curOut)
            continue

        if is_output_hdr:
            output_proc = curOut.astype(np.float32)
        else:
            output_proc = (EU.linearTosRGB(np.clip(curOut, 0.0, 1.0)) * 255).astype(
                "uint8"
            )

        cv2.imwrite(path, cv2.cvtColor(output_proc, cv2.COLOR_RGB2BGR))

    return output
