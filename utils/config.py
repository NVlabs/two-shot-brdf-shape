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

from enum import Enum


class ParameterNames(Enum):
    DIFFUSE = "diffuse.png"
    SPECULAR = "specular.png"
    ROUGHNESS = "roughness.png"
    NORMAL = "normal.exr"
    DEPTH = "depth.exr"
    MASK = "mask.png"
    INPUT_1_FLASH = "cam1_flash.exr"
    INPUT_1_ENV = "cam1_env.exr"
    INPUT_1 = "cam1.exr"
    INPUT_2 = "cam2.exr"
    INPUT_1_LDR = "cam1.png"
    INPUT_2_LDR = "cam2.png"
    SGS = "sgs.npy"
    DIFFUSE_PRED = "diffuse-pred%d.png"
    SPECULAR_PRED = "specular-pred%d.png"
    ROUGHNESS_PRED = "roughness-pred%d.png"
    NORMAL_PRED = "normal-pred%d.exr"
    DEPTH_PRED = "depth-pred%d.exr"
    SGS_PRED = "sgs-pred.npy"
    RERENDER = "rerender%d.exr"


class Stages(Enum):
    INITIAL = 0
    REFINEMENT = 1
