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
import random
import sys
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np
from skimage.morphology import disk, erosion
from tensorpack import *
from tensorpack.dataflow import BatchData, PrefetchDataZMQ, ProxyDataFlow, RNGDataFlow

from utils.config import ParameterNames, Stages
from utils.dataflow_utils import compute_auto_exp, read_image, sRGBToLinear


class ConfigurableDataflow(RNGDataFlow):
    def __init__(
        self,
        path: str,
        parameters: List[ParameterNames],
        processing_step: int = 0,
        add_path: bool = False,
    ):
        print("Preparing dataset ...")
        parents = [os.path.join(path, p) for p in os.listdir(path)]
        print("...")
        data = [
            [
                os.path.join(p, d)
                for d in os.listdir(p)
                if os.path.isdir(os.path.join(p, d))
            ]
            for p in parents
            if os.path.isdir(p)
        ]
        print("...")
        self.data = np.array([item for sublist in data for item in sublist])
        print("... Dataset prepared!")

        if ParameterNames.INPUT_1_FLASH in parameters:
            assert ParameterNames.INPUT_1_ENV in parameters
        if ParameterNames.INPUT_1_ENV in parameters:
            assert ParameterNames.INPUT_1_FLASH in parameters

        self.parameters = parameters
        self.processing_step = processing_step
        self.add_path = add_path

        super(ConfigurableDataflow, self).__init__()

    def reset_state(self):
        RNGDataFlow.reset_state(self)

    def __len__(self):
        return self.data.size

    @staticmethod
    def _parameter_to_image(
        path: str, parameter: ParameterNames, step: int = 0
    ) -> np.ndarray:
        try:
            pname = parameter.value % step
        except TypeError:  # String does not contain formatting
            pname = parameter.value

        # Sometimes the enum value is a tuple. ¯\_(ツ)_/¯
        if isinstance(pname, tuple):
            pname = pname[0]

        full_path = os.path.join(path, pname)

        # Parameters:
        # RGB: diffuse, specular, normal, input_1, input_2, rerender
        # 1CH: depth, roughness, mask
        # NPA: sgs
        if parameter == ParameterNames.SGS or parameter == ParameterNames.SGS_PRED:
            return np.load(full_path)
        elif (
            parameter == ParameterNames.DEPTH
            or parameter == ParameterNames.DEPTH_PRED
            or parameter == ParameterNames.ROUGHNESS
            or parameter == ParameterNames.ROUGHNESS_PRED
        ):  # Grayscale
            return read_image(full_path, True)
        elif parameter == ParameterNames.MASK:  # Mask apply erosion etc.
            mask = read_image(full_path, True)
            mask[mask < 0.5] = 0.0
            mask[mask >= 0.5] = 1.0
            mask = erosion(
                mask[..., 0], disk(3)
            )  # Apply a erosion (channels need to be removed)
            return np.expand_dims(mask, -1)  # And added back
        else:  # RGB
            return read_image(full_path, False)

    @staticmethod
    def process_input_image(img: np.ndarray):
        img, _ = compute_auto_exp(img)
        return np.nan_to_num(img)

    @staticmethod
    def process_input_images(
        input1: np.ndarray, input2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return (
            ConfigurableDataflow.process_input_image(input1),
            ConfigurableDataflow.process_input_image(input2),
        )

    @staticmethod
    def merge_seperate_input_images(
        input_flash: np.ndarray, input_env: np.ndarray, flash_strength: float = 1.0
    ) -> np.ndarray:
        if input_flash.shape[-1] > 3:
            flashMask = input_flash[:, :, 3:4]
            input_flash = input_flash[:, :, :3] * flashMask

        return input_env + input_flash * flash_strength

    def __iter__(self):
        self.rng.shuffle(self.data)
        for path in self.data:
            res = []
            if self.add_path:
                res.append(path)
            yield res + [
                ConfigurableDataflow._parameter_to_image(path, p, self.processing_step)
                for p in self.parameters
            ]


class ShapeDataflow(ProxyDataFlow):
    def __init__(self, train_path: str, add_path: bool = False):
        parameters = [
            ParameterNames.INPUT_1_FLASH,
            ParameterNames.INPUT_1_ENV,
            ParameterNames.INPUT_2,
            ParameterNames.MASK,
            ParameterNames.NORMAL,
            ParameterNames.DEPTH,
        ]
        self.add_path = add_path
        self.ds = ConfigurableDataflow(train_path, parameters, add_path=add_path)

    def __iter__(self):
        for dp in self.ds:
            if self.add_path:
                path, dp = dp[0], dp[1:]
            input1 = ConfigurableDataflow.merge_seperate_input_images(dp[0], dp[1])
            input2 = dp[2]
            input1, input2 = ConfigurableDataflow.process_input_images(input1, input2)

            remDp = dp[3:]
            ret = [input1, input2] + remDp
            if self.add_path:
                yield (path, ret)
            else:
                yield ret


class IlluminationDataflow(ProxyDataFlow):
    def __init__(self, train_path: str, add_path: bool = False):
        parameters = [
            ParameterNames.INPUT_1_FLASH,
            ParameterNames.INPUT_1_ENV,
            ParameterNames.INPUT_2,
            ParameterNames.MASK,
            ParameterNames.NORMAL_PRED,
            ParameterNames.DEPTH_PRED,
            ParameterNames.SGS,
        ]
        self.add_path = add_path
        self.ds = ConfigurableDataflow(train_path, parameters, add_path=add_path)

    def __iter__(self):
        for dp in self.ds:
            if self.add_path:
                path, dp = dp[0], dp[1:]
            input1 = ConfigurableDataflow.merge_seperate_input_images(dp[0], dp[1])
            input2 = dp[2]
            input1, input2 = ConfigurableDataflow.process_input_images(input1, input2)

            remDp = dp[3:]
            ret = [input1, input2] + remDp
            if self.add_path:
                yield (path, ret)
            else:
                yield ret


class BRDFDataflow(ProxyDataFlow):
    def __init__(self, train_path: str, add_path: bool = False):
        parameters = [
            ParameterNames.INPUT_1_FLASH,
            ParameterNames.INPUT_1_ENV,
            ParameterNames.INPUT_2,
            ParameterNames.MASK,
            ParameterNames.NORMAL_PRED,
            ParameterNames.DEPTH_PRED,
            ParameterNames.SGS_PRED,
            ParameterNames.DIFFUSE,
            ParameterNames.SPECULAR,
            ParameterNames.ROUGHNESS,
        ]
        self.add_path = add_path
        self.ds = ConfigurableDataflow(train_path, parameters, add_path=add_path)

    def __iter__(self):
        for dp in self.ds:
            if self.add_path:
                path, dp = dp[0], dp[1:]
            input1 = ConfigurableDataflow.merge_seperate_input_images(dp[0], dp[1])
            input2 = dp[2]
            input1, input2 = ConfigurableDataflow.process_input_images(input1, input2)

            remDp = dp[3:]

            ret = [input1, input2] + remDp
            if self.add_path:
                yield (path, ret)
            else:
                yield ret


class JointDataflow(ProxyDataFlow):
    def __init__(self, train_path: str, add_path: bool = False):
        parameters = [
            ParameterNames.INPUT_1_FLASH,
            ParameterNames.INPUT_1_ENV,
            ParameterNames.RERENDER,  # Rerender
            ParameterNames.DIFFUSE_PRED,
            ParameterNames.SPECULAR_PRED,
            ParameterNames.ROUGHNESS_PRED,
            ParameterNames.NORMAL_PRED,
            ParameterNames.DEPTH_PRED,
            ParameterNames.MASK,
            ParameterNames.SGS_PRED,  # Previous predictions
            ParameterNames.DIFFUSE,
            ParameterNames.SPECULAR,
            ParameterNames.ROUGHNESS,
            ParameterNames.NORMAL,
            ParameterNames.DEPTH,  # GT parameters
        ]
        self.add_path = add_path
        self.ds = ConfigurableDataflow(train_path, parameters, add_path=add_path)

    def __iter__(self):
        for dp in self.ds:
            if self.add_path:
                path, dp = dp[0], dp[1:]
            input1 = ConfigurableDataflow.merge_seperate_input_images(dp[0], dp[1])
            input1 = ConfigurableDataflow.process_input_image(input1)

            remDp = dp[2:]
            ret = [input1] + remDp
            if self.add_path:
                yield (path, ret)
            else:
                yield ret


class Dataflows(Enum):
    SHAPE = ShapeDataflow
    ILLUMINATION = IlluminationDataflow
    BRDF = BRDFDataflow
    JOINT = JointDataflow


class InferenceStage(Enum):
    SHAPE = auto()
    ILLUMINATION = auto()
    BRDF = auto()
    INITIAL_RENDERING = auto()
    JOINT = auto()
    FINAL_RENDERING = auto()


class InferenceDataflow(ProxyDataFlow):
    def __init__(
        self,
        train_path: str,
        step: InferenceStage,
        isSrgb: bool = True,
        isLdr: bool = True,
    ):

        parameters = [
            ParameterNames.INPUT_1_LDR if isLdr else ParameterNames.INPUT_1,
            ParameterNames.INPUT_2_LDR if isLdr else ParameterNames.INPUT_2,
            ParameterNames.MASK,
        ]  # This is needed for every dataflow
        self.shots = 2
        if step == InferenceStage.SHAPE:
            # Nothing to add
            stage = Stages.INITIAL
        elif step == InferenceStage.ILLUMINATION:
            parameters.extend([ParameterNames.NORMAL_PRED, ParameterNames.DEPTH_PRED])
            stage = Stages.INITIAL
        elif step == InferenceStage.BRDF:
            parameters.extend(
                [
                    ParameterNames.NORMAL_PRED,
                    ParameterNames.DEPTH_PRED,
                    ParameterNames.SGS_PRED,
                ]
            )
            stage = Stages.INITIAL
        elif step == InferenceStage.JOINT:
            # This is different. Replace list
            parameters = [
                ParameterNames.INPUT_1_LDR if isLdr else ParameterNames.INPUT_1,
                ParameterNames.RERENDER,
                ParameterNames.DIFFUSE_PRED,
                ParameterNames.SPECULAR_PRED,
                ParameterNames.ROUGHNESS_PRED,
                ParameterNames.NORMAL_PRED,
                ParameterNames.DEPTH_PRED,
                ParameterNames.MASK,
                ParameterNames.SGS_PRED,
            ]
            stage = Stages.INITIAL
            self.shots = 1
        else:
            # Rendering
            parameters = [
                ParameterNames.DIFFUSE_PRED,
                ParameterNames.SPECULAR_PRED,
                ParameterNames.ROUGHNESS_PRED,
                ParameterNames.NORMAL_PRED,
                ParameterNames.DEPTH_PRED,
                ParameterNames.MASK,
                ParameterNames.SGS_PRED,
            ]
            stage = (
                Stages.INITIAL
                if step == InferenceStage.INITIAL_RENDERING
                else Stages.REFINEMENT
            )
            self.shots = 0

        self.isSrgb = isSrgb
        self.isLdr = isLdr
        self.ds = ConfigurableDataflow(
            train_path, parameters, processing_step=stage.value, add_path=True
        )

    def __iter__(self):
        for dp in self.ds:
            path = dp[0]
            dp = dp[1:]
            if self.shots == 2:
                if not self.isLdr:
                    input1, input2 = ConfigurableDataflow.process_input_images(
                        dp[0], dp[1]
                    )
                else:
                    input1, input2 = dp[0:2]

                if self.isSrgb:
                    input1 = sRGBToLinear(input1)
                    input2 = sRGBToLinear(input2)

                remDp = dp[2:]
                yield (path, [input1, input2] + remDp)
            elif self.shots == 1:
                if not self.isLdr:
                    input1 = ConfigurableDataflow.process_input_image(dp[0])
                else:
                    input1 = dp[0]

                if self.isSrgb:
                    input1 = sRGBToLinear(input1)

                remDp = dp[1:]
                yield (path, [input1] + remDp)
            else:
                yield (path, dp)


def get_data(
    dataflow: Dataflows,
    dataset_path: str,
    batch_size: int,
    n_proc: int = 12,
    n_gpus: int = 1,
):
    df = dataflow.value

    ds = df(dataset_path)
    ds = PrefetchDataZMQ(ds, nr_proc=n_proc, hwm=n_gpus * batch_size * 2)
    ds = BatchData(ds, batch_size, use_list=True)

    return ds
