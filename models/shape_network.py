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
import sys
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorpack import (
    Callback,
    ModelDesc,
    ModelSaver,
    QueueInput,
    SaverRestore,
    ScheduledHyperParamSetter,
    StagingInput,
    TrainConfig,
    get_model_loader,
)
from tensorpack.models import (
    BatchNorm,
    BNReLU,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    FullyConnected,
)
from tensorpack.tfutils import gradproc, optimizer
from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.utils import logger
from tensorpack.utils.gpu import change_gpu, get_num_gpu

import dataflow.dataflow as df
from models.base import BaseNetwork, BaseTrainer, graySquareImgShape, rgbSquareImgShape
from utils.common_layers import Fusion2DBlock, INReLU, normalize, uncompressDepth
from utils.config import ParameterNames
from utils.losses import *


class GeomNetwork(BaseNetwork):
    def __init__(
        self,
        imgSize: int = 256,
        base_nf: int = 32,
        downscale_steps: int = 4,
        consistency_loss: float = 0.5,
    ):
        self.imgSize = imgSize
        self.base_nf = base_nf
        self.downscale_steps = downscale_steps
        self.consistency_loss = consistency_loss
        self.enable_consistency = consistency_loss != 0.0

    def inputs(self):
        return [
            rgbSquareImgShape(self.imgSize, "cam1"),  # Cam 1
            rgbSquareImgShape(self.imgSize, "cam2"),  # Cam 2
            graySquareImgShape(self.imgSize, "mask"),  # Mask
            rgbSquareImgShape(self.imgSize, "normal_gt"),  # Normal
            graySquareImgShape(self.imgSize, "depth_gt"),  # Depth
        ]

    def outputs(self) -> List[str]:
        return ["geom_net/predictions/normal", "geom_net/predictions/depth"]

    def output_image_names(self) -> List[str]:
        return [
            p.value for p in [ParameterNames.NORMAL_PRED, ParameterNames.DEPTH_PRED]
        ]

    def network_architecture(self, *args):
        cam1, cam2, mask = args
        base_nf = self.base_nf
        with tf.variable_scope("geom_net"):
            with argscope(
                [Conv2D, Conv2DTranspose, BatchNorm], data_format="channels_last"
            ):
                lmain = tf.concat([cam1, mask[:, :, :, 0:1]], axis=-1)
                lsupp = tf.concat([cam2, mask[:, :, :, 0:1]], axis=-1)
                lfusion = None

                with tf.variable_scope("enc"):
                    for i in range(self.downscale_steps):
                        lmain, lsupp, lfusion = Fusion2DBlock(
                            "fconv%d" % (i + 1),
                            (lmain, lsupp, lfusion),
                            min(base_nf * (2 ** i), 256),
                            4,
                            2,
                            True,
                            INReLU,
                        )

                with tf.variable_scope("dec"):
                    for i in range(self.downscale_steps):
                        inv_i = self.downscale_steps - i
                        nf = min(base_nf * (2 ** (inv_i - 1)), 256)

                        lmain, lsupp, lfusion = Fusion2DBlock(
                            "ftconv%d" % (i + 1),
                            (lmain, lsupp, lfusion),
                            nf,
                            4,
                            2,
                            False,
                            INReLU,
                        )

                geom_estimation = Conv2D(
                    "output", lfusion, 4, 5, activation=tf.nn.sigmoid
                )

            with tf.variable_scope("predictions"):
                repeat = [1 for _ in range(len(mask.shape))]
                repeat[-1] = 3

                mask3 = tf.tile(mask[:, :, :, 0:1], repeat)

                normal = tf.identity(
                    (normalize(geom_estimation[:, :, :, 0:3] * 2 - 1) * 0.5 + 0.5)
                    * mask3,
                    "normal",
                )
                depth = tf.identity(
                    geom_estimation[:, :, :, 3:4] * mask[:, :, :, 0:1], "depth"
                )

            return normal, depth

    def build_graph(
        self,
        cam1: tf.Tensor,
        cam2: tf.Tensor,
        mask: tf.Tensor,
        normal_gt: tf.Tensor,
        depth_gt: tf.Tensor,
    ):
        with tf.variable_scope("prepare"):
            repeat = [1 for _ in range(len(mask.shape))]
            repeat[-1] = 3

            tbutil.two_side_by_side(cam1, cam2, "input", 5)
            mask3 = tf.tile(mask, repeat)

        normal, depth = self.network_architecture(cam1, cam2, mask)

        with tf.variable_scope("loss"):
            with tf.variable_scope("normal"):
                normal_loss = tf.reduce_mean(
                    masked_loss(l1_loss(normal_gt * 2 - 1, normal * 2 - 1), mask3),
                    name="normal_loss",
                )
                add_moving_summary(normal_loss)
                tf.losses.add_loss(normal_loss, tf.GraphKeys.LOSSES)
                tbutil.two_side_by_side(normal_gt, normal, "normal", 5)

            with tf.variable_scope("depth"):
                depth_loss = tf.reduce_mean(
                    masked_loss(l1_loss(depth_gt, depth), mask), name="depth_loss"
                )
                add_moving_summary(depth_loss)
                tf.losses.add_loss(depth_loss, tf.GraphKeys.LOSSES)
                tbutil.two_side_by_side(depth_gt, depth, "depth", 5)

            if self.enable_consistency:
                with tf.variable_scope("consistency"):

                    near = uncompressDepth(1)
                    far = uncompressDepth(0)
                    d = uncompressDepth(depth)
                    h = tf.div_no_nan(d - near, far - near)

                    sobel = tf.image.sobel_edges(
                        h
                    )  # b,h,w,1,[dy,dx] - 1 because height has 1 channel
                    dx = sobel[:, :, :, :, 1]  # b,h,w,1
                    dy = -sobel[:, :, :, :, 0]
                    # We're using a depth map instead of a height. Which means bright
                    # values are at a greater depth. Thus, we need to invert the gradient
                    texelSize = 1 / self.imgSize
                    dz = tf.ones_like(dx) * texelSize * 2

                    n = normalize(tf.concat([dx, dy, dz], -1))
                    n = n * 0.5 + 0.5
                    consistency = masked_loss(l2_loss(n, normal), mask3)

                    consistency_loss = (
                        tf.reduce_mean(consistency, name="consistency_loss")
                        * self.consistency_loss
                    )
                    add_moving_summary(consistency_loss)
                    tf.losses.add_loss(consistency_loss, tf.GraphKeys.LOSSES)

                    tbutil.four_side_by_side(
                        tf.tile(depth, repeat), n, normal, consistency, "consistency", 5
                    )

        self.cost = tf.losses.get_total_loss(name="total_costs")

        add_moving_summary(self.cost)
        add_param_summary((".*/W", ["histogram"]))  # monitor W

        return self.cost

    def optimizer(self):
        lr = tf.get_variable("learning_rate", initializer=0.0002, trainable=False)
        tf.summary.scalar("learning_rate", lr)
        opt = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999)
        return optimizer.apply_grad_processors(
            opt, [gradproc.SummaryGradient(), gradproc.CheckGradient()]
        )


class InferenceModel(GeomNetwork):
    def inputs(self):
        return [
            rgbSquareImgShape(self.imgSize, "input1"),  # Cam1
            rgbSquareImgShape(self.imgSize, "input2"),  # Cam2
            graySquareImgShape(self.imgSize, "mask"),  # Mask
        ]

    def build_graph(self, cam1: tf.Tensor, cam2: tf.Tensor, mask: tf.Tensor):
        from tensorflow.python.framework import tensor_shape

        mask = tf.where(tf.less(mask, 0.9), tf.zeros_like(mask), tf.ones_like(mask))

        normal, depth = self.network_architecture(cam1, cam2, mask)


class ShapeTrainer(BaseTrainer):
    def __init__(self, subparser):
        super().__init__(df.Dataflows.SHAPE, "shape", subparser)

    def _network_specific_callbacks(self):
        self.callbacks.append(
            ScheduledHyperParamSetter(
                "learning_rate", [(1, 0.0002), (int(self.args.epochs / 2), 0.0001)]
            )
        )

    def _network_specific_args(self, parser):
        parser.add_argument(
            "--base_nf",
            type=int,
            default=32,
            help="Number of base features to use. Number of features for each layers are based on this.",
        )
        parser.add_argument(
            "--downscale_steps",
            type=int,
            default=4,
            help="Number of downscaling and upscaling steps to use.",
        )
        parser.add_argument(
            "--consistency_loss",
            type=float,
            default=0.5,
            help="Set the weight of the consistency loss. A value of 0 disables the loss.",
        )

        return parser

    def train_model(self, args) -> BaseNetwork:
        return GeomNetwork(
            imgSize=args.image_size,
            base_nf=args.base_nf,
            downscale_steps=args.downscale_steps,
            consistency_loss=args.consistency_loss,
        )

    def inference_model(self, args) -> BaseNetwork:
        return InferenceModel(
            imgSize=args.image_size,
            base_nf=args.base_nf,
            downscale_steps=args.downscale_steps,
            consistency_loss=args.consistency_loss,
        )
