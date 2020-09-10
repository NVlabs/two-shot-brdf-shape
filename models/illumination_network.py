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
import pickle
import sys
from math import log2
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
    SimpleTrainer,
    StagingInput,
    SyncMultiGPUTrainerReplicated,
    TrainConfig,
    get_model_loader,
    launch_train_with_config,
)
from tensorpack.models import (
    BatchNorm,
    BNReLU,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    FullyConnected,
    InstanceNorm,
    layer_register,
)
from tensorpack.tfutils import gradproc, optimizer
from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.utils import logger
from tensorpack.utils.gpu import change_gpu, get_num_gpu

import dataflow.dataflow as df
import utils.rendering_layer as rl
import utils.sg_utils as sg
import utils.tb_utils as tbutil
from models.base import (
    BaseNetwork,
    BaseTrainer,
    graySquareImgShape,
    rgbSquareImgShape,
    sphericalGaussainsShape,
)
from utils.common_layers import INReLU, normalize
from utils.config import ParameterNames
from utils.losses import *

EPS = 1e-7
MAX_VAL = 2


class IlluminationNetwork(BaseNetwork):
    def __init__(self, imgSize: int = 256, base_nf: int = 16, num_sgs: int = 24):
        self.imgSize = imgSize
        self.base_nf = base_nf
        self.num_sgs = num_sgs

        self.axis_sharpness = tf.convert_to_tensor(
            sg.setup_axis_sharpness(num_sgs), dtype=np.float32
        )

    def inputs(self):
        return [
            rgbSquareImgShape(self.imgSize, "cam1"),  # Cam 1
            rgbSquareImgShape(self.imgSize, "cam2"),  # Cam 2
            graySquareImgShape(self.imgSize, "mask"),  # Mask
            rgbSquareImgShape(self.imgSize, "normal"),  # Previous Estimated Normal
            graySquareImgShape(self.imgSize, "depth"),  # Previous Estimated Depth
            sphericalGaussainsShape(self.num_sgs, "sgs_gt"),  # GT SGs
        ]

    def outputs(self) -> List[str]:
        return ["env_net/predictions/sgs"]

    def output_image_names(self) -> List[str]:
        return [p.value for p in [ParameterNames.SGS_PRED]]

    def network_architecture(self, *args):
        cam1, cam2, mask, normal, depth = args
        with tf.variable_scope("env_net"):
            # Prepare the sgs pca components
            layers_needed = int(log2(cam1.shape[1].value) - 2)
            batch_size = tf.shape(cam1)[0]
            with argscope(
                [Conv2D, Conv2DTranspose, BatchNorm], data_format="channels_last"
            ):
                l = tf.concat([cam1, cam2, mask[:, :, :, 0:1], normal, depth], axis=-1)

                with tf.variable_scope("enc"):
                    for i in range(layers_needed):
                        l = Conv2D(
                            "conv%d" % (i + 1),
                            l,
                            min(self.base_nf * (2 ** i), 256),
                            4,
                            strides=2,
                            activation=INReLU,
                        )

                    encoded = tf.identity(l, "encoded")

                with tf.variable_scope("env_map"):
                    sgs = Conv2D(
                        "conv1", encoded, 256, 3, strides=2, activation=tf.nn.relu
                    )
                    sgs = Conv2D("conv2", sgs, 512, 3, strides=2, activation=tf.nn.relu)

                    sgs = tf.layers.Flatten()(sgs)

                    sgs = FullyConnected("fc1", sgs, 256, activation=tf.nn.relu)
                    sgs = Dropout("drop", sgs, 0.75)  # This is the keep prop
                    outputSize = self.num_sgs * 3
                    sgs = FullyConnected(
                        "fc2", sgs, outputSize, activation=tf.nn.sigmoid
                    )

            with tf.variable_scope("predictions"):
                sgs = tf.identity(
                    tf.reshape(sgs * MAX_VAL, [-1, self.num_sgs, 3]), name="sgs",
                )

            return sgs

    def build_graph(
        self,
        cam1: tf.Tensor,
        cam2: tf.Tensor,
        mask: tf.Tensor,
        normal: tf.Tensor,
        depth: tf.Tensor,
        sgs_gt: tf.Tensor,
    ):
        print("Building graph")
        sgs = self.network_architecture(cam1, cam2, mask, normal, depth)

        with tf.variable_scope("sgs_prep"):
            batch_size = tf.shape(cam1)[0]
            axis_sharpness = tf.tile(
                tf.expand_dims(self.axis_sharpness, 0), tf.stack([batch_size, 1, 1])
            )  # add batch dim

            sgs_joined = tf.concat([sgs, axis_sharpness], -1, name="sgs")
            print(
                "Joined pred shape", sgs_joined.shape, sgs.shape, axis_sharpness.shape
            )
            if self.training:
                sgs_gt = tf.clip_by_value(sgs_gt, 0.0, MAX_VAL)
                sgs_gt_joined = tf.concat([sgs_gt, axis_sharpness], -1, name="sgs_gt")
                print("Joined GT shape", sgs_gt_joined.shape, sgs_gt.shape)

        with tf.variable_scope("loss"):
            with tf.variable_scope("sgs"):
                print("SGS Loss shapes (gt, pred):", sgs_gt.shape, sgs.shape)
                sgs_loss = tf.reduce_mean(l2_loss(sgs_gt, sgs), name="sgs_loss")
                print("sgs_loss", sgs_loss.shape)
                add_moving_summary(sgs_loss)
                tf.losses.add_loss(sgs_loss, tf.GraphKeys.LOSSES)

        with tf.variable_scope("viz"):
            renderer = rl.RenderingLayer(60, 0.7, tf.TensorShape([None, 256, 256, 3]))
            sg_output = tf.zeros([batch_size, 256, 512, 3])
            renderer.visualize_sgs(sgs_joined, sg_output)

            if self.training:
                sg_gt_output = tf.zeros_like(sg_output)
                renderer.visualize_sgs(sgs_gt_joined, sg_gt_output, "sgs_gt")

        print(sgs_loss)
        self.cost = tf.losses.get_total_loss(name="total_costs")
        print(self.cost)

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


class InferenceModel(IlluminationNetwork):
    def inputs(self):
        return [
            rgbSquareImgShape(self.imgSize, "input1"),  # Cam1
            rgbSquareImgShape(self.imgSize, "input2"),  # Cam2
            graySquareImgShape(self.imgSize, "mask"),  # Mask
            rgbSquareImgShape(self.imgSize, "normal"),  # Normal
            graySquareImgShape(self.imgSize, "depth"),  # Depth
        ]

    def build_graph(
        self,
        cam1: tf.Tensor,
        cam2: tf.Tensor,
        mask: tf.Tensor,
        normal: tf.Tensor,
        depth: tf.Tensor,
    ):
        with tf.variable_scope("prepare"):
            mask = tf.where(tf.less(mask, 0.9), tf.zeros_like(mask), tf.ones_like(mask))
            repeat = [1 for _ in range(len(mask.shape))]
            repeat[-1] = 3

            mask3 = tf.tile(mask, repeat)

        self.network_architecture(self, cam1, cam2, mask3, normal, depth)


class IlluminationTrainer(BaseTrainer):
    def __init__(self, subparser):
        super().__init__(df.Dataflows.ILLUMINATION, "illumination", subparser)

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
            default=16,
            help="Number of base features to use. Number of features for each layers are based on this.",
        )
        parser.add_argument(
            "--num_sgs",
            type=int,
            default=24,
            help="The numbers of Spherical Gaussians amplitudes to predict.",
        )

    def train_model(self, args) -> BaseNetwork:
        return IlluminationNetwork(
            imgSize=args.image_size, base_nf=args.base_nf, num_sgs=args.num_sgs,
        )

    def inference_model(self, args) -> BaseNetwork:
        return InferenceModel(
            imgSize=args.image_size, base_nf=args.base_nf, num_sgs=args.num_sgs,
        )
